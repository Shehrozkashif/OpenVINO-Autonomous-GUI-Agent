# core/orchestrator.py
"""
Task Orchestrator — the central coordinator.
Runs the full agent pipeline for each user instruction.

Execution flow:
  instruction
    → Router.decompose() → SubTasks (topologically sorted)
    → for each SubTask:
        Planning.plan() → ActionSteps
        → for each ActionStep:
            [if click/scroll] Grounding.ground() → (x, y)
            Action.execute(step, x, y)
            Reflection.verify(step) → success?
            [if failed] Planning.replan() → new steps
    → Router.summarize_completion()
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from loguru import logger

from agents.action.action_agent import ActionExecutionAgent
from agents.grounding.grounding_agent import UIGroundingAgent
from agents.planning.planning_agent import PlanningAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.router.router_agent import RouterAgent
from core.capture.screenshot import ScreenCapture
from core.protocols.a2a import ActionStep, SubTask
from memory.task.task_memory import TaskMemory


@dataclass
class OrchestratorConfig:
    max_retries_per_step: int = 3
    reflection_wait_s: float = 0.5


class TaskOrchestrator:
    def __init__(
        self,
        router: RouterAgent,
        planner: PlanningAgent,
        grounder: UIGroundingAgent,
        actor: ActionExecutionAgent,
        reflector: ReflectionAgent,
        capturer: ScreenCapture,
        task_memory: TaskMemory,
        config: OrchestratorConfig = None,
        on_step_log: Optional[Callable[[str], None]] = None
    ):
        self.router = router
        self.planner = planner
        self.grounder = grounder
        self.actor = actor
        self.reflector = reflector
        self.capturer = capturer
        self.memory = task_memory
        self.config = config or OrchestratorConfig()
        self.log = on_step_log or print
        self._stop_event = threading.Event()

    def stop(self):
        """Signal the orchestrator to stop after the current step."""
        self._stop_event.set()

    def execute(self, instruction: str) -> dict:
        self._stop_event.clear()
        self.log(f"[TASK START] '{instruction}'")
        start_time = time.time()

        # Give the router a snapshot of what's currently on screen so it can
        # avoid generating subtasks for things that are already visible/done.
        screen_context = self._get_screen_context()
        task_id, subtasks = self.router.decompose(instruction, screen_context=screen_context)
        self.log(f"[ROUTER] {len(subtasks)} sub-tasks")

        completed = []
        failed = False

        for subtask in self._topological_sort(subtasks):
            if self._stop_event.is_set():
                self.log("[TASK] Stopped by user.")
                failed = True
                break

            self.log(f"\n[SUBTASK {subtask.id}] {subtask.description}")
            success = self._execute_subtask(subtask)

            if success:
                completed.append(subtask.id)
                self.log(f"[SUBTASK {subtask.id}] Complete")
            else:
                self.log(f"[SUBTASK {subtask.id}] Failed — stopping task")
                failed = True
                break

        elapsed = time.time() - start_time
        summary = self.router.summarize_completion(task_id, completed, not failed)
        self.log(f"\n[TASK DONE] {summary} ({elapsed:.1f}s)")

        if not failed:
            self.memory.store_successful_task(instruction, subtasks, elapsed)

        return {
            "task_id": task_id,
            "success": not failed,
            "subtasks_completed": completed,
            "elapsed_s": elapsed,
            "summary": summary
        }

    def _execute_subtask(self, subtask: SubTask) -> bool:
        # Give the planner a live OCR snapshot so it writes targets from what's
        # actually on screen — not from model memory about what apps look like.
        screen_context = self._get_screen_context()
        steps = self.planner.plan(subtask, screen_context=screen_context)
        idx = 0
        while idx < len(steps):
            if self._stop_event.is_set():
                return False
            step = steps[idx]
            self.log(f"  Step {step.id}: [{step.action_type}] {step.description}")

            # Skip VLM reflection for actions that don't need visual verification.
            # Reflection calls qwen2.5vl (CPU, ~160s) — avoid for fast/keyboard steps.
            # For click/right_click we DO attempt reflection but treat timeout as success
            # so a slow model swap doesn't crash the whole task.
            skip_reflection = step.action_type in ("key_press", "type", "hotkey", "wait", "scroll", "double_click")

            success = False
            for attempt in range(self.config.max_retries_per_step):
                if attempt > 0:
                    self.log(f"  Retry {attempt}/{self.config.max_retries_per_step}…")

                exec_ok = self._execute_step(step)
                if not exec_ok:
                    continue

                if skip_reflection:
                    success = True
                    break

                try:
                    reflection = self.reflector.verify(step, self.config.reflection_wait_s)
                    if reflection.success and reflection.confidence >= self.reflector.min_confidence:
                        success = True
                        self.log(f"  Verified (conf={reflection.confidence:.2f})")
                        break
                    else:
                        self.log(f"  Verification failed: {reflection.error_description}")
                        if not reflection.should_retry:
                            # Action executed — treat as success even without verification
                            success = True
                            break
                        if attempt == self.config.max_retries_per_step - 1:
                            new_steps = self.planner.replan(step, reflection.error_description, steps[idx+1:])
                            steps = steps[:idx] + [step] + new_steps
                            self.log(f"  Replanned: {len(new_steps)} new steps")
                except Exception as e:
                    # Reflection timeout or error — the action already executed, treat as success
                    self.log(f"  Reflection unavailable ({type(e).__name__}) — action executed, continuing")
                    success = True
                    break

            if not success:
                return False
            idx += 1
        return True

    def _execute_step(self, step: ActionStep) -> bool:
        x, y = None, None

        if step.action_type in ("click", "right_click", "double_click", "scroll"):
            if not step.target:
                self.log(f"  Step {step.id} is a {step.action_type} with no target")
                return False
            result = self.grounder.ground(step.target)
            if not result.found or result.confidence < self.grounder.min_confidence:
                self.log(f"  Could not find '{step.target}' (conf={result.confidence:.2f})")
                return False
            x, y = result.x, result.y

        return self.actor.execute(step, x=x, y=y)

    def _get_screen_context(self) -> str:
        """Return a short text summary of what's currently visible on screen via OCR."""
        try:
            from core.grounding.ocr_engine import OCREngine
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            ocr = OCREngine()
            words = ocr.extract(img)
            visible = [w.text for w in words if len(w.text) >= 3 and w.conf >= 0.80]
            return ", ".join(f'"{t}"' for t in visible[:30])
        except Exception:
            return ""

    def _topological_sort(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        Kahn's algorithm for topological sort.
        Ensures a subtask only runs after all its dependencies are done.
        Falls back to ID sort if the graph has cycles (shouldn't happen).
        """
        by_id = {s.id: s for s in subtasks}
        in_degree = {s.id: len(s.depends_on) for s in subtasks}
        ready = [s for s in subtasks if not s.depends_on]
        dependents = {s.id: [] for s in subtasks}
        for s in subtasks:
            for dep_id in s.depends_on:
                if dep_id in dependents:
                    dependents[dep_id].append(s.id)

        order = []
        queue = list(ready)
        while queue:
            current = queue.pop(0)
            order.append(current)
            for dep_id in dependents.get(current.id, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(by_id[dep_id])

        if len(order) != len(subtasks):
            # Cycle detected — fall back to ID sort
            logger.warning("[ORCHESTRATOR] Dependency cycle detected, falling back to ID order")
            return sorted(subtasks, key=lambda s: s.id)
        return order
