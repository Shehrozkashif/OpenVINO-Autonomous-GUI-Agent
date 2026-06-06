# core/orchestrator.py
"""
Task Orchestrator — the central coordinator.

Execution flow:
  instruction
    → Router.decompose() → SubTasks (topologically sorted)
    → for each SubTask (dynamic loop, sees live screen every step):
        screen_context + task_context → plan_next_step() → ActionStep
        [click/drag]    Grounding.ground() → (x, y)
        [grounding miss] _scroll_to_find() → retry ≤ max_scroll_find_attempts×
        [extract]        _extract_data()   → LLM reads OCR → stores value
        Action.execute(step)
        Reflection.verify(step) → success?
        failure embedded in history → next plan_next_step sees it and recovers
    → Router.summarize_completion()
    → return extracted_data alongside success/failure
"""
import platform
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

_OS = platform.system()

from loguru import logger

from agents.action.action_agent import ActionExecutionAgent
from agents.grounding.grounding_agent import GroundingResult, UIGroundingAgent
from agents.planning.planning_agent import PlanningAgent
from agents.reflection.reflection_agent import ReflectionAgent
from agents.router.router_agent import RouterAgent
from core.capture.screenshot import ScreenCapture, _screen_size
from core.protocols.a2a import ActionStep, SubTask
from memory.task.task_memory import TaskMemory


@dataclass
class OrchestratorConfig:
    max_retries_per_step: int = 3
    reflection_wait_s: float = 0.5
    max_steps_per_subtask: int = 20
    max_scroll_find_attempts: int = 6   # scrolls before giving up on an off-screen element
    consecutive_failures_limit: int = 3


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
        on_step_log: Optional[Callable[[str], None]] = None,
    ):
        self.router    = router
        self.planner   = planner
        self.grounder  = grounder
        self.actor     = actor
        self.reflector = reflector
        self.capturer  = capturer
        self.memory    = task_memory
        self.config    = config or OrchestratorConfig()
        self.log       = on_step_log or print
        self._stop_event = threading.Event()

        from agents.grounding.grounding_agent import OCREngine
        self._ocr = OCREngine()
        self._screen_w, self._screen_h = _screen_size()
        self._extracted_data: Dict[str, str] = {}

    def stop(self):
        self._stop_event.set()

    # ── Public entry point ────────────────────────────────────────────────────

    def execute(self, instruction: str) -> dict:
        self._stop_event.clear()
        self._extracted_data = {}
        self.log(f"[TASK START] '{instruction}'")
        start_time = time.time()

        # Memory hint for the router
        memory_hint = None
        try:
            similar = self.memory.find_similar(instruction, threshold=0.80)
            if similar:
                descs = [s.get("description", "") for s in similar.get("steps", [])]
                memory_hint = (
                    f"Similar past task (succeeded): '{similar['instruction']}'. "
                    f"Subtasks used: {descs}. Reuse if it fits."
                )
                self.log(f"[MEMORY] Similar past task found (sim={similar['similarity']:.2f})")
        except Exception:
            pass

        screen_context = self._get_screen_context()
        task_id, subtasks = self.router.decompose(
            instruction, screen_context=screen_context, memory_hint=memory_hint
        )
        self.log(f"[ROUTER] {len(subtasks)} sub-task(s)")

        completed_subtask_ids = []
        completed_subtask_descs: List[str] = []   # for inter-subtask context
        failed = False

        for subtask in self._topological_sort(subtasks):
            if self._stop_event.is_set():
                self.log("[TASK] Stopped by user.")
                failed = True
                break

            self.log(f"\n[SUBTASK {subtask.id}] {subtask.description}")
            success = self._execute_subtask(subtask, task_context=completed_subtask_descs)

            if success:
                success = self._verify_launch(subtask)

            if success:
                completed_subtask_ids.append(subtask.id)
                completed_subtask_descs.append(subtask.description)
                self.log(f"[SUBTASK {subtask.id}] Complete")
                if subtask != subtasks[-1]:
                    self._wait_for_settle(min_s=0.5, max_s=3.0)
            else:
                self.log(f"[SUBTASK {subtask.id}] Failed — stopping task")
                failed = True
                break

        elapsed = time.time() - start_time
        summary = self.router.summarize_completion(task_id, completed_subtask_ids, not failed)

        # Append any extracted data to the summary log
        if self._extracted_data:
            lines = "\n".join(f"  {k}: {v}" for k, v in self._extracted_data.items())
            self.log(f"\n[EXTRACTED DATA]\n{lines}")

        self.log(f"\n[TASK DONE] {summary} ({elapsed:.1f}s)")

        if not failed:
            self.memory.store_successful_task(instruction, subtasks, elapsed)

        return {
            "task_id": task_id,
            "success": not failed,
            "subtasks_completed": completed_subtask_ids,
            "elapsed_s": elapsed,
            "summary": summary,
            "extracted_data": dict(self._extracted_data),
        }

    # ── Subtask execution loop ────────────────────────────────────────────────

    def _execute_subtask(self, subtask: SubTask, task_context: List[str] = None) -> bool:
        """
        Dynamic loop — plans ONE step at a time using live screen state.

        task_context: summaries of subtasks completed before this one in the same task.
        Each step planner sees: goal + inter-subtask context + within-subtask history + screen.
        """
        completed: List[str] = []
        consecutive_failures = 0
        _cached_ocr: str = ""   # reuse reflection's OCR for next planning step

        for step_idx in range(self.config.max_steps_per_subtask):
            if self._stop_event.is_set():
                return False

            screen_context = _cached_ocr if _cached_ocr else self._get_screen_context()
            _cached_ocr = ""   # consume — will be refreshed after reflection

            # Pass failure hints so planner avoids repeating known-bad patterns
            failure_hints = []
            try:
                failure_hints = self.memory.get_failure_hints(subtask.description)
            except Exception:
                pass

            step = self.planner.plan_next_step(
                subtask, screen_context, completed,
                task_context=task_context, failure_hints=failure_hints,
            )

            if step is None:
                return True   # planner says goal is achieved

            self.log(f"  Step {step_idx + 1}: [{step.action_type}] {step.description}")

            skip_reflection = step.action_type in ("wait", "extract")
            step_success = False
            last_error = ""

            for attempt in range(self.config.max_retries_per_step):
                if attempt > 0:
                    self.log(f"  Retry {attempt}/{self.config.max_retries_per_step}…")

                exec_ok = self._execute_step(step)
                if not exec_ok:
                    last_error = "execution failed (element not found or action error)"
                    continue

                if skip_reflection:
                    step_success = True
                    break

                try:
                    reflection = self.reflector.verify(step, self.config.reflection_wait_s)
                    _cached_ocr = reflection.ocr_text   # reuse for next planning call
                    _key = (step.key or "").lower()
                    _is_launcher_key = step.action_type == "key_press" and _key in ("super", "enter")
                    fail_threshold = (
                        0.95
                        if step.action_type in ("type", "hotkey") or _is_launcher_key
                        else self.reflector.min_confidence
                    )
                    if reflection.success or reflection.confidence < fail_threshold:
                        step_success = True
                        label = "Verified" if reflection.success else "Uncertain→proceeding"
                        self.log(f"  {label} (conf={reflection.confidence:.2f})")
                        time.sleep(0.1)
                        break
                    else:
                        last_error = reflection.error_description
                        self.log(f"  Verification failed: {last_error}")
                        if not reflection.should_retry:
                            step_success = True
                            break
                except (ValueError, KeyError, AttributeError) as e:
                    # Model parse error — proceed optimistically
                    self.log(f"  Reflection parse error ({type(e).__name__}) — proceeding")
                    step_success = True
                    break
                except Exception as e:
                    # Infrastructure failure — retry with back-off
                    self.log(f"  Reflection error ({type(e).__name__}: {e}) — retry")
                    last_error = str(e)
                    time.sleep(1.0)

            if step_success:
                # Append extract result to step description for context
                if step.action_type == "extract":
                    key = step.target or step.description or f"item_{step.id}"
                    val = self._extracted_data.get(key, "")
                    completed.append(f"{step.description} → extracted: '{val}'")
                else:
                    completed.append(step.description)
                consecutive_failures = 0
            else:
                note = (
                    f"[FAILED: {last_error}] {step.description}"
                    if last_error else f"[FAILED] {step.description}"
                )
                completed.append(note)
                consecutive_failures += 1
                self.log(f"  Step failed — re-evaluating next action")

                # Persist failure so future tasks can avoid this pattern
                try:
                    self.memory.store_failure_pattern(
                        target=step.target or step.description or "",
                        action_type=step.action_type,
                        error=last_error[:200] if last_error else "",
                        app_context=screen_context[:100],
                    )
                except Exception:
                    pass

                if consecutive_failures >= self.config.consecutive_failures_limit:
                    self.log(
                        f"  {self.config.consecutive_failures_limit} consecutive failures"
                        f" — aborting subtask"
                    )
                    return False

        self.log(f"  MAX_STEPS ({self.config.max_steps_per_subtask}) reached")
        return False

    # ── Step execution ─────────────────────────────────────────────────────────

    def _execute_step(self, step: ActionStep) -> bool:
        x, y, x2, y2 = None, None, None, None

        # ── Click family ──────────────────────────────────────────────────────
        if step.action_type in ("click", "right_click", "double_click"):
            if not step.target:
                self.log(f"  {step.action_type} has no target")
                return False
            result = self.grounder.ground(step.target)
            if not result.found or result.confidence < self.grounder.min_confidence:
                result = self._scroll_to_find(step.target)
            if not result.found or result.confidence < self.grounder.min_confidence:
                self.log(f"  Could not find '{step.target}' (conf={result.confidence:.2f})")
                return False
            x, y = result.x, result.y

        # ── Drag ──────────────────────────────────────────────────────────────
        elif step.action_type == "drag":
            if not step.target:
                self.log(f"  drag has no source target")
                return False
            src = self.grounder.ground(step.target)
            if not src.found or src.confidence < self.grounder.min_confidence:
                self.log(f"  drag: could not find source '{step.target}'")
                return False
            x, y = src.x, src.y
            if step.value:
                # "x,y" pixel coords or element description
                if "," in step.value and all(
                    p.strip().lstrip("-").isdigit() for p in step.value.split(",", 1)
                ):
                    px, py = step.value.split(",", 1)
                    x2, y2 = int(px.strip()), int(py.strip())
                else:
                    dst = self.grounder.ground(step.value)
                    if not dst.found or dst.confidence < self.grounder.min_confidence:
                        self.log(f"  drag: could not find destination '{step.value}'")
                        return False
                    x2, y2 = dst.x, dst.y
            else:
                self.log(f"  drag has no destination (value empty)")
                return False

        # ── Scroll ────────────────────────────────────────────────────────────
        elif step.action_type == "scroll":
            if step.target:
                result = self.grounder.ground(step.target)
                if result.found and result.confidence >= self.grounder.min_confidence:
                    x, y = result.x, result.y
            # x/y = None → action_agent falls back to screen center

        # ── Extract ───────────────────────────────────────────────────────────
        elif step.action_type == "extract":
            value = self._extract_data(step)
            key = step.target or step.description or f"item_{step.id}"
            if value:
                self._extracted_data[key] = value
                self.log(f"  [EXTRACT] '{key}' = '{value[:80]}'")
            else:
                self.log(f"  [EXTRACT] nothing found for '{key}'")
            return True   # extraction never blocks the task

        return self.actor.execute(step, x=x, y=y, x2=x2, y2=y2)

    # ── Data extraction ────────────────────────────────────────────────────────

    def _extract_data(self, step: ActionStep) -> Optional[str]:
        """
        Use OCR + LLM to extract a specific value from the current screen.
        step.target describes what to extract  (e.g. "the error message",
                                                "the file path shown",
                                                "the page title")
        """
        try:
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            words = self._ocr.extract(img)
            if not words:
                return None
            # Build a readable OCR dump (keep lines, not just tokens)
            ocr_text = " | ".join(
                w.text for w in words if w.conf >= 0.65 and len(w.text) >= 2
            )
            what = step.target or step.description or "the most important value on screen"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract specific data from screen text. "
                        "Reply with ONLY the extracted value — no explanation, no quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Screen text:\n{ocr_text}\n\nExtract: {what}",
                },
            ]
            resp = self.reflector.ovms.query_llm(messages, max_tokens=120, temperature=0.0)
            value = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            return value or None
        except Exception as e:
            logger.warning(f"[EXTRACT] Error: {e}")
            return None

    # ── Scroll-to-find ────────────────────────────────────────────────────────

    def _scroll_to_find(self, target: str) -> GroundingResult:
        """
        Scroll down and retry grounding after each scroll.
        Stops early if the page stops changing (reached end of document).
        Resets scroll position if element is not found.
        """
        cx, cy = self._screen_w // 2, self._screen_h // 2
        scroll_step = ActionStep(
            id=0, subtask_id=0, action_type="scroll",
            target=None, value="down", key=None,
            description=f"scroll to find '{target}'", verification="",
        )
        scroll_up = ActionStep(
            id=0, subtask_id=0, action_type="scroll",
            target=None, value="up", key=None,
            description="scroll back up", verification="",
        )

        scrolls_done = 0
        prev_ocr_text = ""

        for i in range(self.config.max_scroll_find_attempts):
            self.log(
                f"  [SCROLL-FIND] '{target}' not visible — scroll {i + 1}/"
                f"{self.config.max_scroll_find_attempts}"
            )
            self.actor.execute(scroll_step, x=cx, y=cy)
            scrolls_done += 1
            time.sleep(0.35)

            # Check if page content is still changing
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            cur_text = " ".join(w.text for w in self._ocr.extract(img) if w.conf >= 0.6)
            if cur_text == prev_ocr_text and i > 0:
                self.log(f"  [SCROLL-FIND] Reached end of page — stopping")
                break
            prev_ocr_text = cur_text

            result = self.grounder.ground(target)
            if result.found and result.confidence >= self.grounder.min_confidence:
                self.log(f"  [SCROLL-FIND] Found '{target}' after {i + 1} scroll(s)")
                return result

        # Restore original scroll position so subsequent steps aren't confused
        for _ in range(scrolls_done):
            self.actor.execute(scroll_up, x=cx, y=cy)
            time.sleep(0.1)

        return GroundingResult(
            x=0, y=0, confidence=0.0, found=False,
            latency_ms=0, target=target, method="scroll_exhausted",
        )

    # ── App launch verification ────────────────────────────────────────────────
    # OS-specific OCR signals that confirm an app opened successfully.
    # Keys are lowercase fragments of the subtask description.

    _APP_SIGNALS_WINDOWS: dict = {
        "calculator":      ["Calculator", "Standard", "Scientific"],
        "notepad":         ["Notepad", "Untitled", "File"],
        "paint":           ["Paint", "Home", "Image"],
        "command prompt":  ["cmd", "C:\\", "Microsoft"],
        "powershell":      ["PowerShell", "PS", "Windows"],
        "terminal":        ["Terminal", "PowerShell", "cmd", "Windows"],
        "file explorer":   ["File Explorer", "This PC", "Documents", "Quick access"],
        "explorer":        ["File Explorer", "This PC", "Documents", "Quick access"],
        "edge":            ["Microsoft Edge", "New Tab", "Search"],
        "firefox":         ["Firefox", "Mozilla", "Search"],
        "chrome":          ["Google Chrome", "New Tab", "Search"],
        "vs code":         ["Explorer", "Extensions", "Welcome", "Visual Studio"],
        "visual studio":   ["Explorer", "Extensions", "Welcome", "Visual Studio"],
        "libreoffice":     ["Writer", "Calc", "Impress", "LibreOffice"],
        "thunderbird":     ["Thunderbird", "Inbox", "Compose"],
        "settings":        ["Settings", "System", "Bluetooth", "Windows Update"],
        "task manager":    ["Task Manager", "Processes", "CPU"],
        "snipping tool":   ["Snipping Tool", "New", "Mode"],
        "wordpad":         ["WordPad", "Home", "Document"],
    }

    _APP_SIGNALS_LINUX: dict = {
        "calculator":        ["Calculator", "AC", "DEG", "RAD"],
        "gnome calculator":  ["Calculator", "AC", "DEG", "RAD"],
        "terminal":          ["Terminal", "bash", "sh"],
        "gnome-terminal":    ["Terminal", "bash"],
        "firefox":        ["Firefox", "Mozilla", "Search"],
        "vs code":        ["Explorer", "Extensions", "Welcome"],
        "visual studio":  ["Explorer", "Extensions", "Welcome"],
        "libreoffice":    ["Writer", "Calc", "Impress", "LibreOffice"],
        "thunderbird":    ["Thunderbird", "Inbox", "Compose"],
        "settings":       ["Settings", "Wi-Fi", "Bluetooth", "Network"],
        "nautilus":       ["Files", "Home", "Documents"],
    }

    @property
    def _APP_SIGNALS(self) -> dict:
        return self._APP_SIGNALS_WINDOWS if _OS == "Windows" else self._APP_SIGNALS_LINUX

    def _verify_launch(self, subtask) -> bool:
        desc = subtask.description.lower()
        if "already open" in desc or "already running" in desc:
            return True
        if not any(w in desc for w in ("open", "launch", "search launcher")):
            return True

        matched_key = next(
            (k for k in self._APP_SIGNALS if k in desc), None
        )
        if matched_key is None:
            return True

        signals = self._APP_SIGNALS[matched_key]
        for attempt, wait_s in enumerate([1.5, 2.5]):
            time.sleep(wait_s)
            try:
                img = self.capturer.capture()
                img.thumbnail((960, 540))
                ocr_words = {w.text for w in self._ocr.extract(img) if w.conf >= 0.5}
                all_ocr_text = " ".join(ocr_words)
                if any(sig in all_ocr_text for sig in signals):
                    self.log(f"  [CHECK] '{matched_key}' confirmed on screen")
                    return True
                if attempt == 0:
                    self.log(f"  [CHECK] '{matched_key}' not yet visible — retrying in 2.5s")
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Launch check error: {e}")
                return True

        self.log(f"  [CHECK] Expected '{matched_key}' on screen but not found")
        return False

    # ── Adaptive inter-subtask wait ───────────────────────────────────────────

    def _wait_for_settle(self, min_s: float = 0.5, max_s: float = 3.0, poll_interval: float = 0.25):
        """
        Wait until the screen stops changing (transitions complete) or max_s elapses.
        Polls OCR text every poll_interval seconds; if two consecutive reads are identical
        the screen has settled and we return early. Never returns in less than min_s.
        On any error falls back to sleeping max_s.
        """
        import imagehash as _ih
        try:
            time.sleep(min_s)
            deadline = time.time() + (max_s - min_s)
            prev_hash = None
            while time.time() < deadline:
                img = self.capturer.capture()
                img.thumbnail((320, 180))
                cur_hash = str(_ih.phash(img))
                if prev_hash is not None and cur_hash == prev_hash:
                    self.log(f"  [SETTLE] Screen stable — proceeding")
                    return
                prev_hash = cur_hash
                time.sleep(poll_interval)
            self.log(f"  [SETTLE] Max wait {max_s:.1f}s reached — proceeding anyway")
        except Exception as e:
            logger.debug(f"[SETTLE] Error: {e} — falling back to fixed wait")
            time.sleep(max_s)

    # ── Screen context ─────────────────────────────────────────────────────────

    def _get_screen_context(self) -> str:
        try:
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            words = self._ocr.extract(img)
            visible = [
                w.text for w in words
                if 2 <= len(w.text) <= 30
                and w.conf >= 0.80
                and not any(c in w.text for c in ('/', '\\', '=', '{', '}', '$', '#'))
            ]
            seen, unique = set(), []
            for t in visible:
                if t.lower() not in seen:
                    seen.add(t.lower())
                    unique.append(t)
            return ", ".join(f'"{t}"' for t in unique[:40])
        except Exception:
            return ""

    # ── Topological sort ──────────────────────────────────────────────────────

    def _topological_sort(self, subtasks: List[SubTask]) -> List[SubTask]:
        by_id = {s.id: s for s in subtasks}
        in_degree = {s.id: len(s.depends_on) for s in subtasks}
        dependents = {s.id: [] for s in subtasks}
        for s in subtasks:
            for dep_id in s.depends_on:
                if dep_id in dependents:
                    dependents[dep_id].append(s.id)

        order, queue = [], [s for s in subtasks if not s.depends_on]
        while queue:
            current = queue.pop(0)
            order.append(current)
            for dep_id in dependents.get(current.id, []):
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(by_id[dep_id])

        if len(order) != len(subtasks):
            logger.warning("[ORCHESTRATOR] Dependency cycle — falling back to ID order")
            return sorted(subtasks, key=lambda s: s.id)
        return order
