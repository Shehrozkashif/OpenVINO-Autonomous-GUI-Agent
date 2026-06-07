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
import subprocess
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
from core.capture.screen_snapshot import ScreenSnapshot, capture_snapshot
from core.capture.screenshot import ScreenCapture, _screen_size
from core.executor.burst_executor import BurstExecutor, detect_burst, detect_burst_from_instruction
from core.protocols.a2a import ActionStep, SubTask
from memory.task.task_memory import TaskMemory


@dataclass
class OrchestratorConfig:
    max_retries_per_step: int = 3
    reflection_wait_s: float = 0.5
    max_steps_per_subtask: int = 20
    max_scroll_find_attempts: int = 6   # scrolls before giving up on an off-screen element
    consecutive_failures_limit: int = 3


# Per-action-type limit on consecutive identical successful steps.
# Stricter for high-signal actions (type, right_click); lenient for nav keys.
DEDUP_LIMIT_BY_ACTION_TYPE: Dict[str, int] = {
    "type":        1,   # same text typed twice → almost certainly a loop
    "click":       2,   # allow up to 2 repeats (e.g. re-focusing a window)
    "key_press":   3,   # allow up to 3 repeats (e.g. multiple Escape/arrow presses)
    "right_click": 1,   # context menu should open once
}


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
        ocr: Optional["OCREngine"] = None,
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
        self.burst_executor = BurstExecutor(
            grounder=self.grounder,
            actor=self.actor,
            reflector=self.reflector,
        )

        if ocr is not None:
            self._ocr = ocr
        else:
            from agents.grounding.grounding_agent import OCREngine
            self._ocr = OCREngine()
        self._screen_w, self._screen_h = _screen_size()
        self._extracted_data: Dict[str, str] = {}

    def stop(self):
        self._stop_event.set()

    def _refresh_own_window_mask(self) -> None:
        """Dynamically mask the GUI agent window from screen captures when it is the topmost window.

        The GUI window shows the task description text (e.g. 'open Notepad'), and OCR picks that
        up, confusing the planner into thinking apps are already running or grounding to wrong
        coordinates. When another app (e.g. Notepad) is in the foreground, the GUI window is
        behind it so its text isn't visible — no masking needed, and masking would obscure the
        target app.  Using GetForegroundWindow() makes this safe in both cases.
        """
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32

            # Cache hwnd — find once using EnumWindows (partial title match, immune to
            # exact dash character encoding differences in FindWindowW).
            if not hasattr(self, "_own_hwnd"):
                _found = [0]

                @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
                def _cb(hwnd, _lparam):
                    length = user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        user32.GetWindowTextW(hwnd, buf, length + 1)
                        if "Desktop GUI Agent" in buf.value:
                            _found[0] = hwnd
                            return False  # stop enumeration
                    return True

                user32.EnumWindows(_cb, 0)
                self._own_hwnd = _found[0] or None
                if self._own_hwnd:
                    logger.info(f"[ORCHESTRATOR] GUI window handle cached: hwnd={self._own_hwnd}")
                else:
                    logger.debug("[ORCHESTRATOR] GUI window not found via EnumWindows — no masking")

            hwnd = getattr(self, "_own_hwnd", None)
            if not hwnd:
                self.capturer.exclude_regions = []
                return

            # Only mask when our GUI window is the topmost (foreground) window.
            # When another app is in focus, the GUI window's text is hidden behind it.
            foreground = user32.GetForegroundWindow()
            if foreground != hwnd:
                if self.capturer.exclude_regions:
                    self.capturer.exclude_regions = []
                    logger.debug("[ORCHESTRATOR] GUI window not foreground — mask cleared")
                return

            # Our window is foreground — compute bounds once and cache them
            if not self.capturer.exclude_regions:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                bounds = (rect.left, rect.top, rect.right, rect.bottom)
                self.capturer.exclude_regions = [bounds]
                logger.info(f"[ORCHESTRATOR] GUI window masked at {bounds}")
        except Exception as e:
            logger.debug(f"[ORCHESTRATOR] GUI window mask lookup failed: {e}")

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

        # Attempt instruction-level burst detection before invoking the LLM router.
        # Compound sequences (right-click → New → Folder → type → Enter) produce a
        # single synthetic subtask, skipping router latency entirely.
        _instr_burst = detect_burst_from_instruction(instruction)
        if _instr_burst is not None:
            self.log(
                f"[BURST] Instruction-level burst detected "
                f"({len(_instr_burst.steps)} steps) — skipping router"
            )
            synthetic = SubTask(id=1, description=instruction, depends_on=[], burst=_instr_burst)
            task_id, subtasks = "burst_direct", [synthetic]
        else:
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
                if not success:
                    # Planner returned "goal achieved" but the app is not actually running.
                    # Retry with an explicit hint: the icon-click approach failed, use the
                    # search launcher instead. This prevents repeating the same wrong strategy.
                    self.log(f"  [RETRY] Launch not confirmed — re-running subtask {subtask.id}")
                    retry_ctx = completed_subtask_descs + [
                        f"[RETRY NOTE] Clicking the icon or taskbar entry for "
                        f"'{subtask.description}' did NOT launch the app. "
                        f"Use the search launcher instead: "
                        f"key_press winleft → type app name → key_press enter."
                    ]
                    success = self._execute_subtask(subtask, task_context=retry_ctx)
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
        _last_step_sig: tuple = ()    # (action_type, target, value, key)
        _same_step_streak: int = 0   # consecutive successes of the identical step

        # Prefer a pre-attached burst (set by instruction-level detection in execute())
        # over per-subtask pattern matching; fall back to detect_burst() otherwise.
        _burst = (
            subtask.burst
            if getattr(subtask, "burst", None) is not None
            else detect_burst(subtask)
        )
        if _burst is not None:
            self.log(f"  [BURST] {len(_burst.steps)}-step burst pattern detected")
            _burst_result = self.burst_executor.run(_burst)
            if _burst_result.success:
                self.log(f"  [BURST] Succeeded ({len(_burst.steps)} steps, no LLM required)")
                return True
            self.log(
                f"  [BURST] Failed at step {_burst_result.failed_at_step}: "
                f"{_burst_result.reason} — falling back to planning loop"
            )

        # Process-based early-exit only fires for genuine app-launch subtasks, not for
        # subtasks that USE the app ("with Notepad already open, save the file").
        # "notepad running" does not mean "file saved" — avoid false goal achievement.
        _desc_lower_sub = subtask.description.lower()
        _is_launch_goal = (
            not ("already open" in _desc_lower_sub or "already running" in _desc_lower_sub)
            and (
                any(w in _desc_lower_sub for w in ("launch",))
                or (
                    "open" in _desc_lower_sub
                    and any(k in _desc_lower_sub for k in self._PROCESS_MAP_WINDOWS)
                )
            )
        )

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

                # Capture pre-action hash for click steps BEFORE the action fires.
                # The reflection agent's internal "before" capture runs AFTER execute(),
                # so instant UI changes (menus opening) make before==after → false delta=0.
                _pre_click_hash = None
                if step.action_type in ("click", "right_click", "double_click"):
                    try:
                        import imagehash as _ih
                        _pci = self.capturer.capture()
                        _pci.thumbnail((320, 180))
                        _pre_click_hash = _ih.phash(_pci)
                    except Exception:
                        pass

                exec_ok = self._execute_step(step)
                if not exec_ok:
                    last_error = "execution failed (element not found or action error)"
                    continue

                if skip_reflection:
                    step_success = True
                    break

                try:
                    # Give apps extra time to open after a launcher Enter press.
                    # 0.5 s is too short for Calculator/Notepad to show their UI.
                    _is_launch_enter = (
                        step.action_type == "key_press"
                        and (step.key or "").lower() == "enter"
                        and any(kw in step.description.lower()
                                for kw in ("launch", "open", "start", "run"))
                    )
                    _reflect_wait = 1.5 if _is_launch_enter else self.config.reflection_wait_s
                    reflection = self.reflector.verify(step, _reflect_wait,
                                                       pre_hash=_pre_click_hash)
                    _cached_ocr = reflection.ocr_text   # reuse for next planning call

                    if reflection.success:
                        step_success = True
                        self.log(f"  Verified (conf={reflection.confidence:.2f})")
                        time.sleep(0.1)
                        break

                    # Step reported as failed by the reflector.
                    # fail_threshold determines how to interpret uncertain failures:
                    # below the threshold = outcome is inconclusive → retry;
                    # at or above = reflector is confident it failed → retry or stop.
                    # NOTE: reflection.confidence < fail_threshold no longer implies
                    # success. It means "uncertain — try again." Only reflection.success
                    # can set step_success = True.
                    _key = (step.key or "").lower()
                    _is_launcher_key = step.action_type == "key_press" and _key in ("super", "enter")
                    fail_threshold = (
                        0.95
                        if step.action_type in ("type", "hotkey") or _is_launcher_key
                        else self.reflector.min_confidence
                    )

                    if reflection.confidence < fail_threshold:
                        # Reflector is uncertain about the failure — retry the action.
                        last_error = (
                            f"uncertain outcome (conf={reflection.confidence:.2f}, "
                            f"threshold={fail_threshold:.2f})"
                        )
                        self.log(f"  Uncertain result — retrying ({last_error})")
                        # Fast-path: if the goal process is already running after an
                        # uncertain click or launch-Enter, skip remaining retries.
                        if _OS == "Windows" and _is_launch_goal and (
                            step.action_type in ("click", "double_click")
                            or _is_launch_enter
                        ):
                            _proc_early = next(
                                (self._PROCESS_MAP_WINDOWS[k]
                                 for k in self._PROCESS_MAP_WINDOWS
                                 if k in subtask.description.lower()),
                                None,
                            )
                            if _proc_early and self._is_process_running(_proc_early):
                                self.log(
                                    f"  [GOAL-CHECK-EARLY] "
                                    f"'{_proc_early.split('.')[0]}' already running "
                                    f"— accepting step"
                                )
                                step_success = True
                                break
                    else:
                        # Reflector is confident the step failed.
                        last_error = (
                            reflection.error_description
                            or "action did not produce expected result"
                        )
                        self.log(
                            f"  Verification failed: {last_error} "
                            f"(conf={reflection.confidence:.2f})"
                        )
                        if not reflection.should_retry:
                            # Definitive failure, no point retrying this attempt.
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

                # Early-exit: after each successful step in a known app-launch subtask,
                # check the process list on Windows. If the app is already running, the
                # goal is achieved — return True immediately so the planner never gets a
                # chance to generate spurious continuation steps (e.g. pressing Enter
                # again inside a Calculator that is already open).
                if _OS == "Windows" and _is_launch_goal:
                    _desc_lower_chk = subtask.description.lower()
                    _matched_proc = next(
                        (self._PROCESS_MAP_WINDOWS[k]
                         for k in self._PROCESS_MAP_WINDOWS
                         if k in _desc_lower_chk),
                        None,
                    )
                    if _matched_proc and self._is_process_running(_matched_proc):
                        _label = _matched_proc.split(".")[0]
                        self.log(f"  [GOAL-CHECK] '{_label}' confirmed running — goal achieved")
                        return True

                sig = (step.action_type, step.target, step.value, step.key)
                if sig == _last_step_sig:
                    _same_step_streak += 1
                    _dedup_limit = DEDUP_LIMIT_BY_ACTION_TYPE.get(step.action_type, 2)
                    if _same_step_streak > _dedup_limit:
                        self.log(
                            f"  [LOOP-GUARD] '{step.action_type}' repeated "
                            f"{_same_step_streak + 1}× (limit={_dedup_limit}) — "
                            f"loop detected, declaring goal achieved"
                        )
                        if step.action_type in ("click", "right_click"):
                            try:
                                _esc = ActionStep(
                                    id=0, subtask_id=step.subtask_id,
                                    action_type="key_press",
                                    target=None, value=None, key="escape",
                                    description="Escape to dismiss stray menu",
                                    verification="",
                                )
                                self._execute_step(_esc)
                                self.log("  [LOOP-GUARD] Recovery Escape sent")
                            except Exception:
                                pass
                        return True
                else:
                    _same_step_streak = 0
                    _last_step_sig = sig
            else:
                note = (
                    f"[FAILED: {last_error}] {step.description}"
                    if last_error else f"[FAILED] {step.description}"
                )
                completed.append(note)

                # Even on step failure, check if the goal process is now running.
                # This handles the common case where key_press enter launches an app
                # but the reflector times out before the app's UI is fully visible
                # (reflection confidence 0.50 → step marked failed, yet app IS open).
                if _OS == "Windows" and _is_launch_goal:
                    _desc_lower_chk = subtask.description.lower()
                    _matched_proc_fail = next(
                        (self._PROCESS_MAP_WINDOWS[k]
                         for k in self._PROCESS_MAP_WINDOWS
                         if k in _desc_lower_chk),
                        None,
                    )
                    if _matched_proc_fail and self._is_process_running(_matched_proc_fail):
                        _label = _matched_proc_fail.split(".")[0]
                        self.log(
                            f"  [GOAL-CHECK] '{_label}' confirmed running "
                            f"despite step failure — goal achieved"
                        )
                        return True

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
            # Note: element_type check removed — Windows 11 Start menu and search
            # overlays do not register as the foreground window in GetForegroundWindow,
            # so capture_snapshot() cannot tag their OCR words as foreground_interactive.
            # Clicks on these elements (score=1.0, method=ocr_direct) are correct.
            # The GUI window masking (exclude_regions) prevents log-text false positives.
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

    # Maps description keyword → Windows process executable name.
    # Used by _verify_launch on Windows for reliable process-based checking
    # (immune to the OCR false-positive caused by the GUI agent's own window text).
    _PROCESS_MAP_WINDOWS: dict = {
        "windows terminal": "WindowsTerminal.exe",
        "terminal":         "WindowsTerminal.exe",
        "command prompt":   "cmd.exe",
        "powershell":       "powershell.exe",
        "notepad":          "notepad.exe",
        "calculator":       "CalculatorApp.exe",
        "paint":            "mspaint.exe",
        "file explorer":    "explorer.exe",
        "explorer":         "explorer.exe",
        "edge":             "msedge.exe",
        "firefox":          "firefox.exe",
        "chrome":           "chrome.exe",
        "brave":            "brave.exe",
        "vs code":          "Code.exe",
        "visual studio":    "devenv.exe",
        "task manager":     "Taskmgr.exe",
        "snipping tool":    "SnippingTool.exe",
        "wordpad":          "wordpad.exe",
        "settings":         "SystemSettings.exe",
    }

    _APP_SIGNALS_WINDOWS: dict = {
        "calculator":      ["Calculator", "Standard", "Scientific"],
        "notepad":         ["Notepad", "Untitled", "File"],
        "paint":           ["Paint", "Home", "Image"],
        "command prompt":  ["cmd", "C:\\", "Microsoft"],
        "powershell":      ["PowerShell", "PS", "Windows"],
        "windows terminal": ["Windows Terminal", "Terminal", "PowerShell"],
        "terminal":        ["Terminal", "PowerShell", "cmd"],
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

    # Generic words that don't make useful OCR signals on their own
    _GENERIC_NAME_WORDS = frozenset((
        "the", "a", "an", "app", "application", "browser", "program", "tool",
    ))

    @property
    def _APP_SIGNALS(self) -> dict:
        return self._APP_SIGNALS_WINDOWS if _OS == "Windows" else self._APP_SIGNALS_LINUX

    def _derive_launch_signals(self, description: str) -> List[str]:
        """
        Fallback for apps absent from the curated _APP_SIGNALS table: derive OCR
        signal words directly from the app name in "open <AppName>" — e.g.
        "open Brave Browser" → ["Brave Browser", "Brave"]. This keeps launch
        verification working for ANY installed app, not just well-known ones.
        Preserves original casing (unlike `desc`) since OCR text is case-sensitive.
        """
        m = re.search(
            r"\b(?:open|launch|start)\s+(?:the |a |an )?"
            r"([A-Za-z0-9][\w+-]*(?:[ \t]+[A-Za-z0-9][\w+-]*)*?)"
            r"(?:[ \t]+(?:using|via|by|with|and)\b|[.,!?]|$)",
            description,
        )
        if not m:
            return []
        name = m.group(1).strip().rstrip(".")
        words = [w for w in name.split()
                 if len(w) >= 3 and w.lower() not in self._GENERIC_NAME_WORDS]
        if not words:
            return []
        return [name] + words[:2]

    def _is_process_running(self, exe_name: str) -> bool:
        """Check if a process with the given executable name is running (Windows only)."""
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
                capture_output=True, text=True, timeout=3,
            ).stdout
            return exe_name.lower() in out.lower()
        except Exception:
            return False

    def _verify_launch(self, subtask) -> bool:
        desc = subtask.description.lower()
        if "already open" in desc or "already running" in desc:
            return True
        # Only run verification for genuine app-launch subtasks.
        # The old bare "open" substring match triggered on phrases like
        # "right click on the desktop to open the context menu" — a false positive
        # that caused the agent to retry with a Start-menu search.
        # New rule: require "launch"/"search launcher" explicitly, OR "open" paired
        # with a known-app keyword from the curated signal tables.
        _app_vocab = set(self._PROCESS_MAP_WINDOWS) | set(self._APP_SIGNALS)
        _is_app_launch = (
            any(w in desc for w in ("launch", "search launcher"))
            or ("open" in desc and any(k in desc for k in _app_vocab))
        )
        if not _is_app_launch:
            return True

        # On Windows: use process-based check (immune to GUI-text false positives)
        if _OS == "Windows":
            matched_proc = next(
                (self._PROCESS_MAP_WINDOWS[k] for k in self._PROCESS_MAP_WINDOWS if k in desc),
                None,
            )
            if matched_proc:
                label = matched_proc.split(".")[0]
                for attempt, wait_s in enumerate([1.5, 2.5]):
                    time.sleep(wait_s)
                    if self._is_process_running(matched_proc):
                        self.log(f"  [CHECK] '{label}' process confirmed running")
                        return True
                    if attempt == 0:
                        self.log(f"  [CHECK] '{label}' not yet running — retrying in 2.5s")
                self.log(f"  [CHECK] Process '{matched_proc}' not found after launch")
                return False

        # Fallback: OCR-based signal check (Linux/macOS, or apps not in process map)
        matched_key = next(
            (k for k in self._APP_SIGNALS if k in desc), None
        )
        signals = (
            self._APP_SIGNALS[matched_key] if matched_key
            else self._derive_launch_signals(subtask.description)
        )
        if not signals:
            return True
        label = matched_key or signals[0]

        for attempt, wait_s in enumerate([1.5, 2.5]):
            time.sleep(wait_s)
            try:
                # Filter to foreground regions to exclude GUI agent window text.
                snapshot = capture_snapshot(self.capturer, self._ocr)
                ocr_words = {r.text for r in snapshot.ocr_regions if r.is_in_foreground}
                all_ocr_text = " ".join(ocr_words)
                if any(sig in all_ocr_text for sig in signals):
                    self.log(f"  [CHECK] '{label}' confirmed on screen")
                    return True
                if attempt == 0:
                    self.log(f"  [CHECK] '{label}' not yet visible — retrying in 2.5s")
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Launch check error: {e}")
                return True

        self.log(f"  [CHECK] Expected '{label}' on screen but not found")
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
        """
        Return a structured screen context string for the planner.

        Uses capture_snapshot() to separate foreground UI text from background
        window text so the planner can focus on interactable elements only.
        Falls back to a flat token list if snapshot capture fails.
        """
        try:
            if _OS == "Windows":
                self._refresh_own_window_mask()
            snapshot = capture_snapshot(self.capturer, self._ocr)
            return snapshot.format_for_planner()
        except Exception:
            # Fallback: flat token list when snapshot capture fails
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
