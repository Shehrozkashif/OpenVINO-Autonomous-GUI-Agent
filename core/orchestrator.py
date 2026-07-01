# core/orchestrator.py
"""Task Orchestrator — the central coordinator.

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
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from loguru import logger

from agents.action import ActionExecutionAgent
from agents.grounding import GroundingResult, UIGroundingAgent
from agents.planning import PlanningAgent, PlanningParseError
from agents.reflection import ReflectionAgent
from agents.router import RouterAgent
from core.burst_executor import BurstExecutor, detect_burst, detect_burst_from_instruction
from core.capture.screen_snapshot import capture_snapshot
from core.capture.screenshot import ScreenCapture, _screen_size
from core.protocols import ActionStep, SubTask
from memory.task_memory import TaskMemory

if TYPE_CHECKING:
    from agents.grounding import OCREngine


@dataclass
class OrchestratorConfig:
    max_retries_per_step: int = 3
    reflection_wait_s: float = 0.5
    max_steps_per_subtask: int = 20
    max_scroll_find_attempts: int = 6   # scrolls before giving up on an off-screen element
    consecutive_failures_limit: int = 3
    # Wall-clock safety budgets (H8). A stuck task (e.g. repeated 30-50 s VLM
    # swaps) must not run unbounded. 0 disables a budget.
    task_deadline_s: float = 600.0      # hard cap on a whole instruction
    subtask_deadline_s: float = 240.0   # hard cap on a single subtask
    # After this many consecutive step failures, planning escalates from the
    # text path (OCR context → LLM) to the visual path (screenshot → UI-TARS),
    # which sees icons/layout the text path is blind to. 0 disables escalation.
    visual_replan_after: int = 2


# Per-action-type limit on consecutive identical successful steps.
# Stricter for high-signal actions (type, right_click); lenient for nav keys.
DEDUP_LIMIT_BY_ACTION_TYPE: dict[str, int] = {
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
        config: OrchestratorConfig | None = None,
        on_step_log: Callable[[str], None] | None = None,
        ocr: Optional["OCREngine"] = None,
        on_confirm: Callable[[str, str], bool] | None = None,
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
        # Optional human confirmation handler for destructive commands.
        # Signature: on_confirm(summary, command) -> bool. When None, the action
        # firewall blocks HIGH-severity commands and allows MEDIUM ones (logged).
        self.on_confirm = on_confirm
        self._stop_event = threading.Event()
        # Set when a subtask completes via a degraded path (loop-guard recovery,
        # optimistic parse-failure success, etc.). Such a task is NOT stored in
        # success memory, so broken plans never poison future routing hints.
        self._degraded = False
        self.burst_executor = BurstExecutor(
            grounder=self.grounder,
            actor=self.actor,
            reflector=self.reflector,
        )

        if ocr is not None:
            self._ocr = ocr
        else:
            from agents.grounding import OCREngine
            self._ocr = OCREngine()
        self._screen_w, self._screen_h = _screen_size()
        self._extracted_data: dict[str, str] = {}
        # Per-task window-count baselines for apps that were ALREADY running
        # when their launch subtask started. exe_name → window count. Used to
        # require a NEW window (focusing an existing one must not pass).
        self._launch_window_baseline: dict[str, int] = {}

    def stop(self):
        self._stop_event.set()

    # ── Emergency kill switch (C8) ──────────────────────────────────────────────

    def _arm_kill_switch(self) -> None:
        """Arm the global emergency-stop listener for the duration of a task."""
        try:
            self._disarm_kill_switch()  # clear any leaked listener first
            from core.controller import KillSwitch
            controller = getattr(self.actor, "controller", None)
            self._kill_switch = KillSwitch(on_trigger=self.stop, controller=controller)
            self._kill_switch.start()
        except Exception as e:
            logger.debug(f"[ORCHESTRATOR] Kill switch not armed: {e}")
            self._kill_switch = None

    def _disarm_kill_switch(self) -> None:
        ks = getattr(self, "_kill_switch", None)
        if ks is not None:
            try:
                ks.stop()
            except Exception:
                pass
            self._kill_switch = None

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
        self._launch_window_baseline = {}
        self._degraded = False
        self.log(f"[TASK START] '{instruction}'")
        start_time = time.time()
        self._arm_kill_switch()

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
        completed_subtask_descs: list[str] = []   # for inter-subtask context
        failed = False

        for subtask in self._topological_sort(subtasks):
            if self._stop_event.is_set():
                self.log("[TASK] Stopped by user.")
                failed = True
                break

            # H8: enforce the task-level wall-clock budget.
            if self.config.task_deadline_s and (
                time.time() - start_time > self.config.task_deadline_s
            ):
                self.log(
                    f"[TASK] Deadline ({self.config.task_deadline_s:.0f}s) exceeded "
                    f"— aborting before subtask {subtask.id}"
                )
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

        self._disarm_kill_switch()

        elapsed = time.time() - start_time
        summary = self.router.summarize_completion(task_id, completed_subtask_ids, not failed)

        # Append any extracted data to the summary log
        if self._extracted_data:
            lines = "\n".join(f"  {k}: {v}" for k, v in self._extracted_data.items())
            self.log(f"\n[EXTRACTED DATA]\n{lines}")

        self.log(f"\n[TASK DONE] {summary} ({elapsed:.1f}s)")

        # Only persist as a reusable success when the task genuinely completed
        # AND no subtask finished via a degraded path (loop-guard recovery,
        # optimistic parse-failure success). Storing degraded runs would poison
        # future routing hints with plans that did not actually work.
        if not failed and not self._degraded:
            self.memory.store_successful_task(instruction, subtasks, elapsed)
        elif not failed and self._degraded:
            self.log("[MEMORY] Task completed via a degraded path — not stored as a reusable success")

        return {
            "task_id": task_id,
            "success": not failed,
            "subtasks_completed": completed_subtask_ids,
            "elapsed_s": elapsed,
            "summary": summary,
            "extracted_data": dict(self._extracted_data),
        }

    # ── Subtask execution loop ────────────────────────────────────────────────

    def _try_burst(self, subtask: SubTask) -> bool:
        """Run a recognised zero-LLM burst pattern for this subtask, if any.

        Prefers a pre-attached burst (set by instruction-level detection in
        execute()) over per-subtask pattern matching; falls back to
        detect_burst(). Returns True only when a burst ran AND succeeded — in
        which case the subtask is complete. Otherwise returns False and the
        caller falls back to the planning loop.
        """
        _burst = (
            subtask.burst
            if getattr(subtask, "burst", None) is not None
            else detect_burst(subtask)
        )
        if _burst is None:
            return False
        self.log(f"  [BURST] {len(_burst.steps)}-step burst pattern detected")
        _burst_result = self.burst_executor.run(_burst)
        if _burst_result.success:
            self.log(f"  [BURST] Succeeded ({len(_burst.steps)} steps, no LLM required)")
            return True
        self.log(
            f"  [BURST] Failed at step {_burst_result.failed_at_step}: "
            f"{_burst_result.reason} — falling back to planning loop"
        )
        return False

    def _setup_launch_goal(
        self, subtask: SubTask, task_context: list[str] | None,
    ) -> tuple:
        """Detect whether this subtask is a genuine app-launch goal, and on
        Windows record a window-count baseline when the target app is already
        running.

        Process-based early-exit only fires for genuine app-launch subtasks, not
        for subtasks that USE the app ("with Notepad already open, save the
        file") — "notepad running" does not mean "file saved". When the app is
        ALREADY running, "process exists" is a vacuous launch signal (true
        before we did anything) and focusing the existing window is actively
        dangerous (it may be busy running another program), so a baseline is
        recorded and only a NEW window proves the launch.

        Returns (is_launch_goal, goal_proc, baseline_windows, task_context); the
        returned task_context may carry an appended pre-check note.
        """
        desc = subtask.description.lower()
        is_launch_goal = (
            not ("already open" in desc or "already running" in desc)
            and (
                any(w in desc for w in ("launch",))
                or ("open" in desc
                    and any(k in desc for k in self._PROCESS_MAP_WINDOWS))
            )
        )

        goal_proc: str | None = None
        baseline_windows: int | None = None
        if is_launch_goal:
            goal_proc = next(
                (self._PROCESS_MAP_WINDOWS[k] for k in self._PROCESS_MAP_WINDOWS
                 if k in desc),
                None,
            )
            if goal_proc and self._is_process_running(goal_proc):
                baseline_windows = self._count_process_windows(goal_proc)
                self._launch_window_baseline[goal_proc] = baseline_windows
                task_context = (task_context or []) + [
                    f"[NOTE] {goal_proc.split('.')[0]} is ALREADY running with "
                    f"{baseline_windows} window(s). The existing window may be "
                    f"busy running another program — do NOT click its taskbar "
                    f"button and do NOT focus the existing window. Open a NEW "
                    f"window instead: key_press winleft → type the app name → "
                    f"key_press enter."
                ]
                self.log(
                    f"  [PRE-CHECK] {goal_proc} already running "
                    f"({baseline_windows} window(s)) — a NEW window is required"
                )
        return is_launch_goal, goal_proc, baseline_windows, task_context

    def _goal_confirmed(
        self, goal_proc: str | None, baseline_windows: int | None,
    ) -> bool:
        """Launch-goal achieved? A NEW window is required when the app pre-existed."""
        if not goal_proc:
            return False
        if baseline_windows is not None:
            return self._count_process_windows(goal_proc) > baseline_windows
        return self._launch_confirmed(goal_proc)

    def _execute_subtask(self, subtask: SubTask, task_context: list[str] = None) -> bool:
        """Dynamic loop — plans ONE step at a time using live screen state.

        task_context: summaries of subtasks completed before this one in the same task.
        Each step planner sees: goal + inter-subtask context + within-subtask history + screen.
        """
        completed: list[str] = []
        consecutive_failures = 0
        _cached_ocr: str = ""   # reuse reflection's OCR for next planning step
        _last_step_sig: tuple = ()    # (action_type, target, value, key)
        _same_step_streak: int = 0   # consecutive successes of the identical step

        # Fast path: recognised multi-step patterns run with zero LLM calls.
        if self._try_burst(subtask):
            return True

        _is_launch_goal, _goal_proc, _baseline_windows, task_context = (
            self._setup_launch_goal(subtask, task_context)
        )

        # Terminal-command subtasks get DETERMINISTIC verification: a successful
        # shell command prints nothing (new empty prompt), which OCR-based
        # reflection systematically mis-reads as "no change → failed". Check the
        # real world instead: file created/deleted on disk, or error text in
        # the terminal output.
        _is_cmd_subtask = "run:" in subtask.description.lower()
        _typed_ok = False   # a type step succeeded in this subtask

        # "Save the document as <path>" subtasks are ALSO file-producing, but
        # through a GUI dialog. Editors give no OCR-readable "saved" signal, so
        # reflection loops on ctrl+s forever. Verify the same way as commands:
        # the subtask is DONE the moment the named file appears on disk (fresh).
        _save_target = self._subtask_save_target(subtask)

        _subtask_start = time.time()

        # The Save-As key sequence is fixed and well-defined, but the 8B planner
        # is unreliable at threading the dialog (it loops on ctrl+s and never
        # types the path). Run it deterministically first and confirm on disk;
        # only the key sequence is fixed — the path is model-chosen. On any
        # failure, fall through to the normal planning loop.
        if _save_target and self._try_save_as(_save_target, _subtask_start):
            return True

        for step_idx in range(self.config.max_steps_per_subtask):
            if self._stop_event.is_set():
                return False

            # H8: enforce the per-subtask wall-clock budget.
            if self.config.subtask_deadline_s and (
                time.time() - _subtask_start > self.config.subtask_deadline_s
            ):
                self.log(
                    f"  Subtask deadline ({self.config.subtask_deadline_s:.0f}s) "
                    f"exceeded — aborting subtask"
                )
                return False

            # A "save as <path>" subtask is COMPLETE the instant the file lands on
            # disk. Check before planning so a successful save short-circuits the
            # ctrl+s retry loop (editors show no readable confirmation).
            if _save_target and self._file_saved_fresh(_save_target, _subtask_start):
                self.log(
                    f"  [SAVE-CHECK] '{_save_target}' on disk (fresh) — save confirmed"
                )
                return True

            screen_context = _cached_ocr if _cached_ocr else self._get_screen_context()
            _cached_ocr = ""   # consume — will be refreshed after reflection

            # Pass failure hints so planner avoids repeating known-bad patterns
            failure_hints = []
            try:
                failure_hints = self.memory.get_failure_hints(subtask.description)
            except Exception:
                pass

            # Escalate to visual planning (screenshot → UI-TARS) once the text
            # path has failed visual_replan_after times in a row — the text
            # planner is blind to icons/layout, which is usually why it's stuck.
            # Exception: terminals. A failing terminal subtask needs a corrected
            # COMMAND, not a click — visual clicking inside a console wastes a
            # 30-60s VLM swap and can never fix the error.
            _visual_mode = (
                self.config.visual_replan_after > 0
                and consecutive_failures >= self.config.visual_replan_after
            )
            # A "run: <cmd>" subtask is fixed by a CORRECTED COMMAND, never by the
            # VLM clicking/typing into the console. Escalating it to visual mode
            # re-types the command (corrupting the prompt line) and burns a 30-60s
            # model swap. Block it deterministically on _is_cmd_subtask — do not
            # rely on foreground-process detection, which can misfire right after
            # an Enter press and let the destructive re-type through.
            if _visual_mode and (_is_cmd_subtask or self._foreground_is_terminal()):
                self.log(
                    "  [VISUAL-REPLAN] Skipped — terminal command subtask "
                    "(clicks/re-typing can't fix command errors)"
                )
                _visual_mode = False
            try:
                if _visual_mode:
                    self.log("  [VISUAL-REPLAN] Text planning stuck — asking VLM with screenshot")
                    try:
                        step = self._plan_visual(subtask, completed)
                    except PlanningParseError:
                        raise
                    except Exception as e:
                        # VLM infrastructure error (timeout, model swap failure):
                        # degrade gracefully to the text planner for this step.
                        self.log(f"  [VISUAL-REPLAN] VLM error ({e}) — using text planner")
                        _visual_mode = False
                    if _visual_mode and step is None:
                        # VLM says finished() — plausible (earlier failures may
                        # have been reflection false-negatives), but don't store
                        # this run as a clean reusable success.
                        self._degraded = True
                        return True
                if not _visual_mode:
                    step = self.planner.plan_next_step(
                        subtask, screen_context, completed,
                        task_context=task_context, failure_hints=failure_hints,
                    )
                    if step is None:
                        return True   # planner says goal is achieved
            except PlanningParseError as e:
                # Unparseable planner output is a planning FAILURE, not goal
                # achievement. Record it and let the loop try again (or abort
                # via the consecutive-failures limit).
                self.log(f"  Planning failed: {e}")
                completed.append("[FAILED: planner produced unparseable output]")
                consecutive_failures += 1
                if consecutive_failures >= self.config.consecutive_failures_limit:
                    self.log(
                        f"  {self.config.consecutive_failures_limit} consecutive failures"
                        f" — aborting subtask"
                    )
                    return False
                continue

            self.log(f"  Step {step_idx + 1}: [{step.action_type}] {step.description}")

            # Terminal-command subtasks verify the typed command DETERMINISTICALLY
            # at the Enter press (file-on-disk / shell-error check in
            # _verify_command_effect). OCR-reading a dark console to confirm a type
            # step systematically misfires ("FAILED" on correct typing) and used to
            # trigger a destructive re-type. Defer to the authoritative Enter check.
            _cmd_type_step = _is_cmd_subtask and step.action_type == "type"
            skip_reflection = step.action_type in ("wait", "extract") or _cmd_type_step
            if _cmd_type_step:
                self.log(
                    "  [CMD-TYPE] Command typed — deferring verification to the "
                    "Enter/disk check (skipping unreliable OCR reflection)"
                )
            step_success = False
            last_error = ""

            # Fix C5: classify whether re-executing this exact step is safe.
            # Non-idempotent actions change state every time they run: typing
            # appends text again ("hellohello"), Enter submits/creates twice,
            # ctrl+v pastes twice. For these, once the action has physically
            # executed we must NOT blind-retry on an uncertain/failed verdict —
            # we hand control back to the planner, which sees the live screen and
            # decides the next action (it can tell the text is already there).
            # Idempotent actions (click same coords, scroll, escape, nav keys)
            # are safe to repeat, so they keep the normal retry behaviour.
            _key_l = (step.key or "").lower()
            _non_idempotent = (
                step.action_type == "type"
                or (step.action_type == "key_press" and _key_l in ("enter", "return", "space"))
                or (step.action_type == "hotkey" and "v" in _key_l.split("+") and "ctrl" in _key_l)
            )

            for attempt in range(self.config.max_retries_per_step):
                if attempt > 0:
                    self.log(f"  Retry {attempt}/{self.config.max_retries_per_step}…")

                # Capture pre-action hash for click steps BEFORE the action fires.
                # The reflection agent's internal "before" capture runs AFTER execute(),
                # so instant UI changes (menus opening) make before==after → false delta=0.
                _pre_click_hash = None
                if step.action_type in ("click", "right_click", "double_click"):
                    try:
                        from core.capture.screenshot import frame_phash
                        _pre_click_hash = frame_phash(self.capturer.capture())
                    except Exception:
                        pass

                exec_ok = self._execute_step(step)
                if not exec_ok:
                    last_error = "execution failed (element not found or action error)"
                    continue

                if skip_reflection:
                    step_success = True
                    break

                # Deterministic verification for the execute-Enter of a
                # "run: <command>" subtask — checks the filesystem / terminal
                # output instead of LLM reflection. On confirmed effect the
                # whole subtask is done (command typed + executed + verified).
                if (
                    _is_cmd_subtask
                    and step.action_type == "key_press"
                    and (step.key or "").lower() == "enter"
                ):
                    _ok, _why = self._verify_command_effect(
                        subtask, _subtask_start, _typed_ok
                    )
                    if _ok:
                        self.log(f"  [CMD-CHECK] {_why} — command effect confirmed")
                        return True
                    last_error = _why
                    self.log(f"  [CMD-CHECK] {_why}")
                    break   # never blind-retry Enter; planner corrects the command

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

                    # Step reported as not-confirmed by the reflector.
                    # fail_threshold splits "no clear evidence of failure" from
                    # "confidently failed":
                    #   below threshold = UNCERTAIN  → see policy split below
                    #   at/above        = CLEAR FAIL → retry (if safe) or stop
                    _key = (step.key or "").lower()
                    _is_launcher_key = step.action_type == "key_press" and _key in ("winleft", "enter")
                    fail_threshold = (
                        0.95
                        if step.action_type in ("type", "hotkey") or _is_launcher_key
                        else self.reflector.min_confidence
                    )

                    if reflection.confidence < fail_threshold:
                        # UNCERTAIN — no CLEAR evidence the step failed. How we treat
                        # this depends on whether re-doing the action is safe:
                        if _non_idempotent:
                            # type / Enter / Ctrl-V already physically fired and we
                            # have NO reliable signal they failed (the reflector just
                            # couldn't *read* the result — a dark console, an OCR
                            # miss, a field it can't see). Re-doing them is exactly
                            # what caused the double-typing, double-submits, retry
                            # loops and slow runs. Accept and move on: the NEXT
                            # planning step reads the LIVE screen and corrects course
                            # if anything is actually wrong. Backstops remain —
                            # _verify_command_effect checks commands on disk,
                            # _verify_launch checks app launches, and the loop-guard
                            # stops genuine repeats.
                            self.log(
                                f"  Uncertain (conf={reflection.confidence:.2f} < "
                                f"{fail_threshold:.2f}) — action already performed; "
                                f"accepting and letting the next step verify live"
                            )
                            step_success = True
                            break
                        # Idempotent action (click / scroll): safe to repeat, and a
                        # dead click is caught RELIABLY by the screen-delta (phash)
                        # check as a high-confidence failure — so retrying here costs
                        # nothing and recovers genuinely-missed clicks.
                        last_error = (
                            f"uncertain outcome (conf={reflection.confidence:.2f}, "
                            f"threshold={fail_threshold:.2f})"
                        )
                        self.log(f"  Uncertain result — retrying ({last_error})")
                        # Fast-path: if the goal process is already running after an
                        # uncertain click, skip the remaining retries.
                        if _is_launch_goal and (
                            step.action_type in ("click", "double_click")
                        ):
                            if self._goal_confirmed(_goal_proc, _baseline_windows):
                                self.log(
                                    f"  [GOAL-CHECK-EARLY] "
                                    f"'{_goal_proc.split('.')[0]}' confirmed "
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
                        # A confidently-failed non-idempotent action must also not
                        # be blind-retried (Fix C5) — same double-execution risk.
                        if not reflection.should_retry or _non_idempotent:
                            # Definitive failure (or unsafe to retry) — stop here.
                            break
                except (ValueError, KeyError, AttributeError) as e:
                    # Fix C2: a verifier parse error means the outcome is UNKNOWN,
                    # not that the step succeeded. The old code set step_success=True
                    # here, wiring "the verifier broke" to "the action worked" — a
                    # systematic source of false successes. Treat it as an uncertain
                    # result and let the retry loop re-verify; if retries are
                    # exhausted the step is left failed so the planner can recover.
                    self.log(
                        f"  Reflection parse error ({type(e).__name__}) — "
                        f"outcome uncertain, will re-verify"
                    )
                    last_error = f"verifier parse error ({type(e).__name__})"
                    # fall through to next attempt without marking success
                except Exception as e:
                    # Infrastructure failure — retry with back-off
                    self.log(f"  Reflection error ({type(e).__name__}: {e}) — retry")
                    last_error = str(e)
                    time.sleep(1.0)

            if step_success:
                if step.action_type == "type":
                    _typed_ok = True
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
                if _is_launch_goal and self._goal_confirmed(_goal_proc, _baseline_windows):
                    _label = _goal_proc.split(".")[0]
                    self.log(f"  [GOAL-CHECK] '{_label}' confirmed — goal achieved")
                    return True

                sig = (step.action_type, step.target, step.value, step.key)
                if sig == _last_step_sig:
                    _same_step_streak += 1
                    _dedup_limit = DEDUP_LIMIT_BY_ACTION_TYPE.get(step.action_type, 2)
                    if _same_step_streak > _dedup_limit:
                        self.log(
                            f"  [LOOP-GUARD] '{step.action_type}' repeated "
                            f"{_same_step_streak + 1}× (limit={_dedup_limit}) — "
                            f"loop detected, stopping subtask"
                        )
                        # Fix C4: a loop is a strong signal the plan is not working.
                        # We still return True so a benign "planner won't emit done"
                        # loop doesn't abort an otherwise-complete task, but we mark
                        # the run degraded so it is NEVER stored as a reusable
                        # success (which would poison future routing with a plan
                        # that actually looped). See execute()'s memory gate.
                        self._degraded = True
                        # Exception: a terminal-command subtask ("run: <cmd>") that
                        # looped AND has failed steps in its history almost
                        # certainly never ran its command successfully. Declaring
                        # it complete poisons every dependent subtask (they build
                        # on a file/state that doesn't exist) — fail it honestly.
                        if "run:" in subtask.description.lower() and any(
                            c.startswith("[FAILED") for c in completed
                        ):
                            self.log(
                                "  [LOOP-GUARD] Command subtask looped after "
                                "failures — marking subtask FAILED"
                            )
                            return False
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
                if _is_launch_goal and self._goal_confirmed(_goal_proc, _baseline_windows):
                    _label = _goal_proc.split(".")[0]
                    self.log(
                        f"  [GOAL-CHECK] '{_label}' confirmed "
                        f"despite step failure — goal achieved"
                    )
                    return True

                consecutive_failures += 1
                self.log("  Step failed — re-evaluating next action")

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

    # Foreground processes where visual replanning can't help: command errors
    # need corrected text, not clicks.
    _TERMINAL_PROCS = frozenset({
        "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
        "conhost.exe", "openconsole.exe",
    })

    def _foreground_is_terminal(self) -> bool:
        import os
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False  # tests themselves run inside a terminal
        try:
            from core.capture.screen_snapshot import (
                _get_foreground_hwnd_and_title,
                _get_foreground_process,
            )
            hwnd, _ = _get_foreground_hwnd_and_title()
            proc = _get_foreground_process(hwnd)
            return proc.lower() in self._TERMINAL_PROCS
        except Exception:
            return False

    # ── Visual planning (UI-TARS recovery path) ────────────────────────────────

    def _plan_visual(self, subtask: SubTask, completed: list[str]):
        """Capture the screen and ask the VLM for the next action directly."""
        import base64
        import io
        img = self.capturer.capture()
        thumb = img.copy()
        thumb.thumbnail((960, 540))
        buf = io.BytesIO()
        thumb.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return self.planner.plan_next_step_visual(
            subtask, img_b64, completed,
            screen_w=self._screen_w, screen_h=self._screen_h,
        )

    # ── Step execution ─────────────────────────────────────────────────────────

    @staticmethod
    def _explicit_coords(value: str | None) -> tuple | None:
        """Parse an explicit "x,y" pixel pair from a step's value field.

        Visual-planner click steps (and burst steps) carry direct screen
        coordinates this way, bypassing grounding entirely.
        """
        if not value or "," not in value:
            return None
        parts = value.split(",", 1)
        if all(p.strip().lstrip("-").isdigit() for p in parts):
            return int(parts[0].strip()), int(parts[1].strip())
        return None

    # Keys that are safe to press even when the agent's own console is focused —
    # they navigate AWAY from it (launcher) or dismiss overlays, never inject
    # input into the console session itself.
    _SAFE_OWN_CONSOLE_KEYS = frozenset({"winleft", "win", "escape", "esc"})

    def _keyboard_blocked_by_own_console(self, step: ActionStep) -> bool:
        """True when keyboard input must be blocked: the foreground window is the
        agent's own host console (classic conhost). Typing there would inject
        into the very terminal session that launched the agent. Best-effort —
        under Windows Terminal the console window is a hidden ConPTY handle that
        is never foreground, so this guard stays inert there (the new-window
        launch enforcement covers that case).
        """
        import os
        if "PYTEST_CURRENT_TEST" in os.environ:
            return False  # unit tests run inside a console themselves
        if step.action_type == "key_press" and (step.key or "").lower() in self._SAFE_OWN_CONSOLE_KEYS:
            return False
        try:
            import ctypes
            own = ctypes.windll.kernel32.GetConsoleWindow()
            if not own or not ctypes.windll.user32.IsWindowVisible(own):
                return False
            if ctypes.windll.user32.GetForegroundWindow() == own:
                self.log(
                    "  [GUARD] Foreground is the agent's own console — keyboard "
                    "input blocked (would inject into the agent's host terminal)"
                )
                return True
        except Exception:
            return False
        return False

    def _execute_step(self, step: ActionStep) -> bool:
        x, y, x2, y2 = None, None, None, None

        # Never type into the agent's own host terminal session.
        if step.action_type in ("type", "key_press", "hotkey") and \
                self._keyboard_blocked_by_own_console(step):
            return False

        # Ctrl+C in a terminal interrupts the shell (it does NOT copy) — the
        # planner sometimes invents it as a "copy the error" recovery step.
        # Hard-block it; the planner must type a corrected command instead.
        if (
            step.action_type == "hotkey"
            and (step.key or "").lower().replace(" ", "") == "ctrl+c"
            and self._foreground_is_terminal()
        ):
            self.log(
                "  [GUARD] ctrl+c in a terminal interrupts the shell — blocked "
                "(type a corrected command instead)"
            )
            return False

        # ── Click family ──────────────────────────────────────────────────────
        if step.action_type in ("click", "right_click", "double_click"):
            _coords = self._explicit_coords(step.value)
            if _coords is not None:
                # Explicit pixel coordinates (visual planner / burst convention)
                x, y = _coords
                return self.actor.execute(step, x=x, y=y)
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
                self.log("  drag has no source target")
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
                self.log("  drag has no destination (value empty)")
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
                # H10: extraction found nothing. We still don't block the task
                # (the user may want partial results), but we record the miss and
                # mark the run degraded so it is not stored as a clean reusable
                # success — "read me the value" tasks that found no value must not
                # masquerade as fully-successful in memory.
                self._extracted_data[key] = ""
                self._degraded = True
                self.log(f"  [EXTRACT] nothing found for '{key}' — marked incomplete")
            return True   # extraction never blocks the step sequence

        # ── Type — run through the destructive-action firewall ─────────────────
        elif step.action_type == "type":
            if not self._firewall_allows(step.value):
                self.log(
                    "  [FIREWALL] Blocked typing a destructive command — "
                    "step aborted for safety"
                )
                return False

        return self.actor.execute(step, x=x, y=y, x2=x2, y2=y2)

    # ── Deterministic terminal-command verification ─────────────────────────────

    # Shell error fragments that indicate a command failed. Deliberately
    # specific — a bare "error" would false-positive on file contents.
    _SHELL_ERROR_MARKERS = (
        "is denied", "cannot find", "could not find", "not recognized",
        "no such file", "exception", "categoryinfo", "command not found",
        "permission denied", "syntax error", "fatal:", "access denied",
    )

    def _verify_command_effect(
        self, subtask: SubTask, started_at: float, typed_ok: bool
    ) -> tuple:
        """Ground-truth verification for "run: <command>" subtasks.

        A successful shell command usually prints NOTHING (just a new empty
        prompt) — OCR reflection mis-reads that silence as failure. Instead:
          1. delete commands  → success when the target no longer exists
          2. create/redirect  → success when the target exists AND is fresh
             (mtime within this subtask — stale files from old runs don't pass)
          3. anything else    → success when no shell error text is on screen
        Returns (ok, reason).
        """
        import os

        m = re.search(r"run:\s*(.+)$", subtask.description,
                      re.IGNORECASE | re.DOTALL)
        cmd = (m.group(1) if m else subtask.description).strip()
        time.sleep(0.8)   # let the command finish writing

        _PATH = r"(\"[^\"]+\"|'[^']+'|\S+)"

        def _norm(p: str) -> str:
            return os.path.expandvars(p.strip().strip("'\""))

        # 1. Deletion — target must be gone
        m_del = re.search(rf"^(?:del|rm|remove-item)\s+(?:-\S+\s+)*{_PATH}",
                          cmd, re.IGNORECASE)
        if m_del:
            t = _norm(m_del.group(1))
            if not os.path.exists(t):
                return True, f"'{t}' no longer exists"
            return False, f"'{t}' still exists — delete did not run or failed"

        # 2. Creation — redirect target or ni/touch/mkdir argument must exist
        #    and be fresh (modified during this subtask).
        m_new = re.search(rf">>?\s*{_PATH}", cmd) or re.search(
            rf"^(?:ni|new-item|touch|mkdir|md)\s+(?:-\S+\s+)*{_PATH}",
            cmd, re.IGNORECASE)
        if m_new:
            t = _norm(m_new.group(1))
            if os.path.exists(t):
                try:
                    fresh = os.path.getmtime(t) >= started_at - 2
                except OSError:
                    fresh = True
                if fresh:
                    return True, f"'{t}' exists on disk (freshly written)"
                return False, (
                    f"'{t}' exists but was NOT modified by this command "
                    f"(stale file from an earlier run)"
                )
            return False, (
                f"expected '{t}' on disk but it does not exist — "
                f"the command did not run or failed"
            )

        # 3. Generic command — shell silence means success
        if not typed_ok:
            return False, "Enter pressed but no command was typed first"
        try:
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            text = " ".join(
                w.text for w in self._ocr.extract(img) if w.conf >= 0.6
            ).lower()
            hit = next((mk for mk in self._SHELL_ERROR_MARKERS if mk in text), None)
            if hit:
                return False, f"error text visible in terminal output ('{hit}')"
            return True, "no error output — shell silence means success"
        except Exception:
            return True, "no error detected (output check unavailable)"

    @staticmethod
    def _subtask_save_target(subtask) -> str | None:
        """Extract the destination path from a "save ... as <path>" subtask.

        Returns the expanded path string when the subtask is a save-to-named-file
        (the description contains "save ... as <something with a / or \\>"), else
        None. Lets the orchestrator confirm the save deterministically on disk
        instead of OCR-reading a silent editor.
        """
        import os
        desc = subtask.description or ""
        m = re.search(
            r"\bsave\b[^\n]*?\bas\s+['\"]?([^'\"\n]+?)['\"]?\s*$",
            desc, re.IGNORECASE,
        )
        if not m:
            return None
        path = m.group(1).strip().rstrip(".")
        # Only a real path (has a separator + filename) is disk-verifiable.
        if "/" not in path and "\\" not in path:
            return None
        return os.path.expanduser(os.path.expandvars(path))

    def _file_saved_fresh(self, path: str, started_at: float) -> bool:
        """True if `path` exists and was written during this subtask.

        Freshness (mtime >= started_at - 2) prevents a stale file from an earlier
        run from falsely passing the save.
        """
        import os
        try:
            if not os.path.exists(path):
                return False
            return os.path.getmtime(path) >= started_at - 2
        except OSError:
            return False

    def _save_dialog_visible(self) -> bool:
        """True when a Windows Save/Save-As dialog is on screen.

        Detected by its stable static labels (immune to the dynamic content OCR
        struggles with). Used to confirm ctrl+s opened a dialog BEFORE typing the
        path — so we never type a path into the document body of an already-named
        file that ctrl+s saved silently.
        """
        try:
            img = self.capturer.capture()
            img.thumbnail((960, 540))
            text = " ".join(
                w.text for w in self._ocr.extract(img) if w.conf >= 0.6
            ).lower()
            return "file name" in text and ("save" in text or "cancel" in text)
        except Exception:
            return False

    def _try_save_as(self, target: str, started_at: float) -> bool:
        """Deterministically perform Windows "Save As <target>" and verify on disk.

        The plan→reflect loop is unreliable here: editors give no OCR-readable
        save signal, so the 8B planner loops on ctrl+s and never completes the
        dialog. The key sequence IS fixed and well-defined, so run it directly and
        confirm on disk (the only reliable signal). The path comes from the
        sub-task (model-chosen) — only the sequence is fixed. Returns False on any
        deviation so the caller falls back to the planning loop.
        """
        import time

        def _step(action_type, key=None, value=None, desc=""):
            return ActionStep(
                id=0, subtask_id=0, action_type=action_type, target=None,
                value=value, key=key, description=desc, verification="",
            )

        # The Windows Save-As filename field rejects forward slashes as invalid
        # characters ("The file name is not valid"). Type the native separator;
        # disk verification (_file_saved_fresh) still accepts either form.
        typed_path = target.replace("/", "\\")

        # Already saved (e.g. an earlier step did it) — nothing to do.
        if self._file_saved_fresh(target, started_at):
            return True

        self.log(f"  [SAVE-AS] Saving deterministically to '{typed_path}'")

        # 1. Open the Save dialog from the editor — unless one is already open
        #    (a prior step may have left it up), in which case reopening it is at
        #    best a no-op and at worst dismisses it.
        if not self._save_dialog_visible():
            if not self._execute_step(
                _step("hotkey", key="ctrl+s", desc="Open Save dialog")
            ):
                return False
            self._wait_for_settle(0.8, 3.0)

        # 2. Confirm a dialog is actually open before typing — otherwise ctrl+s may
        #    have saved silently (already-named file) and typing would corrupt the
        #    document body.
        if not self._save_dialog_visible():
            if self._file_saved_fresh(target, started_at):
                self.log(f"  [SAVE-AS] '{target}' saved (no dialog needed)")
                return True
            self.log("  [SAVE-AS] Save dialog not detected — deferring to planning loop")
            return False

        # 3. Replace the default filename with the full target path.
        self._execute_step(_step("hotkey", key="ctrl+a", desc="Select filename field"))
        if not self._execute_step(_step("type", value=typed_path, desc="Type save path")):
            return False

        # 4. Confirm. Poll the disk; a "Replace existing file?" prompt (file already
        #    exists) needs one extra Enter, sent only while the file is still absent
        #    so we never inject a stray newline into a saved document.
        self._execute_step(_step("key_press", key="enter", desc="Confirm save"))
        for i in range(8):
            time.sleep(0.4)
            if self._file_saved_fresh(target, started_at):
                self.log(f"  [SAVE-AS] '{target}' written to disk — confirmed")
                return True
            if i == 2:
                self._execute_step(
                    _step("key_press", key="enter", desc="Confirm overwrite"))
        self.log(f"  [SAVE-AS] '{target}' not on disk after save sequence — falling back")
        return False

    # ── Destructive-action firewall ─────────────────────────────────────────────

    def _firewall_allows(self, text: str | None) -> bool:
        """Return True if `text` is safe to type, or was confirmed/allowed.

        Uses a deterministic regex classifier (immune to prompt injection) plus an
        optional human confirmation handler. HIGH-severity commands are blocked
        when no handler is wired; MEDIUM commands are allowed but logged.
        """
        try:
            from core.action_firewall import Decision, Severity, decide, evaluate
        except Exception:
            return True  # never let a safety-module import error break execution
        verdict = evaluate(text)
        if not verdict.is_dangerous:
            return True
        reasons = "; ".join(verdict.matched)
        self.log(f"  [FIREWALL] {verdict.severity.value.upper()} risk detected: {reasons}")
        decision = decide(verdict, self.on_confirm)
        if decision == Decision.BLOCK:
            return False
        if verdict.severity == Severity.MEDIUM:
            self.log("  [FIREWALL] Allowed (medium risk) — proceeding")
        return True

    # ── Data extraction ────────────────────────────────────────────────────────

    def _extract_data(self, step: ActionStep) -> str | None:
        """Use OCR + LLM to extract a specific value from the current screen.
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
            resp = self.reflector.client.query_llm(messages, max_tokens=120, temperature=0.0)
            value = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            return value or None
        except Exception as e:
            logger.warning(f"[EXTRACT] Error: {e}")
            return None

    # ── Scroll-to-find ────────────────────────────────────────────────────────

    def _scroll_to_find(self, target: str) -> GroundingResult:
        """Scroll down and retry grounding after each scroll.
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
                self.log("  [SCROLL-FIND] Reached end of page — stopping")
                break
            prev_ocr_text = cur_text

            # Fast grounding only (UIA + OCR). Full ground() escalates to the
            # VLM, which forces a 30-60s model swap on small-VRAM machines —
            # per scroll, that alone can blow the whole subtask deadline.
            # Check the class dict (not the instance) so MagicMock grounders in
            # tests don't fabricate a ground_fast attribute.
            if "ground_fast" in type(self.grounder).__dict__:
                result = self.grounder.ground_fast(target)
            else:
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

    _APP_SIGNALS: dict = {
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

    # Generic words that don't make useful OCR signals on their own
    _GENERIC_NAME_WORDS = frozenset((
        "the", "a", "an", "app", "application", "browser", "program", "tool",
    ))

    def _derive_launch_signals(self, description: str) -> list[str]:
        """Fallback for apps absent from the curated _APP_SIGNALS table: derive OCR
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

    # Processes that are effectively always running, so bare "is it running"
    # checks are meaningless for confirming a launch (H9). For these we require a
    # NEW visible top-level window owned by the process instead.
    _ALWAYS_RUNNING_WINDOWS = frozenset({
        "explorer.exe",   # the Windows shell itself
    })

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

    def _count_process_windows(self, exe_name: str) -> int:
        """Count visible, non-trivial top-level windows owned by `exe_name`.

        Used both for launch confirmation (H9: explorer.exe is always running,
        so only a window proves anything) and for new-window verification when
        the app was already running before the launch subtask started (focusing
        the pre-existing window keeps the count flat; a real launch raises it).
        """
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            target = exe_name.lower()
            count = [0]

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            def _cb(hwnd, _):
                if not user32.IsWindowVisible(hwnd):
                    return True
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                if (rect.right - rect.left) < 200 or (rect.bottom - rect.top) < 120:
                    return True  # skip tray/tooltip/zero-size windows
                pid = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                h_proc = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid.value)
                if not h_proc:
                    return True
                buf = ctypes.create_unicode_buffer(260)
                try:
                    ctypes.windll.psapi.GetModuleFileNameExW(h_proc, None, buf, 260)
                except Exception:
                    try:
                        ctypes.windll.kernel32.GetModuleFileNameExW(h_proc, None, buf, 260)
                    except Exception:
                        buf.value = ""
                ctypes.windll.kernel32.CloseHandle(h_proc)
                import os as _os
                if buf.value and _os.path.basename(buf.value).lower() == target:
                    count[0] += 1
                return True

            user32.EnumWindows(_cb, 0)
            return count[0]
        except Exception:
            return 0

    def _process_has_visible_window(self, exe_name: str) -> bool:
        """True if `exe_name` owns at least one visible, non-trivial top-level window."""
        return self._count_process_windows(exe_name) > 0

    def _launch_confirmed(self, exe_name: str) -> bool:
        """Confirm an app launched. Uses window presence for always-running
        processes (H9), bare process existence otherwise.
        """
        if exe_name.lower() in self._ALWAYS_RUNNING_WINDOWS:
            return self._process_has_visible_window(exe_name)
        # For normal apps, a running process is a strong signal, but prefer a
        # visible window when we can confirm one (avoids the background-process
        # false positive). Fall back to process existence if window enumeration
        # found nothing (some apps draw via non-standard windows).
        if self._process_has_visible_window(exe_name):
            return True
        return self._is_process_running(exe_name)

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

        # Use process-based check (immune to GUI-text false positives)
        matched_proc = next(
            (self._PROCESS_MAP_WINDOWS[k] for k in self._PROCESS_MAP_WINDOWS if k in desc),
            None,
        )
        if matched_proc:
            label = matched_proc.split(".")[0]
            baseline = self._launch_window_baseline.get(matched_proc)
            for attempt, wait_s in enumerate([1.5, 2.5]):
                time.sleep(wait_s)
                if baseline is not None:
                    # App pre-existed this subtask: a bare process check is
                    # vacuous and focusing the old window must NOT pass —
                    # only a NEW window proves the launch.
                    if self._count_process_windows(matched_proc) > baseline:
                        self.log(f"  [CHECK] '{label}' NEW window confirmed")
                        return True
                elif self._launch_confirmed(matched_proc):
                    self.log(f"  [CHECK] '{label}' process confirmed running")
                    return True
                if attempt == 0:
                    self.log(f"  [CHECK] '{label}' not yet confirmed — retrying in 2.5s")
            self.log(f"  [CHECK] Launch of '{matched_proc}' not confirmed")
            return False

        # Fallback: OCR-based signal check (apps not in the curated process map)
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
        """Wait until the screen stops changing (transitions complete) or max_s elapses.
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
                    self.log("  [SETTLE] Screen stable — proceeding")
                    return
                prev_hash = cur_hash
                time.sleep(poll_interval)
            self.log(f"  [SETTLE] Max wait {max_s:.1f}s reached — proceeding anyway")
        except Exception as e:
            logger.debug(f"[SETTLE] Error: {e} — falling back to fixed wait")
            time.sleep(max_s)

    # ── Screen context ─────────────────────────────────────────────────────────

    def _get_screen_context(self) -> str:
        """Return a structured screen context string for the planner.

        Uses capture_snapshot() to separate foreground UI text from background
        window text so the planner can focus on interactable elements only.
        Falls back to a flat token list if snapshot capture fails.
        """
        try:
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

    def _topological_sort(self, subtasks: list[SubTask]) -> list[SubTask]:
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
