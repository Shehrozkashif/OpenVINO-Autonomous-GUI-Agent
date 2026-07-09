# core/orchestrator.py
"""Task Orchestrator — the central coordinator.

Execution flow:
  instruction
    → elicit missing parameters from the user (time, recipient, …)
    → Router.decompose() → SubTasks (topologically sorted)
    → for each SubTask (dynamic loop, sees live screen every step):
        launch-skip: "open X" is done if X already owns the foreground
        screen context = OCR + CLICKABLE CONTROLS (accessibility tree,
          disabled controls labeled) + SYSTEM CLOCK + goal-check evidence
        goal check: one LLM yes/no — is this subtask ALREADY satisfied?
          (cached per screen state; a confident yes completes the subtask)
        plan_next_step() → ActionStep
        [click]          Grounding.ground() → (x, y); UIA-grounded clicks
                         try pattern invoke first, pixel click as fallback
                         (invoke blacklisted per target after a failed verify)
        [grounding miss] _scroll_to_find(); misses cached per screen state
        [set_value]      UIA ValuePattern, else focus + type + read-back
        [extract]        _extract_data() → LLM reads OCR → stores value
        Action.execute(step)  (action firewall vets typed text first)
        Reflection.verify(step): phash delta → LLM on OCR (+ focused-control
          ground truth) → VLM screenshot check on uncertainty
        failures: dead-point + invoke blacklists, app-anchor refocus when a
          click escapes to another process, replan with the objective pinned
    → Router.summarize_completion() (+ [BLOCKER] evidence when failed)
    → return extracted_data alongside success/failure
"""
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from loguru import logger

from agents.action import ActionExecutionAgent
from agents.grounding import GroundingResult, UIGroundingAgent
from agents.planning import PlanningAgent, PlanningParseError
from agents.reflection import ReflectionAgent
from agents.router import RouterAgent
from core.capture.screen_snapshot import capture_snapshot
from core.capture.screenshot import OCR_THUMB, ScreenCapture, _screen_size
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
    # task_deadline_s is a FLOOR: the effective budget scales with plan size
    # (n_subtasks × subtask_deadline_s) so a legitimate 8-subtask task is not
    # killed by a cap tuned for 2-subtask tasks. See _effective_task_deadline.
    task_deadline_s: float = 600.0      # minimum budget for a whole instruction
    subtask_deadline_s: float = 240.0   # hard cap on a single subtask
    # After this many consecutive step failures, planning escalates from the
    # text path (OCR context → LLM) to the visual path (screenshot → UI-TARS),
    # which sees icons/layout the text path is blind to. 0 disables escalation.
    visual_replan_after: int = 2
    # When a subtask fails, the router is re-consulted with the completed state
    # to produce a fresh plan for the remaining work (different approach) this
    # many times before the task is declared failed. 0 disables replanning.
    max_task_replans: int = 2


# Per-action-type limit on identical successful steps within ONE subtask —
# counted as a TOTAL, not consecutively, so A-B-A-B ping-pong loops (e.g. a
# hallucinated verifier success alternating between two clicks) trip it too.
# Stricter for high-signal actions (type, right_click); lenient for nav keys.
DEDUP_LIMIT_BY_ACTION_TYPE: dict[str, int] = {
    "type":        1,   # same text typed twice → almost certainly a loop
    "click":       2,   # allow up to 2 repeats (e.g. re-focusing a window)
    "key_press":   3,   # allow up to 3 repeats (e.g. multiple Escape/arrow presses)
    "right_click": 1,   # context menu should open once
    "invoke":      1,   # a button pressed twice → double submit
    "set_value":   1,   # same field set to the same value twice → a loop
    "select":      2,   # re-selecting is harmless but twice is enough
}


@dataclass
class _SubtaskRun:
    """Mutable state threaded through one _execute_subtask() step loop.

    Bundling it in one object lets each phase of the loop live in its own
    named method (_plan_next_step, _run_step_attempts, _judge_reflection,
    _record_step_success/_failure) instead of one interleaved 500-line body.
    """

    started_at: float
    # Ground-truth facts about the subtask, fixed at setup time.
    is_launch_goal: bool = False           # "open <app>" — verified via process list
    goal_proc: str = ""                    # expected process name for the launch goal
    baseline_windows: int = 0              # window count before the launch attempt
    is_cmd_subtask: bool = False           # "run: <cmd>" — verified on disk/terminal
    save_target: str = ""                  # "save as <path>" — verified on disk
    type_payload: str = ""                 # "type: <text>" — done once typed+verified
    # Rolling execution state, updated as steps run.
    completed: list = field(default_factory=list)   # step history shown to the planner
    consecutive_failures: int = 0
    cached_ocr: str = ""                   # reuse reflection's OCR for the next plan
    screen_context: str = ""               # OCR context the current step was planned on
    step_sig_counts: dict = field(default_factory=dict)  # (action_type, target, value, key) → successes
    goal_check_cache: dict = field(default_factory=dict)  # hash(screen ctx) → (ok, evidence)
    typed_ok: bool = False                 # a type step succeeded in this subtask
    last_error: str = ""                   # most recent step failure reason
    # Steps planned ahead by the last plan_steps() call. Executing from this
    # queue skips the ~7-10 s planning call per step; it is flushed whenever
    # a step fails or its outcome is uncertain, forcing a fresh plan against
    # the live screen.
    step_queue: list = field(default_factory=list)


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
        on_ask: Callable[[str], str | None] | None = None,
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
        # Optional question handler for missing required parameters (a meeting
        # time, an email recipient). Signature: on_ask(question) -> answer|None.
        # When None the agent proceeds with the instruction as given.
        self.on_ask = on_ask
        self._stop_event = threading.Event()
        self._invoke_dead = set()   # targets whose pattern-invoke provably no-ops
        self._last_was_invoke = False
        self._last_goal_evidence = ""
        self._last_controls = None   # controls snapshot for verifier diffing
        self._exec_fail_reason = ""
        self._blocking_overlay = None   # (target, cover) from the hit-test gate
        # Set when a subtask completes via a degraded path (loop-guard recovery,
        # optimistic parse-failure success, etc.). Such a task is NOT stored in
        # success memory, so broken plans never poison future routing hints.
        self._degraded = False

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
        self._app_anchor = None   # (hwnd, pid, exe) of the task's app window
        self._invoke_dead = set()   # targets whose pattern-invoke provably no-ops
        self._last_goal_evidence = ""
        self._last_controls = None
        self._blocking_overlay = None
        self.log(f"[TASK START] '{instruction}'")
        start_time = time.time()
        self._arm_kill_switch()

        # Ask the user for required details the instruction omits (a meeting
        # time, a recipient) BEFORE any planning — guessing them wastes the
        # whole run. No-op when no question handler is wired.
        instruction = self._elicit_missing_parameters(instruction)

        # Memory hint for the router
        memory_hint = None
        try:
            similar = self.memory.find_similar(instruction, threshold=0.80)
            if similar:
                descs = [s.get("description", "") for s in similar.get("steps", [])]
                # "Reuse if it fits" alone let a stored plan REPLACE proper
                # decomposition: a past run's subtask list became the whole
                # plan — no launch, no Save, no invite (live 13:05 run).
                memory_hint = (
                    f"Similar past task (succeeded): '{similar['instruction']}'. "
                    f"Subtasks used: {descs}. Treat this only as a hint for "
                    f"approach — your plan must still cover the ENTIRE current "
                    f"instruction end to end (launching the app, every value "
                    f"it names, and the final save/send/confirm step)."
                )
                self.log(f"[MEMORY] Similar past task found (sim={similar['similarity']:.2f})")
        except Exception:
            pass

        # Resume hint — a recent interrupted run of this exact instruction left
        # a checkpoint of what already completed; the router plans the rest.
        try:
            resume_descs = self.memory.load_checkpoint(instruction)
        except Exception:
            resume_descs = None
        if resume_descs:
            done = "; ".join(resume_descs)
            resume_hint = (
                f"RESUME: an earlier run of this exact task was interrupted after "
                f"completing these sub-tasks (their effects are already on the "
                f"machine — do NOT repeat them): {done}. Plan ONLY the remaining work."
            )
            memory_hint = f"{memory_hint}\n{resume_hint}" if memory_hint else resume_hint
            self.log(
                f"[RESUME] Checkpoint found — {len(resume_descs)} subtask(s) "
                f"already completed in a previous run"
            )

        screen_context = self._get_screen_context()

        task_id, subtasks = self.router.decompose(
            instruction, screen_context=screen_context, memory_hint=memory_hint
        )
        self.log(f"[ROUTER] {len(subtasks)} sub-task(s)")

        queue: list[SubTask] = self._topological_sort(subtasks)
        # H8, adaptive: budget scales with plan size so long multi-step tasks
        # are not killed by a cap tuned for short ones.
        task_deadline_s = self._effective_task_deadline(len(queue))
        if task_deadline_s:
            self.log(
                f"[BUDGET] Task deadline {task_deadline_s:.0f}s "
                f"for {len(queue)} subtask(s)"
            )

        completed_subtask_ids = []
        completed_subtask_descs: list[str] = []   # for inter-subtask context
        executed_subtasks: list[SubTask] = []     # what actually ran (incl. replans)
        skipped_descs: list[str] = []             # permanently-failed, skipped work
        failed = False
        replans_used = 0

        while queue:
            subtask = queue.pop(0)
            if self._stop_event.is_set():
                self.log("[TASK] Stopped by user.")
                failed = True
                break

            # H8: enforce the (adaptive) task-level wall-clock budget.
            if task_deadline_s and (time.time() - start_time > task_deadline_s):
                self.log(
                    f"[TASK] Deadline ({task_deadline_s:.0f}s) exceeded "
                    f"— aborting before subtask {subtask.id}"
                )
                failed = True
                break

            self.log(f"\n[SUBTASK {subtask.id}] {subtask.description}")
            success = self._run_subtask_with_launch_retry(
                subtask, completed_subtask_descs
            )

            if not success and not self._stop_event.is_set() and (
                replans_used < self.config.max_task_replans
            ):
                # Task-level recovery: hand the completed state back to the
                # router for a fresh plan of the remaining work, instead of
                # throwing away everything done so far.
                replans_used += 1
                self.log(
                    f"[REPLAN] Subtask {subtask.id} failed — re-planning remaining "
                    f"work ({replans_used}/{self.config.max_task_replans})"
                )
                new_queue = self._replan_remaining(
                    instruction, list(completed_subtask_descs), subtask,
                    executed_subtasks, queue,
                )
                if new_queue:
                    queue = new_queue
                    # Extend the budget to cover the replanned remainder.
                    if task_deadline_s and self.config.subtask_deadline_s:
                        task_deadline_s = max(
                            task_deadline_s,
                            (time.time() - start_time)
                            + len(queue) * self.config.subtask_deadline_s,
                        )
                    self.log(
                        f"[REPLAN] Continuing with {len(new_queue)} recovery subtask(s)"
                    )
                    continue
                self.log("[REPLAN] Router could not produce a recovery plan")

            if success:
                completed_subtask_ids.append(subtask.id)
                completed_subtask_descs.append(subtask.description)
                executed_subtasks.append(subtask)
                self.log(f"[SUBTASK {subtask.id}] Complete")
                # A successful "open <app>" anchors the task to the window
                # that now owns the foreground — later subtasks act on it.
                self._maybe_set_app_anchor(subtask.description)
                # Checkpoint after every completed subtask so an interrupted or
                # failed long task resumes here instead of starting over.
                try:
                    self.memory.save_checkpoint(instruction, list(completed_subtask_descs))
                except Exception:
                    pass
                if queue:
                    self._wait_for_settle(min_s=0.5, max_s=3.0)
            else:
                # Replans exhausted (or none produced). When work is still
                # QUEUED behind this subtask, dying here throws it away —
                # live: the time-zone hunt burned the whole replan budget and
                # the already-filled form was never Saved. Skip the stuck
                # subtask, mark the run degraded, and let the queued work
                # run; the summary names what was skipped. A subtask with
                # NOTHING queued after it IS the remaining objective —
                # skipping that would fake success, so it still fails.
                if queue and not self._stop_event.is_set():
                    self._degraded = True
                    skipped_descs.append(subtask.description)
                    self.log(
                        f"[SKIP] Subtask {subtask.id} cannot be completed — "
                        f"continuing with the {len(queue)} remaining "
                        f"subtask(s) so finished work is not thrown away"
                    )
                    continue
                # A subtask returns failure the moment Stop is pressed mid-step;
                # report that as a user stop (not a task failure) so the UI shows
                # "Stopped", the summary reads right, and the run is not recorded
                # as a genuine failure pattern.
                if self._stop_event.is_set():
                    self.log("[TASK] Stopped by user.")
                else:
                    self.log(f"[SUBTASK {subtask.id}] Failed — stopping task")
                failed = True
                break

        self._disarm_kill_switch()

        elapsed = time.time() - start_time
        blocker = getattr(self, "_last_goal_evidence", "") if failed else ""
        if blocker:
            self.log(f"[BLOCKER] {blocker}")
        for d in skipped_descs:
            self.log(f"[SKIPPED] {d}")
        summary = self.router.summarize_completion(
            task_id, completed_subtask_ids, not failed, blocker=blocker,
            skipped=skipped_descs,
        )

        # Append any extracted data to the summary log
        if self._extracted_data:
            lines = "\n".join(f"  {k}: {v}" for k, v in self._extracted_data.items())
            self.log(f"\n[EXTRACTED DATA]\n{lines}")

        self.log(f"\n[TASK DONE] {summary} ({elapsed:.1f}s)")

        # Only persist as a reusable success when the task genuinely completed
        # AND no subtask finished via a degraded path (loop-guard recovery,
        # optimistic parse-failure success). Storing degraded runs would poison
        # future routing hints with plans that did not actually work.
        # executed_subtasks (not the original decomposition) is stored — after a
        # replan they are the plan that actually worked.
        if not failed and not self._degraded:
            self.memory.store_successful_task(
                instruction, executed_subtasks or subtasks, elapsed
            )
        elif not failed and self._degraded:
            self.log("[MEMORY] Task completed via a degraded path — not stored as a reusable success")

        # A finished task needs no resume checkpoint; a failed one keeps it so
        # the next run of the same instruction continues from here.
        if not failed:
            try:
                self.memory.clear_checkpoint(instruction)
            except Exception:
                pass

        return {
            "task_id": task_id,
            "success": not failed,
            "subtasks_completed": completed_subtask_ids,
            "elapsed_s": elapsed,
            "summary": summary,
            "extracted_data": dict(self._extracted_data),
        }

    # ── Long-task support: budgets, replanning, resume, clarification ─────────

    def _effective_task_deadline(self, n_subtasks: int) -> float:
        """Task budget scaled to plan size.

        task_deadline_s acts as a floor; each subtask brings its own
        subtask_deadline_s of budget, so an 8-subtask meeting-scheduling task
        gets the time it legitimately needs while a stuck 2-subtask task still
        dies quickly. Returns 0 when budgets are disabled.
        """
        if not self.config.task_deadline_s:
            return 0.0
        if not self.config.subtask_deadline_s:
            return self.config.task_deadline_s
        return max(
            self.config.task_deadline_s,
            n_subtasks * self.config.subtask_deadline_s,
        )

    def _run_subtask_with_launch_retry(
        self, subtask: SubTask, completed_subtask_descs: list[str]
    ) -> bool:
        """Execute one subtask; on an unconfirmed app launch, retry once with an
        explicit use-the-search-launcher hint (the icon-click route failed).
        """
        success = self._execute_subtask(subtask, task_context=completed_subtask_descs)
        if success:
            success = self._verify_launch(subtask)
            if not success:
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
        return success

    def _replan_remaining(
        self,
        instruction: str,
        completed_descs: list[str],
        failed_subtask: SubTask,
        executed_subtasks: list[SubTask],
        old_queue: list[SubTask],
    ) -> list[SubTask]:
        """Ask the router for a fresh plan of the FAILED subtask's work; the
        subtasks queued after it are preserved verbatim.

        The replan LLM only ever rewrites the stuck part. Letting it re-derive
        "everything remaining" from the instruction silently dropped the final
        'click Save and send the invitation' subtask on a live run — the form
        was filled perfectly and never saved. Downstream work is appended
        deterministically; if the rewrite made it obsolete, its own goal check
        skips it in one cycle.

        Returns the new subtask queue (ids renumbered above every id used so
        far, so logs and dependency references never collide), or [] when the
        router cannot help — the caller then fails the task as before.
        """
        try:
            screen_context = (
                self._get_screen_context() + self._clickable_controls_block()
            )
            fresh = self.router.replan(
                instruction, completed_descs, failed_subtask.description,
                screen_context=screen_context,
                pending_descs=[s.description for s in old_queue],
            )
        except Exception as e:
            self.log(f"  [REPLAN] Router error: {e}")
            return []
        if not fresh:
            return []
        watermark = max(
            [s.id for s in executed_subtasks]
            + [s.id for s in old_queue]
            + [failed_subtask.id, 0]
        )
        renumbered = [
            SubTask(
                id=watermark + st.id,
                description=st.description,
                depends_on=[watermark + d for d in st.depends_on],
            )
            for st in fresh
        ]
        last_new = max(s.id for s in renumbered)
        # Re-append the untouched downstream queue after the rewrite, deps
        # rewired: the failed id now means "the rewrite finished", and ids
        # stay above every new one so the ID-order fallback keeps the order.
        id_map = {failed_subtask.id: last_new}
        for i, st in enumerate(old_queue):
            id_map[st.id] = last_new + 1 + i
        kept = [
            SubTask(
                id=id_map[st.id],
                description=st.description,
                depends_on=[id_map.get(d, last_new) for d in st.depends_on],
            )
            for st in old_queue
        ]
        return self._topological_sort(renumbered + kept)

    def _elicit_missing_parameters(self, instruction: str) -> str:
        """Ask the user (via on_ask) for required details the instruction omits,
        and fold the answers back into the instruction. Proceeds unchanged when
        no handler is wired, nothing is missing, or the user declines to answer.
        """
        if self.on_ask is None:
            return instruction
        try:
            questions = self.router.missing_parameters(instruction)
        except Exception:
            return instruction
        if not questions:
            return instruction
        answers = []
        for q in questions:
            self.log(f"[CLARIFY] {q}")
            try:
                ans = self.on_ask(q)
            except Exception:
                ans = None
            if ans and ans.strip():
                answers.append(f"{q} -> {ans.strip()}")
        if answers:
            instruction = (
                f"{instruction} (details provided by the user: "
                + "; ".join(answers) + ")"
            )
            self.log(f"[CLARIFY] Instruction enriched with {len(answers)} detail(s)")
        return instruction

    # ── Task app anchor ───────────────────────────────────────────────────────
    # Ground truth for "which window are we working in". After a successful
    # "open <app>" subtask, the window owning the foreground IS the launched
    # app — no name→process mapping needed, works for any app. Later subtasks
    # verify the anchor still owns the foreground before planning/acting; if
    # another window is on top (seen live: a leftover Edge error page swallowed
    # every action while Outlook sat behind it), the anchor is refocused first.

    _LAUNCH_DESC_RX = re.compile(r"^\s*(?:open|launch|start)\b", re.IGNORECASE)

    @staticmethod
    def _foreground_app() -> tuple[int, int, str]:
        """(hwnd, pid, exe_name) of the current foreground window; zeros on failure."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return 0, 0, ""
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            name = ""
            h = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
            if h:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    name = buf.value.rsplit("\\", 1)[-1]
                kernel32.CloseHandle(h)
            return hwnd, pid.value, name
        except Exception:
            return 0, 0, ""

    @staticmethod
    def _window_title(hwnd: int) -> str:
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
            return buf.value or ""
        except Exception:
            return ""

    def _launch_already_satisfied(self, subtask: SubTask) -> bool:
        """Ground-truth launch skip: "open X" when X already owns the foreground.

        The app name is parsed from the subtask's own text and matched against
        the foreground window's title and process name — no app catalogue.
        Launching blindly when the app is already in front is actively harmful
        (seen live: Start-menu Enter landed on the WEB result and opened a
        browser page over the real Outlook). When satisfied, the window is
        anchored so the rest of the task acts on it.
        """
        desc = subtask.description
        if not self._LAUNCH_DESC_RX.match(desc.lower()):
            return False
        # Apps with mapped processes keep the stricter launch semantics in
        # _setup_launch_goal (a pre-existing terminal may be busy running
        # another program — reusing it is the documented H-bug). The skip is
        # for everything else, where "it's already in front" means done.
        if any(k in desc.lower() for k in self._PROCESS_MAP_WINDOWS):
            return False
        signals = self._derive_launch_signals(desc)
        if not signals:
            return False
        hwnd, pid, name = self._foreground_app()
        if not hwnd or hwnd == getattr(self, "_own_hwnd", None):
            return False
        title = self._window_title(hwnd)
        hay = f"{title} {name}".lower()
        if not any(s.lower() in hay for s in signals if len(s) >= 3):
            return False
        self._app_anchor = (hwnd, pid, name)
        self.log(
            f"  [LAUNCH-SKIP] '{signals[0]}' already owns the foreground "
            f"('{title[:60]}') — anchored it, nothing to launch"
        )
        logger.info(
            f"[ORCHESTRATOR] Launch skipped: '{signals[0]}' foreground "
            f"(title='{title[:60]}', proc={name}) — anchored hwnd={hwnd}"
        )
        return True

    def _maybe_set_app_anchor(self, description: str):
        """Record the foreground window as the task's app anchor after a
        successful launch subtask ("open X" / "launch X" / "start X").
        """
        desc = description.lower()
        if not self._LAUNCH_DESC_RX.match(desc) or "already" in desc:
            return
        hwnd, pid, name = self._foreground_app()
        if not hwnd or hwnd == getattr(self, "_own_hwnd", None):
            return
        self._app_anchor = (hwnd, pid, name)
        self.log(f"  [ANCHOR] Task app window = {name} (hwnd={hwnd})")
        logger.info(f"[ORCHESTRATOR] App anchor set: {name} hwnd={hwnd}")

    def _ensure_anchor_foreground(self):
        """Bring the task's app window back to the foreground if something
        else has covered it. Same-process windows (dialogs, pickers) count as
        the anchor being active. Drops the anchor when its window is gone.
        """
        anchor = getattr(self, "_app_anchor", None)
        if not anchor:
            return
        hwnd, pid, name = anchor
        try:
            import ctypes
            user32 = ctypes.windll.user32
            if not user32.IsWindow(hwnd):
                logger.info(f"[ORCHESTRATOR] App anchor window gone ({name}) — cleared")
                self._app_anchor = None
                return
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                return
            fg_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))
            if fg_pid.value == pid:
                return   # a dialog/popup of the same app — still our context
            self.log(f"  [FOCUS] Foreground is not {name} — refocusing the task app")
            logger.info(f"[ORCHESTRATOR] Refocusing task app {name} (hwnd={hwnd})")
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)   # SW_RESTORE
            if not user32.SetForegroundWindow(hwnd):
                # SetForegroundWindow is refused when we lack foreground
                # rights; SwitchToThisWindow (alt-tab semantics) still works.
                user32.SwitchToThisWindow(hwnd, True)
            time.sleep(0.6)   # let the window paint before capture/grounding
        except Exception:
            pass

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

        This method is only the skeleton; each phase of the loop is a named
        method so it can be read (and tested) in isolation:
          _plan_next_step        → pick the next action (queue / text / visual)
          _run_step_attempts     → execute it under the retry + verification policy
          _record_step_success   → history, early-exit checks, loop guard
          _record_step_failure   → history, failure memory, abort threshold
        """
        # Ground truth fast path: "open X" when X is already the foreground
        # window — anchor it and skip the launch entirely (zero LLM calls,
        # zero keystrokes). See _launch_already_satisfied.
        if self._launch_already_satisfied(subtask):
            return True

        _is_launch_goal, _goal_proc, _baseline_windows, task_context = (
            self._setup_launch_goal(subtask, task_context)
        )
        run = _SubtaskRun(
            started_at=time.time(),
            is_launch_goal=_is_launch_goal,
            goal_proc=_goal_proc,
            baseline_windows=_baseline_windows,
            # Terminal-command subtasks get DETERMINISTIC verification: a
            # successful shell command prints nothing (new empty prompt), which
            # OCR-based reflection systematically mis-reads as "no change →
            # failed". Check the real world instead: file created/deleted on
            # disk, or error text in the terminal output.
            is_cmd_subtask="run:" in subtask.description.lower(),
            # "Save the document as <path>" subtasks are ALSO file-producing,
            # but through a GUI dialog. Editors give no OCR-readable "saved"
            # signal, so reflection loops on ctrl+s forever. Verify the same
            # way as commands: the subtask is DONE the moment the named file
            # appears on disk (fresh).
            save_target=self._subtask_save_target(subtask),
        )
        # "... type: <text>" subtasks are complete the moment that text has been
        # typed and verified. Without this, the planner re-types the payload
        # (duplicating it in the document) or drifts into ctrl+s save steps that
        # belong to a later subtask — burning the whole subtask budget.
        run.type_payload = None if run.save_target else self._subtask_type_payload(subtask)

        # The Save-As key sequence is fixed and well-defined, but the planner
        # is unreliable at threading the dialog (it loops on ctrl+s and never
        # types the path). Run it deterministically first and confirm on disk;
        # only the key sequence is fixed — the path is model-chosen. On any
        # failure, fall through to the normal planning loop.
        if run.save_target and self._try_save_as(run.save_target, run.started_at):
            return True

        for step_idx in range(self.config.max_steps_per_subtask):
            if self._stop_event.is_set():
                return False

            # H8: enforce the per-subtask wall-clock budget.
            if self.config.subtask_deadline_s and (
                time.time() - run.started_at > self.config.subtask_deadline_s
            ):
                self.log(
                    f"  Subtask deadline ({self.config.subtask_deadline_s:.0f}s) "
                    f"exceeded — aborting subtask"
                )
                return False

            # A "save as <path>" subtask is COMPLETE the instant the file lands on
            # disk. Check before planning so a successful save short-circuits the
            # ctrl+s retry loop (editors show no readable confirmation).
            if run.save_target and self._file_saved_fresh(run.save_target, run.started_at):
                self.log(
                    f"  [SAVE-CHECK] '{run.save_target}' on disk (fresh) — save confirmed"
                )
                return True

            outcome, step = self._plan_next_step(run, subtask, task_context)
            if outcome == "done":
                return True
            if outcome == "abort":
                return False
            if outcome == "retry":
                continue

            self.log(f"  Step {step_idx + 1}: [{step.action_type}] {step.description}")

            attempt_outcome = self._run_step_attempts(run, subtask, step)
            if attempt_outcome == "subtask_done":
                return True

            decision = (
                self._record_step_success(run, subtask, step)
                if attempt_outcome == "step_ok"
                else self._record_step_failure(run, step)
            )
            if decision is not None:
                return decision

        self.log(f"  MAX_STEPS ({self.config.max_steps_per_subtask}) reached")
        return False

    def _note_planning_failure(self, run: "_SubtaskRun") -> str:
        """Count a planning failure; abort once the consecutive limit is hit."""
        run.consecutive_failures += 1
        if run.consecutive_failures >= self.config.consecutive_failures_limit:
            self.log(
                f"  {self.config.consecutive_failures_limit} consecutive failures"
                f" — aborting subtask"
            )
            return "abort"
        return "retry"

    def _clickable_controls_block(self) -> str:
        """The CLICKABLE CONTROLS prompt block from the accessibility tree,
        or "" when unavailable. Shared by step planning and replanning —
        WebView2 apps are nearly invisible to OCR, so an LLM that only gets
        OCR text plans blind (a replanner kept emitting 'click Save' while
        the empty required fields sat unreadable right next to it).
        """
        try:
            from core.windows_uia import get_interactive_elements
            _elems = get_interactive_elements(max_elements=60, timeout_s=3.0)
            if not _elems:
                return ""
            # Snapshot for the verifier: controls that appear/disappear
            # after an action are tree-level proof of its effect.
            self._last_controls = {f"'{n}' [{t}]" for n, t in _elems}
            _names = ", ".join(f"'{n}' [{t}]" for n, t in _elems)
            # Bound the prompt cost: ~3.6 KB ≈ 850 tokens ≈ seconds of
            # prefill per planning cycle on the iGPU.
            if len(_names) > 3600:
                _names = _names[:3600].rsplit(", ", 1)[0] + ", …(truncated)"
            logger.info(
                f"[PLANNING] UIA clickable controls ({len(_elems)}): {_names}"
            )
            return (
                "\nCLICKABLE CONTROLS (exact names from the Windows "
                "accessibility tree — choose click targets ONLY from "
                "this list, verbatim; = '…' after a field is its CURRENT "
                "content, ground truth for what is already filled in): "
                + _names
            )
        except Exception:
            return ""

    def _goal_already_satisfied(
        self, run: "_SubtaskRun", subtask: SubTask,
        skip_exclusions: bool = False,
    ) -> bool:
        """One focused LLM question: does the current screen already show the
        state this subtask is trying to reach?

        Judged against ground truth the planner also sees — OCR text plus the
        CLICKABLE CONTROLS list from the accessibility tree. Conservative by
        contract: only a confident, evidence-backed YES skips work; anything
        else (uncertainty, parse failure, missing values) plans normally.
        Subtask kinds with deterministic completion checks are excluded.
        """
        if not skip_exclusions and (
                run.is_launch_goal or run.is_cmd_subtask
                or run.save_target or run.type_payload):
            return False
        # Same screen → same answer: don't re-pay a 6-10 s LLM call to re-ask
        # a question about identical pixels. Keyed on the full context string
        # (OCR + controls list), so any real change misses the cache.
        ctx_key = hash(run.screen_context)
        if ctx_key in run.goal_check_cache:
            ok, evidence = run.goal_check_cache[ctx_key]
            if not ok and evidence:
                run.screen_context += (
                    f"\nGOAL CHECK (what is still missing): {evidence}"
                )
            return ok
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You check whether a desktop-automation subtask is "
                        "ALREADY complete before any further action is taken. "
                        "Judge ONLY from the screen evidence provided. Reply "
                        "with strict JSON: {\"satisfied\": true|false, "
                        "\"confidence\": 0.0-1.0, \"evidence\": \"<what on "
                        "screen proves it>\"}. satisfied=true ONLY if the end "
                        "state the goal aims at is fully present (e.g. the "
                        "goal is to open a form and that form's controls are "
                        "on screen). In evidence, go REQUIREMENT BY "
                        "REQUIREMENT: name each value/state the goal asks for "
                        "and quote what on screen shows it present or "
                        "missing. satisfied=true only when NONE are missing "
                        "— a filled title never compensates for a wrong time "
                        "zone. When uncertain, satisfied=false."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Subtask goal: {subtask.description}\n"
                        # The check judges END STATE, not whether the action
                        # "still needs doing" — without this line it re-failed
                        # an OPEN form because "the button was not clicked",
                        # and the planner re-clicked until Escape closed it.
                        + (
                            f"Last verified action in this subtask: "
                            f"{_last_done} (it succeeded — judge whether its "
                            f"end state is on the screen below, not whether "
                            f"the action should be repeated).\n"
                            if (_last_done := next(
                                (c for c in reversed(run.completed)
                                 if not c.startswith("[FAILED")), ""))
                            else ""
                        )
                        + f"\nCurrent screen:\n{run.screen_context}"
                    ),
                },
            ]
            resp = self.reflector.client.query_llm(
                messages, max_tokens=280, temperature=0.0
            )
            text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return False
            verdict = json.loads(m.group(0))
            ok = (
                bool(verdict.get("satisfied"))
                and float(verdict.get("confidence", 0.0)) >= 0.8
            )
            # A "not yet" verdict names exactly what's still missing (e.g.
            # "subject is set, but start time is not 3:00 PM"). Hand that to
            # the planner in the same cycle so it works on the missing part
            # instead of redoing the whole subtask from step one.
            evidence = str(verdict.get("evidence", "")).strip()[:400]
            run.goal_check_cache[ctx_key] = (ok, evidence)
            if not ok and evidence:
                run.screen_context += (
                    f"\nGOAL CHECK (what is still missing): {evidence}"
                )
                # Keep the newest evidence for the final task report: when a
                # run dies grinding against a screen that lacks the needed UI
                # (e.g. an app stuck at its sign-in window), this line is the
                # difference between "task failed" and knowing WHY.
                self._last_goal_evidence = evidence
            return ok
        except Exception as e:
            logger.debug(f"[GOAL-CHECK] Error: {e}")
            return False

    def _plan_next_step(
        self, run: "_SubtaskRun", subtask: SubTask, task_context: list[str],
    ) -> tuple[str, ActionStep | None]:
        """Choose the next action: queued step, fresh text plan, or visual escalation.

        Returns (outcome, step):
          ("step", step)  — an action to execute now
          ("done", None)  — planner/VLM says the goal is achieved
          ("retry", None) — planning failed; the loop may try again
          ("abort", None) — consecutive planning failures exhausted the budget
        """
        # The task's app window must own the foreground before we read the
        # screen and plan against it — otherwise every action lands on
        # whatever window happens to be on top. Launch subtasks are creating
        # that window, so they skip the check.
        if not run.is_launch_goal:
            self._ensure_anchor_foreground()
        run.screen_context = run.cached_ocr if run.cached_ocr else self._get_screen_context()
        run.cached_ocr = ""   # consume — will be refreshed after reflection

        # Ground-truth clickables from the accessibility tree. Without this
        # the planner INVENTS button names from world knowledge — it planned
        # "New Meeting" for a whole run while the app's real button was named
        # differently, and every grounding stage honestly failed to find the
        # imaginary name. Click targets must come from names that exist.
        run.screen_context += self._clickable_controls_block()

        # A hit-test-proven overlay goes into the SCREEN CONTEXT, the one
        # channel the planner reliably follows — the same fact in the step
        # failure record was ignored for six straight cycles while the
        # overlay's own 'Close' button sat in the controls list.
        _overlay = getattr(self, "_blocking_overlay", None)
        if _overlay:
            run.screen_context += (
                f"\nBLOCKING OVERLAY (hit-test ground truth): clicks on "
                f"'{_overlay[0]}' physically land on '{_overlay[1]}' — a "
                f"popup/overlay is on top of the form. NOTHING beneath it "
                f"can react until it is dismissed via its own control "
                f"(Close / Cancel / OK / X) from the list above."
            )
            self._blocking_overlay = None   # re-set by the next blocked click

        # Ask "is this goal ALREADY met?" before planning another action.
        # Without this, "click X to open Y" subtasks loop after they succeed:
        # once Y is open, every further click is a no-op that state-describing
        # verifiers keep blessing ("the form is visible → success"), and the
        # planner re-plans the same click instead of returning []. Launches,
        # commands, saves and typing have their own deterministic checks.
        # (Runs BEFORE the clock line is appended: the check is cached per
        # screen-context and the clock changes every minute.)
        # Skipped while queued steps remain: the queue was planned against
        # this context moments ago and the planner is not consulted between
        # queue pops — paying 10-16 s of LLM per queued step re-answers a
        # question nobody reads (live: ~1 min of pure goal checks per form).
        if not run.step_queue and self._goal_already_satisfied(run, subtask):
            self.log("  [GOAL-CHECK] Goal already satisfied on current screen")
            return ("done", None)

        # Real date/time from the OS — the planner otherwise reasons from the
        # model's frozen sense of "now" (it clicked 'Today' to select a date
        # two days ahead, silently resetting a correctly-set date).
        run.screen_context += time.strftime(
            "\nSYSTEM CLOCK (ground truth): %A, %B %d, %Y %I:%M %p"
        )

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
            and run.consecutive_failures >= self.config.visual_replan_after
        )
        # A "run: <cmd>" subtask is fixed by a CORRECTED COMMAND, never by the
        # VLM clicking/typing into the console. Escalating it to visual mode
        # re-types the command (corrupting the prompt line) and burns a 30-60s
        # model swap. Block it deterministically on is_cmd_subtask — do not
        # rely on foreground-process detection, which can misfire right after
        # an Enter press and let the destructive re-type through.
        if _visual_mode and (run.is_cmd_subtask or self._foreground_is_terminal()):
            self.log(
                "  [VISUAL-REPLAN] Skipped — terminal command subtask "
                "(clicks/re-typing can't fix command errors)"
            )
            _visual_mode = False
        try:
            if _visual_mode:
                self.log("  [VISUAL-REPLAN] Text planning stuck — asking VLM with screenshot")
                try:
                    step = self._plan_visual(subtask, run.completed)
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
                    return "done", None
            if not _visual_mode:
                if run.step_queue:
                    step = run.step_queue.pop(0)
                    step.id = len(run.completed) + 1
                    self.log(
                        f"  [PLAN-QUEUE] Executing queued step "
                        f"({len(run.step_queue)} remaining after this)"
                    )
                else:
                    planned = self.planner.plan_steps(
                        subtask, run.screen_context, run.completed,
                        task_context=task_context, failure_hints=failure_hints,
                    )
                    if not planned:
                        # A "save as <path>" subtask has ONE ground truth:
                        # the file on disk. The planner's "goal achieved"
                        # must not overrule its absence — that reports a
                        # false success with no file produced.
                        if run.save_target and not self._file_saved_fresh(
                            run.save_target, run.started_at
                        ):
                            self.log(
                                "  Planner declared done but "
                                f"'{run.save_target}' is not on disk — rejecting"
                            )
                            run.completed.append(
                                "[FAILED: planner declared done but the "
                                "target file is not on disk]"
                            )
                            return self._note_planning_failure(run), None
                        # A planner returning [] before ANY action executed in
                        # this subtask claims the work was already done by
                        # someone else — demand screen evidence (live: 'set
                        # the date to 07/08/2026' was declared achieved while
                        # Start date read '7/6/2026'). Kinds normally excluded
                        # from the goal check get a forced check here; for the
                        # rest, this cycle's pre-plan check already said the
                        # goal is NOT on screen.
                        acted = any(
                            not c.startswith("[FAILED") for c in run.completed
                        )
                        if not acted:
                            _excluded = (
                                run.is_launch_goal or run.is_cmd_subtask
                                or run.save_target or run.type_payload
                            )
                            confirmed = _excluded and self._goal_already_satisfied(
                                run, subtask, skip_exclusions=True
                            )
                            if not confirmed:
                                self.log(
                                    "  Planner declared done with no action "
                                    "taken and no screen evidence — rejecting"
                                )
                                run.completed.append(
                                    "[FAILED: planner declared done, but the "
                                    "goal state is not on screen and no "
                                    "action was executed]"
                                )
                                return self._note_planning_failure(run), None
                        return "done", None   # planner says goal is achieved
                    step, run.step_queue = planned[0], list(planned[1:])
        except PlanningParseError as e:
            # Unparseable planner output is a planning FAILURE, not goal
            # achievement. Record it and let the loop try again (or abort
            # via the consecutive-failures limit).
            self.log(f"  Planning failed: {e}")
            run.completed.append("[FAILED: planner produced unparseable output]")
            return self._note_planning_failure(run), None
        return "step", step

    def _run_step_attempts(
        self, run: "_SubtaskRun", subtask: SubTask, step: ActionStep,
    ) -> str:
        """Execute one planned step under the retry + verification policy.

        Returns:
          "subtask_done" — a deterministic check proved the WHOLE subtask done
          "step_ok"      — the step succeeded (verified, or safely assumed)
          "step_failed"  — every attempt failed, or verification said stop
        """
        # Terminal-command subtasks verify the typed command DETERMINISTICALLY
        # at the Enter press (file-on-disk / shell-error check in
        # _verify_command_effect). OCR-reading a dark console to confirm a type
        # step systematically misfires ("FAILED" on correct typing) and used to
        # trigger a destructive re-type. Defer to the authoritative Enter check.
        _cmd_type_step = run.is_cmd_subtask and step.action_type == "type"
        # set_value verifies itself by reading the control back through the
        # accessibility tree — LLM reflection would only add latency.
        skip_reflection = (
            step.action_type in ("wait", "extract", "set_value") or _cmd_type_step
        )
        if _cmd_type_step:
            self.log(
                "  [CMD-TYPE] Command typed — deferring verification to the "
                "Enter/disk check (skipping unreliable OCR reflection)"
            )
        run.last_error = ""

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
        non_idempotent = (
            step.action_type == "type"
            # invoke presses a real button — invoking twice double-submits.
            or step.action_type == "invoke"
            or (step.action_type == "key_press" and _key_l in ("enter", "return", "space"))
            or (step.action_type == "hotkey" and "v" in _key_l.split("+") and "ctrl" in _key_l)
        )

        for attempt in range(self.config.max_retries_per_step):
            # The kill switch must halt attempts IMMEDIATELY: one verify round
            # here costs 20-40 s of model calls, and a live run kept clicking
            # for 38 s after EMERGENCY STOP because only the outer step loop
            # checked the event.
            if self._stop_event.is_set():
                return "step_failed"
            if attempt > 0:
                self.log(f"  Retry {attempt}/{self.config.max_retries_per_step}…")

            # Capture pre-action hash for click steps BEFORE the action fires.
            # The reflection agent's internal "before" capture runs AFTER execute(),
            # so instant UI changes (menus opening) make before==after → false delta=0.
            pre_click_hash = None
            if step.action_type in ("click", "right_click", "double_click"):
                try:
                    from core.capture.screenshot import frame_phash
                    pre_click_hash = frame_phash(self.capturer.capture())
                except Exception:
                    pass

            if not self._execute_step(step):
                run.last_error = (
                    getattr(self, "_exec_fail_reason", "")
                    or "execution failed (element not found or action error)"
                )
                # Tree actions are deterministic: the same pattern call
                # against the same tree gives the same result — blind
                # retries only burn seconds (live: the same 'no option'
                # miss re-ran 3× per cycle for minutes).
                if step.action_type in ("set_value", "select", "invoke"):
                    if step.action_type == "select":
                        # Put the mismatch in the failure record the planner
                        # reads: what the plan asked for vs. the labels the
                        # app really shows ('GST' vs '(UTC+04:00) Abu
                        # Dhabi, Muscat') — so it picks a real label or
                        # scrolls the open list for more options.
                        from core import windows_uia
                        miss = windows_uia.pop_select_miss()
                        if miss is not None:
                            shown = ", ".join(miss["items"]) or (
                                "none rendered yet — items appear as the "
                                "open list is scrolled"
                            )
                            run.last_error = (
                                f"no option named '{miss['option']}' exists "
                                f"in '{miss['target']}'; the open list shows: "
                                f"{shown}. Use the app's exact option label, "
                                f"or scroll the open list to reveal more"
                            )
                    break
                continue

            if skip_reflection:
                return "step_ok"

            # Deterministic type verification: read the focused control's
            # value from the accessibility tree. Exact, ~50 ms, and skips
            # the ~3-4 s OCR+LLM reflection call. Falls through to normal
            # reflection when the control exposes no readable value.
            if step.action_type == "type":
                time.sleep(0.3)   # let the control commit the input
                if self._typed_text_in_focused_control(step.value, step.target):
                    self.log(
                        "  [TYPE-VERIFY] Focused control contains the typed "
                        "text — verified via accessibility tree"
                    )
                    return "step_ok"

            # Deterministic verification for the execute-Enter of a
            # "run: <command>" subtask — checks the filesystem / terminal
            # output instead of LLM reflection. On confirmed effect the
            # whole subtask is done (command typed + executed + verified).
            if (
                run.is_cmd_subtask
                and step.action_type == "key_press"
                and (step.key or "").lower() == "enter"
            ):
                _ok, _why = self._verify_command_effect(
                    subtask, run.started_at, run.typed_ok
                )
                if _ok:
                    self.log(f"  [CMD-CHECK] {_why} — command effect confirmed")
                    return "subtask_done"
                run.last_error = _why
                self.log(f"  [CMD-CHECK] {_why}")
                # Never blind-retry Enter; the planner corrects the command.
                return "step_failed"

            verdict = self._judge_reflection(run, step, non_idempotent, pre_click_hash)
            if verdict == "success":
                return "step_ok"
            if verdict == "stop":
                return "step_failed"
            # "retry" → next attempt

        return "step_failed"

    def _judge_reflection(
        self, run: "_SubtaskRun", step: ActionStep,
        non_idempotent: bool, pre_click_hash,
    ) -> str:
        """One verification round for a step that has physically executed.

        Returns "success", "retry" (safe to run the step again), or "stop"
        (definitive failure — hand control back to the planner).
        """
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
            # Controls snapshot from the planning cycle that produced this
            # step: appeared/disappeared control names give the verifier
            # tree-level proof of effect on WebView2 screens OCR can't read.
            _controls_before = (
                getattr(self, "_last_controls", None)
                if step.action_type in
                ("click", "right_click", "double_click", "invoke")
                else None
            )
            reflection = self.reflector.verify(step, _reflect_wait,
                                               pre_hash=pre_click_hash,
                                               controls_before=_controls_before)
            run.cached_ocr = reflection.ocr_text   # reuse for next planning call

            # Ground truth override: a click that flipped the foreground to a
            # DIFFERENT app than the task's anchor hit the wrong element, no
            # matter what any model verdict says (seen live: an Outlook button
            # literally named 'New' opened an Edge error page — the VLM even
            # blessed it). Blacklist the point on the screen it was grounded
            # on, bring the task app back, and fail the step honestly.
            if (
                step.action_type in ("click", "right_click", "double_click")
                and not run.is_launch_goal
                and getattr(self, "_app_anchor", None)
            ):
                _fg_hwnd, _fg_pid, _fg_name = self._foreground_app()
                if _fg_pid and _fg_pid != self._app_anchor[1] \
                        and _fg_hwnd != getattr(self, "_own_hwnd", None):
                    self.log(
                        f"  [ANCHOR] Click switched foreground to {_fg_name} — "
                        f"wrong element for a task anchored to {self._app_anchor[2]}"
                    )
                    logger.warning(
                        f"[ORCHESTRATOR] Click on '{step.target}' opened {_fg_name} "
                        f"(task app: {self._app_anchor[2]}) — marking point dead"
                    )
                    if getattr(self, "_last_click_xy", None) and step.target:
                        self.grounder.mark_dead(step.target, *self._last_click_xy)
                    self._ensure_anchor_foreground()
                    run.last_error = (
                        f"click opened {_fg_name} instead of acting inside "
                        f"{self._app_anchor[2]} — wrong element"
                    )
                    return "stop"

            if reflection.success:
                self.log(f"  Verified (conf={reflection.confidence:.2f})")
                time.sleep(0.1)
                return "success"

            # "Screen unchanged" after a click is a false failure when
            # the click's whole effect was placing the caret — verify
            # against the accessibility tree instead of retrying blind.
            if (
                step.action_type == "click"
                and "unchanged" in (reflection.error_description or "").lower()
                and self._click_holds_focus(step)
            ):
                self.log(
                    "  [FOCUS-CHECK] Clicked point owns keyboard focus "
                    "(text control) — click confirmed"
                )
                return "success"

            # A save hotkey that opened the Save-As dialog IS a success,
            # but the LLM verifier reads it as "no confirmation dialog"
            # and fails it — which retries ctrl+s into the open dialog
            # (a no-op) until the budget is gone. Check deterministically.
            if (
                step.action_type == "hotkey"
                and (step.key or "").lower() in ("ctrl+s", "ctrl+shift+s")
                and self._save_dialog_visible()
            ):
                self.log("  [SAVE-CHECK] Save-As dialog is open — hotkey confirmed")
                return "success"

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
                if non_idempotent:
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
                    # The "next step verifies live" guarantee requires a
                    # fresh plan against the live screen — drop any steps
                    # planned before this uncertain outcome.
                    run.step_queue = []
                    return "success"
                # Idempotent action (click / scroll): safe to repeat, and a
                # dead click is caught RELIABLY by the screen-delta (phash)
                # check as a high-confidence failure — so retrying here costs
                # nothing and recovers genuinely-missed clicks.
                run.last_error = (
                    f"uncertain outcome (conf={reflection.confidence:.2f}, "
                    f"threshold={fail_threshold:.2f})"
                )
                self.log(f"  Uncertain result - retrying ({run.last_error})")
                # Fast-path: if the goal process is already running after an
                # uncertain click, skip the remaining retries.
                if run.is_launch_goal and (
                    step.action_type in ("click", "double_click")
                ):
                    if self._goal_confirmed(run.goal_proc, run.baseline_windows):
                        self.log(
                            f"  [GOAL-CHECK-EARLY] "
                            f"'{run.goal_proc.split('.')[0]}' confirmed "
                            f"— accepting step"
                        )
                        return "success"
                return "retry"

            # Reflector is confident the step failed.
            run.last_error = (
                reflection.error_description
                or "action did not produce expected result"
            )
            self.log(
                f"  Verification failed: {run.last_error} "
                f"(conf={reflection.confidence:.2f})"
            )
            # Ground truth: a click that changed ZERO pixels (phash delta=0)
            # hit an inert point. Blacklist it so the retry's re-grounding
            # cannot serve the same coordinate from cache/UIA/OCR and must
            # fall through to another stage (or fail honestly). Without this,
            # "Re-ground the target" retries are no-ops: unchanged screen →
            # same phash → cache hit → same dead pixel, forever.
            # NEVER after a pattern invoke: the invoke consumed the attempt
            # without touching that pixel, so the coordinate is unproven —
            # dead-marking it here starved the retry's pixel click of the
            # REAL button and sent it to a VLM guess instead (seen live:
            # 'Send' invoked → delta=0 → true point blacklisted → the one
            # click that would have worked landed on the title bar).
            if (
                step.action_type in ("click", "right_click", "double_click")
                and "unchanged" in run.last_error.lower()
                and getattr(self, "_last_click_xy", None)
                and not getattr(self, "_last_was_invoke", False)
            ):
                self.grounder.mark_dead(step.target, *self._last_click_xy)
            # The pattern-invoke equivalent of a dead point: the provider
            # accepted Invoke/Toggle but nothing happened. Coordinates are
            # not involved, so blacklist the invoke PATH for this target —
            # the retry then re-executes as a real pixel click.
            if (
                step.action_type in ("click", "right_click", "double_click")
                and getattr(self, "_last_was_invoke", False)
                and step.target
            ):
                self._invoke_dead.add(step.target.lower())
                self.log(
                    f"  [INVOKE-DEAD] Pattern invoke on '{step.target}' had no "
                    f"verified effect — retries will use a pixel click"
                )
            # When the proven-inert point turns out to be COVERED, the step
            # isn't failing because the target is wrong — something is on
            # top of it. Name the blocker in the failure record the planner
            # reads, so the next plan dismisses the overlay.
            if (
                step.action_type in ("click", "right_click", "double_click")
                and "unchanged" in run.last_error.lower()
                and getattr(self, "_last_click_xy", None)
                and step.target
            ):
                from core import windows_uia
                _cover = windows_uia.covering_element(
                    *self._last_click_xy, step.target
                )
                if _cover:
                    run.last_error += (
                        f" — the point belongs to '{_cover}', not "
                        f"'{step.target}': an overlay/dialog is on top; "
                        f"dismiss it before retrying"
                    )
                    self._blocking_overlay = (step.target or "", _cover)
                    self.log(
                        f"  [OCCLUSION] '{step.target}' is covered by "
                        f"'{_cover}'"
                    )
            # A confidently-failed non-idempotent action must also not
            # be blind-retried (Fix C5) — same double-execution risk.
            if not reflection.should_retry or non_idempotent:
                # Definitive failure (or unsafe to retry) — stop here.
                return "stop"
            return "retry"
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
            run.last_error = f"verifier parse error ({type(e).__name__})"
            return "retry"
        except Exception as e:
            # Infrastructure failure — retry with back-off
            self.log(f"  Reflection error ({type(e).__name__}: {e}) — retry")
            run.last_error = str(e)
            time.sleep(1.0)
            return "retry"

    def _record_step_success(
        self, run: "_SubtaskRun", subtask: SubTask, step: ActionStep,
    ) -> bool | None:
        """Bookkeeping + early-exit checks after a successful step.

        Returns True/False to finish the whole subtask, or None to keep looping.
        """
        if step.action_type == "type":
            run.typed_ok = True
        # Append extract result to step description for context
        if step.action_type == "extract":
            key = step.target or step.description or f"item_{step.id}"
            val = self._extracted_data.get(key, "")
            run.completed.append(f"{step.description} → extracted: '{val}'")
        else:
            run.completed.append(step.description)
        run.consecutive_failures = 0

        # Early-exit: a "... type: <text>" subtask is DONE once that text
        # has been typed and verified — never give the planner a chance
        # to re-type it or drift into save steps of a later subtask.
        if (
            run.type_payload
            and step.action_type == "type"
            and self._texts_equivalent(step.value or "", run.type_payload)
        ):
            self.log(
                "  [TYPE-CHECK] Requested text typed and verified — "
                "subtask complete"
            )
            return True

        # Early-exit: after each successful step in a known app-launch subtask,
        # check the process list on Windows. If the app is already running, the
        # goal is achieved — return True immediately so the planner never gets a
        # chance to generate spurious continuation steps (e.g. pressing Enter
        # again inside a Calculator that is already open).
        if run.is_launch_goal and self._goal_confirmed(run.goal_proc, run.baseline_windows):
            _label = run.goal_proc.split(".")[0]
            self.log(f"  [GOAL-CHECK] '{_label}' confirmed — goal achieved")
            return True

        # Normalize the signature so trivial respellings of the same target
        # ("New Tab" vs "Newtab") cannot dodge the loop guard — the planner
        # varies casing/spacing between cycles when it is stuck.
        def _norm(s):
            return re.sub(r"[^a-z0-9]", "", (s or "").lower())
        sig = (step.action_type, _norm(step.target), _norm(step.value),
               _norm(step.key))
        count = run.step_sig_counts.get(sig, 0) + 1
        run.step_sig_counts[sig] = count
        _dedup_limit = DEDUP_LIMIT_BY_ACTION_TYPE.get(step.action_type, 2)
        # limit = allowed REPEATS beyond the first success (same semantics as
        # the old consecutive streak: total appearances ≤ limit+1 are fine).
        if count <= _dedup_limit + 1:
            return None
        self.log(
            f"  [LOOP-GUARD] '{step.action_type}' on '{step.target}' "
            f"succeeded {count}× in this subtask (limit={_dedup_limit}) — "
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
            c.startswith("[FAILED") for c in run.completed
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

    def _record_step_failure(
        self, run: "_SubtaskRun", step: ActionStep,
    ) -> bool | None:
        """Bookkeeping after a failed step; abort once failures exhaust the budget.

        Returns True/False to finish the whole subtask, or None to keep looping.
        """
        note = (
            f"[FAILED: {run.last_error}] {step.description}"
            if run.last_error else f"[FAILED] {step.description}"
        )
        run.completed.append(note)
        # The rest of the queued plan was built on this step succeeding —
        # flush it so the next iteration re-plans from the live screen.
        run.step_queue = []

        # Even on step failure, check if the goal process is now running.
        # This handles the common case where key_press enter launches an app
        # but the reflector times out before the app's UI is fully visible
        # (reflection confidence 0.50 → step marked failed, yet app IS open).
        if run.is_launch_goal and self._goal_confirmed(run.goal_proc, run.baseline_windows):
            _label = run.goal_proc.split(".")[0]
            self.log(
                f"  [GOAL-CHECK] '{_label}' confirmed "
                f"despite step failure — goal achieved"
            )
            return True

        run.consecutive_failures += 1
        self.log("  Step failed - re-evaluating next action")

        # Persist failure so future tasks can avoid this pattern
        try:
            self.memory.store_failure_pattern(
                target=step.target or step.description or "",
                action_type=step.action_type,
                error=run.last_error[:200] if run.last_error else "",
                app_context=run.screen_context[:100],
            )
        except Exception:
            pass

        if run.consecutive_failures >= self.config.consecutive_failures_limit:
            self.log(
                f"  {self.config.consecutive_failures_limit} consecutive failures"
                f" — aborting subtask"
            )
            return False
        return None

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

        Visual-planner click steps carry direct screen coordinates this way,
        bypassing grounding entirely.
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
        # Stop pressed: never BEGIN a new action. The step loop already checks
        # between steps, but grounding one target can block 30-60 s on a VLM
        # call — without a check here (and before each dispatch below) the
        # agent finished grounding and still fired the click after the user
        # hit Stop, which is exactly why the button felt dead.
        if self._stop_event.is_set():
            return False
        x, y, x2, y2 = None, None, None, None
        # Where the last click-family step actually landed — lets the
        # reflector's "screen unchanged" verdict blacklist the exact point.
        self._last_click_xy = None
        # Specific failure reason for the caller's [FAILED: …] record; the
        # planner reads that record, so name the real blocker when we know it.
        self._exec_fail_reason = ""

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
            self._last_was_invoke = False
            _coords = self._explicit_coords(step.value)
            if _coords is not None:
                # Explicit pixel coordinates (visual-planner convention)
                x, y = _coords
                self._last_click_xy = (x, y)
                if self._stop_event.is_set():
                    return False
                return self.actor.execute(step, x=x, y=y)
            if not step.target:
                self.log(f"  {step.action_type} has no target")
                return False
            result = self.grounder.ground(step.target)
            if not result.found or result.confidence < self.grounder.min_confidence:
                # A cached miss means the FULL cascade — including the scroll
                # hunt that follows it — already failed on this exact screen.
                if result.method == "cached_miss":
                    self.log(f"  '{step.target}' known-absent on this screen")
                    return False
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
            self._last_click_xy = (x, y)
            # Hit-test before acting: ElementFromPoint is OS ground truth for
            # "what would a click here actually land on". When a dialog or
            # overlay sits on top (live: Teams' 'Meeting created' popup over
            # the form), every click AND pattern invoke beneath it is a
            # provable no-op — fail fast and hand the planner the blocker's
            # name so it dismisses the overlay instead of grinding retries.
            if "uia" in (result.method or ""):
                from core import windows_uia
                _cover = windows_uia.covering_element(x, y, step.target)
                if _cover:
                    self._exec_fail_reason = (
                        f"'{step.target}' is covered by '{_cover}' — an "
                        f"overlay/dialog is on top; dismiss it (its Close/"
                        f"Cancel/OK button) before retrying"
                    )
                    # Also surface it in the screen context: the failure
                    # record alone was ignored for six planning cycles while
                    # 'Close' sat in the controls list (live 15:14-15:24).
                    self._blocking_overlay = (step.target or "", _cover)
                    self.log(
                        f"  [OCCLUSION] '{step.target}' at ({x},{y}) is "
                        f"covered by '{_cover}' — click would not reach it"
                    )
                    return False
            # Ground truth first: when Stage 0 found this element in the UIA
            # tree, activate the control OBJECT via its Invoke/Toggle pattern
            # instead of synthesizing a mouse click at its pixels. A pattern
            # invoke cannot hit the wrong pixel and cannot be swallowed by
            # overlays or WebView input handling (seen live: every click on
            # Outlook's 'New' button landed with delta=0 while the control
            # was perfectly invokable). Reflection still verifies the visible
            # outcome afterwards. Falls back to the pixel click whenever the
            # control exposes no pattern or the re-lookup misses.
            # ... unless invoking THIS target already failed verification once:
            # WebView2 providers accept Invoke/Toggle without doing anything
            # (live: 'Invite to Teams' invoked 6× with zero screen change —
            # coordinate blacklists never fire because no pixel is involved).
            # One failed verify per target sends retries down the pixel path.
            if (
                step.action_type == "click"
                and "uia" in (result.method or "")
                and (step.target or "").lower() not in self._invoke_dead
            ):
                from core import windows_uia
                if self._stop_event.is_set():
                    return False
                if windows_uia.invoke_element(step.target, timeout_s=2.5):
                    self._last_was_invoke = True
                    return True

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

        # ── Type / set_value — run through the destructive-action firewall ─────
        elif step.action_type in ("type", "set_value"):
            if not self._firewall_allows(step.value):
                self.log(
                    "  [FIREWALL] Blocked typing a destructive command — "
                    "step aborted for safety"
                )
                return False

        # Final stop gate: grounding above may have blocked for seconds on a
        # model call during which the user hit Stop — do not fire the action.
        if self._stop_event.is_set():
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
            img.thumbnail(OCR_THUMB)
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

    @staticmethod
    def _subtask_type_payload(subtask) -> str | None:
        """Extract the literal text a "... type: <text>" subtask asks to type.

        Returns the payload when the subtask ends with the text to type (the
        router phrases typing subtasks as "click in the document area and
        type: <text>"), else None. Lets the orchestrator complete the subtask
        deterministically the moment that text has been typed and verified —
        otherwise the planner re-types the text or drifts into save steps that
        belong to a later subtask.
        """
        desc = subtask.description or ""
        m = re.search(r"\btype:?\s+['\"]?(.+?)['\"]?\s*$", desc, re.IGNORECASE)
        if not m:
            return None
        payload = m.group(1).strip()
        # Too-short payloads ("type notepad") are launcher/search idioms, not a
        # document-typing goal — don't short-circuit those.
        return payload if len(payload) >= 6 else None

    @staticmethod
    def _texts_equivalent(a: str, b: str) -> bool:
        """True when two strings are the same text modulo case/whitespace/quotes.

        Fuzzy (difflib) because the planner re-emits the payload from the
        subtask description and may vary punctuation slightly. Containment
        counts only when lengths are comparable, so typing a fragment of the
        payload never passes as the full text.
        """
        import difflib

        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").strip().strip("'\"")).lower()

        na, nb = _norm(a), _norm(b)
        if not na or not nb:
            return False
        if (na in nb or nb in na) and min(len(na), len(nb)) / max(len(na), len(nb)) >= 0.8:
            return True
        return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85

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

    def _typed_text_in_focused_control(
        self, text: str | None, target: str | None = None,
    ) -> bool:
        """Deterministic type verification: the focused control's accessible
        value contains the text that was just typed — and, when the step
        names a destination field, the focused control IS that field.

        Content alone is not enough: an attendee email typed into the Title
        field still "contains the typed text" (seen live). Reads the real
        widget from the accessibility tree — exact and ~50 ms, versus the
        ~3-4 s OCR+LLM reflection call it replaces. Returns False (fall back
        to reflection) when the text is trivial, contains a credential
        placeholder, the control exposes no readable value, or the focused
        control's name does not match the intended field.
        """
        if not text or len(text.strip()) < 2 or "{{cred:" in text:
            return False
        try:
            from core import windows_uia
            info = windows_uia.focused_element_info()
            value = (info or {}).get("value") or ""
            if not value:
                return False

            def _norm(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip().lower()

            if target:
                name = _norm((info or {}).get("name") or "")
                want = _norm(target)
                if not name or (want not in name and name not in want):
                    self.log(
                        f"  [TYPE-VERIFY] Focus is on '{name or '?'}', not "
                        f"'{target}' — text landed in the wrong field"
                    )
                    return False
            return _norm(text) in _norm(value)
        except Exception:
            return False

    def _click_holds_focus(self, step: ActionStep) -> bool:
        """True when the clicked point lies inside the control that owns
        keyboard focus and that control accepts text input.

        A click whose effect is placing the caret produces zero pixel change
        (the caret is invisible to the phash diff), so reflection scores it as
        a no-op and retries it forever. The accessibility tree is the ground
        truth: if the focused control contains the clicked coordinates and is
        a text control, the click achieved everything a click can achieve.
        Works for any app and any task — no target-name matching involved.

        The clicked PIXEL itself must also be a text control. WebView apps
        (new Outlook, Teams) expose one focused DocumentControl whose rect
        spans nearly the whole window — "focused rect contains the point"
        alone rescued dead clicks on buttons for 10+ minutes (seen live),
        which also kept the dead-point blacklist from ever firing.
        """
        try:
            from core import windows_uia
            xy = getattr(self, "_last_click_xy", None)
            if not xy:
                return False
            at_point = windows_uia.control_type_at_point(*xy)
            if at_point not in ("EditControl", "DocumentControl"):
                return False
            info = windows_uia.focused_element_info()
            if not info or info["control_type"] not in (
                "EditControl", "DocumentControl",
            ):
                return False
            left, top, right, bottom = info["rect"]
            return left <= xy[0] <= right and top <= xy[1] <= bottom
        except Exception:
            return False

    def _save_dialog_visible(self) -> bool:
        """True when a Windows Save/Save-As dialog is on screen.

        Detected by its stable static labels (immune to the dynamic content OCR
        struggles with). Used to confirm ctrl+s opened a dialog BEFORE typing the
        path — so we never type a path into the document body of an already-named
        file that ctrl+s saved silently.
        """
        # Accessibility tree first — exact control names straight from the
        # dialog, immune to OCR misreading small labels on a downscaled
        # screenshot (which made this check miss an open dialog entirely).
        try:
            from core import windows_uia
            verdict = windows_uia.save_dialog_open()
            if verdict is not None:
                return verdict
        except Exception:
            pass
        # OCR fallback — UIA unavailable or timed out.
        try:
            img = self.capturer.capture()
            img.thumbnail(OCR_THUMB)
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

        # 3. Replace the default filename with the full target path, then READ
        #    BACK the field through the accessibility tree to confirm the paste
        #    actually landed before pressing Enter (a too-early Enter saves
        #    under the default name and the dialog closes silently).
        for _attempt in range(2):
            self._execute_step(_step("hotkey", key="ctrl+a", desc="Select filename field"))
            if not self._execute_step(_step("type", value=typed_path, desc="Type save path")):
                return False
            time.sleep(0.3)
            _field_val = None
            try:
                from core import windows_uia
                _field_val = (windows_uia.focused_element_info() or {}).get("value")
            except Exception:
                pass
            if not _field_val:
                break   # field value not readable — proceed optimistically
            if typed_path.lower() in re.sub(r"\s+", " ", _field_val).strip().lower():
                break   # read-back confirms the path landed in the field
            self.log("  [SAVE-AS] Filename field read-back mismatch — retyping")
        else:
            # Wrong content twice — close the dialog so the planner starts from
            # a clean editor, not a half-filled dialog.
            self._execute_step(_step("key_press", key="escape", desc="Close dialog"))
            self.log("  [SAVE-AS] Could not confirm filename field — deferring to planning loop")
            return False

        # 4. Confirm. Poll the disk. A "Replace existing file?" prompt appears
        #    when the target already exists; its DEFAULT button is "No", so a
        #    blind Enter cancels the save. Alt+Y presses the accelerated "Yes"
        #    deterministically — sent only while the file is still absent, so a
        #    completed save never receives stray input.
        self._execute_step(_step("key_press", key="enter", desc="Confirm save"))
        for i in range(10):
            time.sleep(0.4)
            if self._file_saved_fresh(target, started_at):
                self.log(f"  [SAVE-AS] '{target}' written to disk — confirmed")
                return True
            if i == 2:
                self._execute_step(
                    _step("hotkey", key="alt+y", desc="Confirm overwrite (Yes)"))
            elif i == 5:
                self._execute_step(
                    _step("key_press", key="enter", desc="Confirm overwrite (fallback)"))
        # Leave a CLEAN state for the planning loop: whatever dialog/prompt is
        # still up would swallow the planner's keystrokes.
        self._execute_step(_step("key_press", key="escape", desc="Close stale dialog"))
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
            img.thumbnail(OCR_THUMB)
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
            if self._stop_event.is_set():
                break
            self.log(
                f"  [SCROLL-FIND] '{target}' not visible — scroll {i + 1}/"
                f"{self.config.max_scroll_find_attempts}"
            )
            self.actor.execute(scroll_step, x=cx, y=cy)
            scrolls_done += 1
            time.sleep(0.35)

            # Check if page content is still changing
            img = self.capturer.capture()
            img.thumbnail(OCR_THUMB)
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
        "zoom":             "Zoom.exe",
        "teams":            "ms-teams.exe",
        "slack":            "slack.exe",
        "skype":            "Skype.exe",
        "thunderbird":      "thunderbird.exe",
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
        "zoom":            ["Zoom", "New Meeting", "Schedule", "Join"],
        "teams":           ["Teams", "Chat", "Calendar", "Meet"],
        "slack":           ["Slack", "Channels", "Direct messages"],
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
                img.thumbnail(OCR_THUMB)
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
