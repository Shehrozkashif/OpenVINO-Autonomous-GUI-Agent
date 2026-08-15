# core/orchestrator.py
"""The See → Plan → Act → Verify loop.

    instruction
      → ask the user for details it omits (meeting time, recipient…)
      → Router.decompose() → SubTasks, in dependency order
      → for each SubTask, planning ONE step at a time against the live screen:
            is it already done?      → goal check (deterministic, else one LLM yes/no)
            what next?               → Planner (text path; screenshot path when stuck)
            where is it?             → Grounding: UIA tree → OCR → VLM
            do it                    → Actor, after the firewall vets typed text
            did it work?             → ground truth first, Reflection only as fallback
            it failed                → blacklists, refocus, replan with the goal pinned
      → Router.summarize_completion() (+ [BLOCKER] evidence when it failed)

This module owns the LOOP and its recovery policy. The facts it decides on come
from elsewhere, which is what keeps it readable:

    core/runstate.py    budgets, limits, per-subtask state
    core/groundtruth.py checks the OS can prove (file on disk, process, title)
    core/subtasks.py    what a subtask's own words say it wants
    core/apps.py        which executable an app name means
    core/anchor.py      which window this task owns, and staying inside it

Log lines here are a public interface: ui/events.py parses them into the live
mission timeline, so their format must not drift.
"""
import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

from loguru import logger

from agents.action import ActionExecutionAgent
from agents.grounding import GroundingResult, UIGroundingAgent
from agents.planning import PlanningAgent, PlanningParseError
from agents.reflection import ReflectionAgent
from agents.router import RouterAgent
from core import apps, groundtruth, subtasks
from core.anchor import AppAnchor
from core.groundtruth import GroundTruth
from core.history import TaskHistory
from core.runstate import DEDUP_LIMIT_BY_ACTION_TYPE, OrchestratorConfig, SubtaskRun
from core.types import ActionStep, SubTask
from desktop import system
from desktop.capture import OCR_THUMB, OwnWindowMask, ScreenCapture, _screen_size
from desktop.snapshot import capture_snapshot

if TYPE_CHECKING:
    from desktop.ocr import OCREngine


class TaskOrchestrator:
    def __init__(
        self,
        router: RouterAgent,
        planner: PlanningAgent,
        grounder: UIGroundingAgent,
        actor: ActionExecutionAgent,
        reflector: ReflectionAgent,
        capturer: ScreenCapture,
        history: TaskHistory,
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
        self.history   = history
        self.config    = config or OrchestratorConfig()
        # Orchestrator decisions (Step/[GOAL-CHECK]/[CLICK-CHECK]/[SUBTASK]/…)
        # must land in BOTH the GUI log panel AND loguru. The dev loop is
        # "run on the Windows PC, paste the terminal log" — and the terminal
        # only captures loguru, so a UI-only self.log left every routing
        # decision invisible in pasted logs (a fix's own marker couldn't be
        # confirmed). Mirror to loguru at the caller's call site, then fan out
        # to the UI callback when one is wired.
        _ui_log = on_step_log
        def _log(msg: str) -> None:
            logger.opt(depth=1).info(msg)
            if _ui_log is not None:
                _ui_log(msg)
        self.log = _log
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
            from desktop.ocr import OCREngine
            self._ocr = OCREngine()

        # Collaborators: the loop decides, these three know the world.
        self.mask   = OwnWindowMask(capturer, log=self.log)
        self.anchor = AppAnchor(log=self.log, own_hwnd=lambda: self.mask.hwnd)
        self.truth  = GroundTruth(capturer, self._ocr, log=self.log)

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
            from desktop.input import KillSwitch
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


    # ── Public entry point ────────────────────────────────────────────────────

    def execute(self, instruction: str) -> dict:
        self._stop_event.clear()
        self._extracted_data = {}
        self._launch_window_baseline = {}
        self._degraded = False
        self.anchor.clear()         # forget the previous task's app window
        self._invoke_dead = set()   # targets whose pattern-invoke provably no-ops
        self._last_goal_evidence = ""
        self._last_controls = None
        self._blocking_overlay = None
        # Dead/foreign-point blacklists describe the PREVIOUS task's screens —
        # they must not veto valid coordinates in this one.
        try:
            self.grounder.clear_dead_points()
        except Exception:
            pass
        self.log(f"[TASK START] '{instruction}'")
        start_time = time.time()
        self._arm_kill_switch()

        # Ask the user for required details the instruction omits (a meeting
        # time, a recipient) BEFORE any planning — guessing them wastes the
        # whole run. No-op when no question handler is wired.
        # What they typed is kept: the enriched string carries this run's answers
        # ("...(details provided by the user: time zone -> PST)"), and history is
        # keyed on the instruction text, so storing it would file every run under
        # its own answers instead of merging them onto one reusable task.
        user_instruction = instruction
        instruction = self._elicit_missing_parameters(instruction)

        screen_context = self._get_screen_context()

        task_id, plan = self.router.decompose(
            instruction, screen_context=screen_context
        )
        self.log(f"[ROUTER] {len(plan)} sub-task(s)")

        queue: list[SubTask] = subtasks.topological_order(plan)
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
                self.anchor.adopt_foreground(subtask.description)
                if queue:
                    self.truth.wait_for_settle(min_s=0.5, max_s=3.0)
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
            self.history.store_successful_task(
                user_instruction, executed_subtasks or plan, elapsed
            )
        elif not failed and self._degraded:
            self.log("[MEMORY] Task completed via a degraded path — not recorded as a clean success")

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
            success = self.truth.verify_launch(subtask, self._launch_window_baseline)
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
                    success = self.truth.verify_launch(subtask, self._launch_window_baseline)
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
        return subtasks.topological_order(renumbered + kept)

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
        is_launch_goal = apps.is_launch_description(desc)

        goal_proc: str | None = None
        baseline_windows: int | None = None
        if is_launch_goal:
            goal_proc = apps.process_for(desc)
            if goal_proc and system.is_process_running(goal_proc):
                baseline_windows = system.count_process_windows(goal_proc)
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
        if self.anchor.launch_already_satisfied(subtask):
            return True

        _is_launch_goal, _goal_proc, _baseline_windows, task_context = (
            self._setup_launch_goal(subtask, task_context)
        )
        run = SubtaskRun(
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
            save_target=subtasks.save_target(subtask),
        )
        # "... type: <text>" subtasks are complete the moment that text has been
        # typed and verified. Without this, the planner re-types the payload
        # (duplicating it in the document) or drifts into ctrl+s save steps that
        # belong to a later subtask — burning the whole subtask budget.
        run.type_payload = None if run.save_target else subtasks.type_payload(subtask)

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
            if run.save_target and groundtruth.file_saved_fresh(run.save_target, run.started_at):
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

    def _note_planning_failure(self, run: "SubtaskRun") -> str:
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
            from desktop.uia import get_interactive_elements
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

    def _first_action_targets_live_control(
        self, run: "SubtaskRun", subtask: SubTask,
    ) -> bool:
        """True when this subtask has done NOTHING yet but the control it must
        actuate is present and enabled in the live tree.

        Grounded in UIA truth (control still there = click not yet made) and
        bounded to the first cycle, so it forces exactly one real action for an
        actuation subtask and can never loop (after the action run.completed is
        non-empty). This is what stops a pre-action goal-check from silently
        skipping a Save/Submit/Create the model wrongly judged 'already done'.
        """
        # Only the first cycle: any successful (non-[FAILED) step means the
        # subtask has already acted, and the normal goal-check governs.
        if any(not c.startswith("[FAILED") for c in run.completed):
            return False
        desc = subtask.description.lower()
        if not any(v in desc for v in
                   ("click", "press", "save", "submit", "create", "invoke", "select")):
            return False
        # Match the subtask against the exact, enabled controls the planner sees
        # ("'Save' [Button]"; disabled ones are tagged "(disabled)").
        for m in re.finditer(r"'([^']+)' \[([^\]]*)\]", run.screen_context or ""):
            name, meta = m.group(1), m.group(2)
            if "disabled" in meta.lower():
                continue
            nl = name.lower()
            # Whole-word/phrase match only: substring matching let 'Meet'
            # fire inside "meeting", forcing an action on the wrong control.
            if len(nl) >= 3 and re.search(
                r"(?<![a-z0-9])" + re.escape(nl) + r"(?![a-z0-9])", desc
            ):
                return True
        return False

    def _goal_already_satisfied(
        self, run: "SubtaskRun", subtask: SubTask,
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
                        "on screen). CRUCIAL: a control that would PERFORM the "
                        "goal — a button/tab/menu item whose label matches the "
                        "action (a 'Calendar' button for 'open the calendar', "
                        "a 'New meeting'/'Save' button for 'create/save') — is "
                        "proof the action has NOT happened yet, so "
                        "satisfied=false. Only the RESULT of the action being "
                        "visible (the calendar grid, the opened form's input "
                        "fields, a confirmation message) counts as satisfied. "
                        "IMPORTANT EXCEPTION: many apps give the OPENED form or "
                        "dialog the SAME name as the button that opened it (a "
                        "'New meeting' button opens a form titled 'New meeting'). "
                        "If that label now appears as a form/dialog TITLE or "
                        "HEADING — alongside input fields, an editor, or dialog "
                        "chrome (Save/Close/Cancel) rather than as a lone "
                        "toolbar/sidebar button — the form has OPENED and "
                        "satisfied=true. Do not read the opened form's own title "
                        "as the still-unclicked button. "
                        "In evidence, go REQUIREMENT BY "
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
        self, run: "SubtaskRun", subtask: SubTask, task_context: list[str],
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
            self.anchor.ensure_foreground()
        if run.step_queue:
            # A queued step is executed WITHOUT a planning call — the plan it
            # came from was written against this same screen moments ago, which
            # is also why the goal check below is skipped while a queue remains.
            # OCR text feeds the planner prompt and nothing else, so on this
            # path it was 1-2 s of ONNX per step building a prompt nobody sends.
            # The accessibility control list appended further down is NOT
            # skipped: it costs ~100 ms and the commit guard reads live field
            # values out of it before it will let a Save fire.
            run.screen_context = ""
            # Drop rather than keep: by the time a planning call happens again
            # the queue will have moved the screen on, so this text would be
            # several actions out of date.
            run.cached_ocr = ""
        else:
            run.screen_context = run.cached_ocr if run.cached_ocr else self._get_screen_context()
            run.cached_ocr = ""   # consume — will be refreshed after reflection

        # The foreground WINDOW TITLE, read via Win32 (survives OCR/UIA being
        # disabled). The planner reasons about the title for Teams/Office view
        # switches ('Calendar | Microsoft Teams') but was never handed it — so
        # it fired ctrl+4 blind and could not tell Chat from Calendar. Give it
        # the ground truth; the goal check reads the same context.
        try:
            from desktop.snapshot import _get_foreground_hwnd_and_title
            _, _fg_title = _get_foreground_hwnd_and_title()
            if _fg_title and _fg_title not in ("Unknown", "Desktop"):
                run.screen_context += (
                    f"\nFOREGROUND WINDOW TITLE (ground truth): {_fg_title}"
                )
        except Exception:
            pass

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

        # "Is this goal ALREADY met?" — asked before planning another action,
        # because "click X to open Y" subtasks loop once Y is open: every extra
        # click is a no-op that state-describing verifiers keep blessing ("the
        # form is visible → success"). Kinds with a deterministic check of their
        # own are excluded. Runs BEFORE the clock line is appended (the answer
        # is cached per screen context, and the clock changes every minute) and
        # is skipped while queued steps remain (they were planned against this
        # same screen moments ago — ~1 min of pure goal checks per form).
        # _screen_says_not_done records a genuine "not met" verdict so a planner
        # "[]" below can be rejected without paying for a second LLM call.
        _screen_says_not_done = False
        if not run.step_queue:
            # Ground-truth override, checked BEFORE paying for the goal-check
            # LLM: on a subtask's FIRST cycle (nothing actioned yet), an
            # "already satisfied" verdict is not trustworthy while the very
            # control the subtask must actuate is still sitting in the tree
            # ENABLED — that is proof the action has NOT happened. Live: the
            # 8B blessed "click Save to create the meeting" as done because the
            # form was filled and 'Save' was present, so the meeting was never
            # created and the run falsely reported success. Act once here; once
            # a step has run, run.completed is non-empty and this guard is off,
            # so a nav button that stays visible after its click cannot loop.
            if self._first_action_targets_live_control(run, subtask):
                self.log(
                    "  [GOAL-CHECK] Target control present and unactuated — "
                    "acting once before trusting a 'done' verdict"
                )
            elif self._goal_already_satisfied(run, subtask):
                self.log("  [GOAL-CHECK] Goal already satisfied on current screen")
                return ("done", None)
            elif not (run.is_launch_goal or run.is_cmd_subtask
                      or run.save_target or run.type_payload):
                _screen_says_not_done = True

        # Real date/time from the OS — the planner otherwise reasons from the
        # model's frozen sense of "now" (it clicked 'Today' to select a date
        # two days ahead, silently resetting a correctly-set date).
        run.screen_context += time.strftime(
            "\nSYSTEM CLOCK (ground truth): %A, %B %d, %Y %I:%M %p"
        )

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
        if _visual_mode and (run.is_cmd_subtask or system.foreground_is_terminal()):
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
                elif (macro := self._start_menu_launch(subtask, run)) is not None:
                    step, run.step_queue = macro[0], macro[1:]
                else:
                    planned = self.planner.plan_steps(
                        subtask, run.screen_context, run.completed,
                        task_context=task_context,
                    )
                    if not planned:
                        # A "save as <path>" subtask has ONE ground truth:
                        # the file on disk. The planner's "goal achieved"
                        # must not overrule its absence — that reports a
                        # false success with no file produced.
                        if run.save_target and not groundtruth.file_saved_fresh(
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
                        _excluded = (
                            run.is_launch_goal or run.is_cmd_subtask
                            or run.save_target or run.type_payload
                        )
                        # The planner's [] is a CLAIM, never proof — with or
                        # without prior actions. The goal check is the referee:
                        # this cycle's pre-plan check already said the goal is
                        # NOT on screen, so accepting [] here contradicts the
                        # screen (live: 8 'Goal achieved' loops on a subtask
                        # whose form never opened, ending in a false success).
                        confirmed = (
                            (acted or _excluded)
                            and not _screen_says_not_done
                            and self._goal_already_satisfied(
                                run, subtask, skip_exclusions=True
                            )
                        )
                        if not confirmed:
                            self.log(
                                "  Planner declared done but the goal check "
                                "finds no screen evidence — rejecting"
                            )
                            run.completed.append(
                                "[FAILED: planner declared done, but the "
                                "goal state is not visible on screen]"
                            )
                            _outcome = self._note_planning_failure(run)
                            # The text planner just proved it is blind here:
                            # it sees nothing to do on a screen the goal check
                            # says is wrong. A second text attempt returns []
                            # again (deterministic on an unchanged screen) —
                            # skip it and escalate to the screenshot planner
                            # on the next cycle.
                            if (
                                _outcome == "retry"
                                and self.config.visual_replan_after > 0
                            ):
                                run.consecutive_failures = max(
                                    run.consecutive_failures,
                                    self.config.visual_replan_after,
                                )
                            return _outcome, None
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

    def _start_menu_launch(
        self, subtask: SubTask, run: "SubtaskRun"
    ) -> list[ActionStep] | None:
        """The Start-menu launch sequence for an "open <app>" subtask, or None.

        Asking an 8B model how to open a named, installed app is a question
        with one answer. It answered win → type the name → Enter in the live
        Notepad run (14.4 s) and the live Teams run (14.5 s), because that is
        the only answer, and it is the exact route the launch-retry hint tells
        it to use after any other route fails. Emitting it directly buys back
        ~14 s per launch with no loss: the same three actions, each still
        verified, and the launch still has to satisfy groundtruth.verify_launch.

        Returns None — falling back to the planner — unless this is a launch
        goal, nothing has been tried yet, and apps.launch_query() is confident
        about which app to type. An unknown or oddly-worded app still goes to
        the model.
        """
        if not run.is_launch_goal or run.completed:
            return None
        query = apps.launch_query(subtask.description)
        if not query:
            return None
        self.log(
            f"  [LAUNCH] '{query}' is a known app — using the Start-menu "
            f"sequence directly (no planning call)"
        )
        return [
            ActionStep(id=1, subtask_id=subtask.id, action_type="key_press",
                       key="winleft", description="Open Windows Start menu search"),
            ActionStep(id=2, subtask_id=subtask.id, action_type="type",
                       value=query, description=f"Type '{query}' into Start"),
            ActionStep(id=3, subtask_id=subtask.id, action_type="key_press",
                       key="enter", description=f"Launch {query}"),
        ]

    def _run_step_attempts(
        self, run: "SubtaskRun", subtask: SubTask, step: ActionStep,
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
        # Launch subtasks legitimately click OUTSIDE the anchor app (Start
        # menu results, desktop icons) — the pre-click anchor gate must not
        # veto those. Read by _foreign_app_at_point for every attempt below.
        self.anchor.gate_off = run.is_launch_goal
        # "wait" and "extract" have no visible outcome to verify. set_value and
        # select are decided AFTER they run, on whether the tree read the value
        # back (see the [TREE-VERIFY] branch below) — set_value used to skip
        # unconditionally, which also skipped its unverified click+type fallback.
        skip_reflection = (
            step.action_type in ("wait", "extract") or _cmd_type_step
        )
        if _cmd_type_step:
            self.log(
                "  [CMD-TYPE] Command typed — deferring verification to the "
                "Enter/disk check (skipping unreliable OCR reflection)"
            )
        run.last_error = ""

        # COMMIT-GUARD: never let a form COMMIT (Save/Submit/Create/Send) fire
        # before the values the subtask names are actually IN the form. The 8B
        # planner sometimes orders the Save click early, and a queue-flush
        # re-plan can promote it — a Save on a half-filled form creates nothing
        # (or a wrong entry) yet 'verifies' as a successful click, so the run
        # reports success while the calendar stays empty. This is deterministic
        # ground truth: read each required value back from the live UIA control
        # list; if any is still missing, block the commit and make the planner
        # fill the fields first. Only fires when the control list is readable
        # (UIA on) — on the OCR/VLM path we can't prove a field is empty, so we
        # don't block.
        _submit_t = subtasks.submit_target(subtask)
        if (
            _submit_t
            and step.action_type in ("click", "invoke", "double_click")
            and subtasks.targets_match(step.target, _submit_t)
        ):
            _missing = self.truth.unfilled_form_values(subtask, run.screen_context)
            if _missing:
                run.last_error = (
                    f"the form is NOT fully filled yet — {', '.join(_missing)} "
                    f"still missing from the form; set every field BEFORE "
                    f"clicking {_submit_t}"
                )
                self.log(
                    f"  [COMMIT-GUARD] Blocking '{_submit_t}' — form still "
                    f"missing: {', '.join(_missing)}"
                )
                run.step_queue = []   # re-plan against the live, half-filled form
                return "step_failed"

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

        # A left-rail view-switch hotkey (ctrl+<digit>) is idempotent in STATE
        # but useless to RETRY: if the first press didn't switch the view, two
        # more identical presses won't either — each just burns a ~40 s verify
        # round (live 06:56-07:16: ctrl+4 retried 3× per cycle for minutes).
        # Give nav hotkeys ONE attempt and let the planner pick another route
        # (a rail click). A switch that DID happen is confirmed deterministically
        # by the window title in _judge_reflection, so nothing real is lost.
        _nav_hotkey = (
            step.action_type == "hotkey" and bool(re.fullmatch(r"ctrl\+\d", _key_l))
        )
        _max_attempts = 1 if _nav_hotkey else self.config.max_retries_per_step

        for attempt in range(_max_attempts):
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
                    from desktop.capture import frame_phash
                    pre_click_hash = frame_phash(self.capturer.capture())
                except Exception:
                    pass

            # An unhandled exception inside step execution must become a
            # LOGGED step failure, never a silent mission death: the GUI
            # worker swallows exceptions from execute(), so a crash here left
            # zero trace in the terminal (live: an AttributeError in a helper
            # stopped a run mid-form with the log simply ending).
            try:
                _exec_ok = self._execute_step(step)
            except Exception as e:
                logger.exception(f"[STEP-CRASH] Unhandled error executing step: {e}")
                run.last_error = f"internal error executing this step: {e}"
                return "step_failed"
            if not _exec_ok:
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
                        from desktop import uia
                        miss = uia.pop_select_miss()
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

            # Tree read-back beats LLM reflection, and costs ~7 s less.
            # set_value and select write through a UIA pattern and read the
            # control back; when that read-back matched, the value IS on the
            # form and there is nothing an 8B model squinting at OCR text can
            # add. Live 03:40 Teams run: four selects each paid a 2.0 s OCR
            # pass plus a 5.6 s reflection call to re-derive what ValuePattern
            # had already confirmed — ~30 s per meeting.
            #
            # Only when the actor could NOT read the result back (a provider
            # that never exposes selection state, or the pixel click+type
            # fallback) does this fall through to the verifier, exactly as
            # before. `is True` is deliberate: a MagicMock actor in tests must
            # not be mistaken for a verified action.
            if (step.action_type in ("set_value", "select")
                    and getattr(self.actor, "verified_by_readback", False) is True):
                self.log(
                    f"  [TREE-VERIFY] '{step.target}' read back from the "
                    f"accessibility tree — verified without an LLM call"
                )
                return "step_ok"

            # Deterministic type verification: read the focused control's
            # value from the accessibility tree. Exact, ~50 ms, and skips
            # the ~3-4 s OCR+LLM reflection call. Falls through to normal
            # reflection when the control exposes no readable value.
            if step.action_type == "type":
                time.sleep(0.3)   # let the control commit the input
                if self.truth.typed_text_in_focused_control(step.value, step.target):
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
                _ok, _why = self.truth.command_effect(
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
        self, run: "SubtaskRun", step: ActionStep,
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
            # Re-decide the own-GUI mask NOW: the action just performed (an
            # alt+tab, a click) may have raised the agent's own window, and
            # the verifier is about to capture. Without this, the VLM read
            # the agent's own log panel and 'verified' a Calendar click at
            # conf=1.00 while Teams never left the Chat view (live 21:08).
            self.mask.refresh()
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
                and self.anchor.is_set
            ):
                _fg_hwnd, _fg_pid, _fg_name = system.foreground_app()
                if _fg_pid and _fg_pid != self.anchor.pid \
                        and _fg_hwnd != self.mask.hwnd:
                    self.log(
                        f"  [ANCHOR] Click switched foreground to {_fg_name} — "
                        f"wrong element for a task anchored to {self.anchor.exe}"
                    )
                    logger.warning(
                        f"[ORCHESTRATOR] Click on '{step.target}' opened {_fg_name} "
                        f"(task app: {self.anchor.exe}) — marking point dead"
                    )
                    if getattr(self, "_last_click_xy", None):
                        # foreign=True: the point belongs to another app's
                        # window — that stays true across screen changes, so
                        # blacklist it task-wide for every target (the VLM
                        # re-emits identical coordinates at temperature 0).
                        self.grounder.mark_dead(
                            step.target or "[visual]", *self._last_click_xy,
                            foreign=True,
                        )
                    self.anchor.ensure_foreground()
                    run.last_error = (
                        f"click opened {_fg_name} instead of acting inside "
                        f"{self.anchor.exe} — wrong element"
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
                and groundtruth.click_holds_focus(self._last_click_xy)
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
                and self.truth.save_dialog_visible()
            ):
                self.log("  [SAVE-CHECK] Save-As dialog is open — hotkey confirmed")
                return "success"

            # A Teams/Office left-rail view switch (ctrl+<digit>, or a click on
            # the sidebar icon) is proven by the foreground WINDOW TITLE — the
            # one signal that survives OCR/UIA being off. The OCR verifier can
            # never read the title bar, so it scored every switch uncertain and
            # the run retried ctrl+4 into a multi-minute stall (live 06:56-07:16,
            # a Calendar that HAD opened was declared unconfirmed). Check it.
            if (
                step.action_type in ("hotkey", "click", "invoke", "double_click")
                and groundtruth.view_switch_confirmed(step)
            ):
                self.log(
                    "  [VIEW-CHECK] Window title confirms the view switch "
                    "— step verified"
                )
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
                    # type / Enter / Ctrl-V already fired, and "uncertain"
                    # means the verifier could not READ the result (dark
                    # console, OCR miss), not that it failed. Re-doing them is
                    # what caused double-typing and double-submits, so accept
                    # and let the NEXT step read the live screen. Backstops
                    # remain: on-disk command checks, launch verification, and
                    # the loop guard for genuine repeats.
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
                    if groundtruth.new_window_appeared(run.goal_proc, run.baseline_windows):
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
                # Visual-planner clicks have no target — record them under the
                # shared '[visual]' pseudo-target so is_dead_point() can refuse
                # the same provably-inert pixel on this screen next time.
                self.grounder.mark_dead(
                    step.target or "[visual]", *self._last_click_xy
                )
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
                from desktop import uia
                _cover = uia.covering_element(
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
        self, run: "SubtaskRun", subtask: SubTask, step: ActionStep,
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
            and subtasks.texts_equivalent(step.value or "", run.type_payload)
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
        if run.is_launch_goal and groundtruth.new_window_appeared(run.goal_proc, run.baseline_windows):
            _label = run.goal_proc.split(".")[0]
            self.log(f"  [GOAL-CHECK] '{_label}' confirmed — goal achieved")
            return True

        # Early-exit: a bare "click X" subtask is DONE the moment that click is
        # verified. Reaching _record_step_success for a click means reflection
        # already passed — and a click that changed zero pixels FAILS reflection
        # (delta=0), so a passing click provably moved the screen. Trust it
        # instead of handing the OCR-blind goal-check a WebView2 result it can't
        # read: that loop (goal-check reads the opened form's "New meeting" title
        # as "button still present → not clicked" and re-clicks the label) is
        # exactly what burned an 8-minute Teams run. Not gated for multi-part
        # form subtasks — those still run the full plan/verify loop.
        if (
            not run.is_launch_goal
            and step.action_type in ("click", "right_click", "double_click")
            and subtasks.is_single_click(subtask)
        ):
            # A blind visual-planner click (raw coordinates, no target) proves
            # only that SOME pixel changed the screen — nothing ties it to the
            # control the subtask names. Trusting it completed 'click the New
            # meeting button' off a click that had opened a browser tab (live
            # VLM-only run), and the form-fill subtask that followed died on a
            # form that never opened. Let the next cycle's goal check judge
            # from the live screen instead.
            if step.target or subtasks.explicit_coords(step.value) is None:
                self.log(
                    "  [CLICK-CHECK] Single-click subtask — verified click "
                    "landed and changed the screen — subtask complete"
                )
                return True
            self.log(
                "  [CLICK-CHECK] Visual click changed the screen, but its "
                "target is unproven — requiring goal-check evidence before "
                "completing the subtask"
            )

        # Early-exit: a form-fill subtask ends by clicking a COMMIT button
        # (Save / Send / Schedule / Create…) that SUBMITS and then CLOSES the
        # form. The instant that click verifies, the work is done — but its
        # success evidence (the filled form) vanishes with the form, so a
        # goal-check on the next cycle sees no form and reports "not
        # satisfied", and the planner re-clicks Save forever on a form that no
        # longer exists (live: a Teams meeting was created, then the agent
        # spent 6 minutes wandering the Meet page trying to re-Save it). A
        # verified click/invoke on the subtask's own commit button IS the
        # completion — trust it, exactly like the single-click case above.
        _submit = subtasks.submit_target(subtask)
        # Guard: only after the form's fields were actually filled. run.completed
        # already includes THIS commit step, so ">= 2 successful steps" means at
        # least one field-set ran before it — a lone premature Save falls through
        # to the normal goal-check instead of falsely completing.
        _acted_steps = sum(1 for c in run.completed if not c.startswith("[FAILED"))
        if (
            _submit
            and not run.is_launch_goal
            and _acted_steps >= 2
            and step.action_type in ("click", "invoke", "right_click", "double_click")
            and subtasks.targets_match(step.target, _submit)
        ):
            self.log(
                f"  [SUBMIT-CHECK] Form commit '{_submit}' clicked and verified "
                "— the form was submitted, subtask complete"
            )
            return True

        # A [SET-CHECK] no-op recognised an ALREADY-set field — no input was
        # fired, so it cannot be a harmful action loop. Counting it tripped
        # the loop-guard mid-form (live: the third 'Title already set'
        # recognition ended the subtask while attendee/times/Save sat in the
        # 12-step queue). Planner spirals are still bounded by
        # max_steps_per_subtask and the subtask deadline.
        if getattr(self, "_last_step_was_noop", False):
            return None

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
        # Same honesty rule for form-commit subtasks: looping out of a
        # "set fields… then click Save" subtask WITHOUT the commit click ever
        # succeeding means nothing was submitted — returning True here
        # reported 'All sub-tasks completed' for a meeting that was never
        # created (live: loop-guard fired on the third title recognition
        # while Save had never been clicked).
        if _submit and not any(
            _submit.lower() in c.lower()
            for c in run.completed if not c.startswith("[FAILED")
        ):
            self.log(
                f"  [LOOP-GUARD] Form subtask looped but '{_submit}' was "
                "never clicked — nothing was submitted; marking subtask FAILED"
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
        self, run: "SubtaskRun", step: ActionStep,
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
        if run.is_launch_goal and groundtruth.new_window_appeared(run.goal_proc, run.baseline_windows):
            _label = run.goal_proc.split(".")[0]
            self.log(
                f"  [GOAL-CHECK] '{_label}' confirmed "
                f"despite step failure — goal achieved"
            )
            return True

        run.consecutive_failures += 1
        self.log("  Step failed - re-evaluating next action")

        if run.consecutive_failures >= self.config.consecutive_failures_limit:
            self.log(
                f"  {self.config.consecutive_failures_limit} consecutive failures"
                f" — aborting subtask"
            )
            return False
        return None



    # ── Visual planning (UI-TARS recovery path) ────────────────────────────────

    def _plan_visual(self, subtask: SubTask, completed: list[str]):
        """Capture the screen and ask the VLM for the next action directly."""
        import base64
        import io
        self.mask.refresh()
        img = self.capturer.capture()
        thumb = img.copy()
        # 1280 (not 960): UI-TARS-1.5 grounds small targets better at higher
        # resolution, and _parse_visual_action maps the returned absolute pixels
        # against the ACTUAL sent size (display_w/h below), so the larger image
        # improves accuracy without breaking coordinate mapping.
        thumb.thumbnail((1280, 720))
        buf = io.BytesIO()
        thumb.convert("RGB").save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        return self.planner.plan_next_step_visual(
            subtask, img_b64, completed,
            screen_w=self._screen_w, screen_h=self._screen_h,
            display_w=thumb.width, display_h=thumb.height,
        )

    # ── Step execution ─────────────────────────────────────────────────────────


    # Keys that are safe to press even when the agent's own console is focused —
    # they navigate AWAY from it (launcher) or dismiss overlays, never inject
    # input into the console session itself.
    _SAFE_OWN_CONSOLE_KEYS = frozenset({"winleft", "win", "escape", "esc"})

    def _keyboard_blocked_by_own_console(self, step: ActionStep) -> bool:
        """True when keyboard input must be blocked because the agent's own host
        console owns the foreground — typing there injects into the terminal
        session that launched the agent.
        """
        if step.action_type == "key_press" and (step.key or "").lower() in self._SAFE_OWN_CONSOLE_KEYS:
            return False
        if not system.own_console_is_foreground():
            return False
        self.log(
            "  [GUARD] Foreground is the agent's own console — keyboard "
            "input blocked (would inject into the agent's host terminal)"
        )
        return True

    @staticmethod
    def _force_mouse() -> bool:
        """True when grounded clicks must use the real (gliding) mouse rather than
        the UIA pattern-invoke shortcut. Env AGENT_FORCE_MOUSE=1/0 wins; otherwise
        config.FORCE_MOUSE (default True) decides. See config.FORCE_MOUSE.
        """
        env = os.environ.get("AGENT_FORCE_MOUSE", "").strip().lower()
        if env in ("1", "true", "yes"):
            return True
        if env in ("0", "false", "no"):
            return False
        try:
            import config as _cfg  # noqa: PLC0415
            return bool(getattr(_cfg, "FORCE_MOUSE", True))
        except Exception:
            return True

    def _execute_step(self, step: ActionStep) -> bool:
        # Stop pressed: never BEGIN a new action. The step loop already checks
        # between steps, but grounding one target can block 30-60 s on a VLM
        # call — without a check here (and before each dispatch below) the
        # agent finished grounding and still fired the click after the user
        # hit Stop, which is exactly why the button felt dead.
        if self._stop_event.is_set():
            return False
        x = y = None
        # Where the last click-family step actually landed — lets the
        # reflector's "screen unchanged" verdict blacklist the exact point.
        self._last_click_xy = None
        # Specific failure reason for the caller's [FAILED: …] record; the
        # planner reads that record, so name the real blocker when we know it.
        self._exec_fail_reason = ""
        # True when this step succeeded WITHOUT firing any input (a [SET-CHECK]
        # recognised an already-set field). The loop-guard must not count these
        # as repeated actions — see _record_step_success.
        self._last_step_was_noop = False

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
            and system.foreground_is_terminal()
        ):
            self.log(
                "  [GUARD] ctrl+c in a terminal interrupts the shell — blocked "
                "(type a corrected command instead)"
            )
            return False

        # ── Click family ──────────────────────────────────────────────────────
        if step.action_type in ("click", "right_click", "double_click"):
            self._last_was_invoke = False
            # Grounding is about to capture the screen — make sure the agent's
            # own window pixels are masked as of NOW, not as of the last
            # planning cycle.
            self.mask.refresh()
            _coords = subtasks.explicit_coords(step.value)
            if _coords is not None:
                # Explicit pixel coordinates (visual-planner convention)
                x, y = _coords
                # Consult the dead-point blacklist explicit clicks used to
                # bypass: re-clicking a pixel a phash delta=0 already proved
                # inert on THIS screen can never work — fail fast and tell
                # the planner to pick a different element.
                if self.grounder.is_dead_point(x, y):
                    self._exec_fail_reason = (
                        f"point ({x},{y}) was already proven inert on this "
                        f"exact screen (a click there changed nothing) — "
                        f"choose a DIFFERENT element or navigate another way"
                    )
                    self.log(
                        f"  [DEAD-POINT] ({x},{y}) already proven inert on "
                        f"this screen — refusing to re-click it"
                    )
                    return False
                # Pre-click ground truth: a visual-planner point inside
                # another app's window can never advance a task anchored to
                # this app — reject it BEFORE the click steals focus.
                _foreign = self.anchor.foreign_app_at(x, y)
                if _foreign:
                    self.grounder.mark_dead("[visual]", x, y, foreign=True)
                    self._exec_fail_reason = (
                        f"point ({x},{y}) is inside {_foreign}'s window, not "
                        f"the task app's — clicking there would leave the app; "
                        f"choose an element INSIDE the task app's window"
                    )
                    self.log(
                        f"  [ANCHOR-GATE] ({x},{y}) belongs to {_foreign} — "
                        f"refusing to click outside the task app"
                    )
                    return False
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
            # Pre-click anchor gate (OCR/VLM grounding hits especially): the
            # target's text may exist in ANOTHER app's window — a background
            # browser tab, a search-result flyout — and grounding happily
            # returns that pixel. Hit-test whose window owns the point before
            # any input fires; a foreign point is marked dead task-wide so no
            # stage can serve it again.
            _foreign = self.anchor.foreign_app_at(x, y)
            if _foreign:
                self.grounder.mark_dead(step.target or "[visual]", x, y,
                                        foreign=True)
                self._exec_fail_reason = (
                    f"'{step.target}' was located at ({x},{y}), but that "
                    f"point is inside {_foreign}'s window — the matched text "
                    f"belongs to another app, not the task app; find the "
                    f"control INSIDE the task app's window instead"
                )
                self.log(
                    f"  [ANCHOR-GATE] '{step.target}' at ({x},{y}) belongs "
                    f"to {_foreign} — refusing to click outside the task app"
                )
                return False
            # Hit-test before acting: ElementFromPoint is OS ground truth for
            # "what would a click here actually land on". When a dialog or
            # overlay sits on top (live: Teams' 'Meeting created' popup over
            # the form), every click AND pattern invoke beneath it is a
            # provable no-op — fail fast and hand the planner the blocker's
            # name so it dismisses the overlay instead of grinding retries.
            if "uia" in (result.method or ""):
                from desktop import uia
                _cover = uia.covering_element(x, y, step.target)
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
            # A control found in the UIA tree can be actuated through its own
            # Invoke/Toggle pattern instead of a synthesized click: that cannot
            # miss the pixel and cannot be swallowed by an overlay (every click
            # on Outlook's 'New' button landed with delta=0 while the control
            # invoked perfectly). Two exceptions: a target whose invoke already
            # failed verification once goes back to pixels (WebView2 providers
            # accept Invoke and do nothing, and no coordinate is involved for a
            # blacklist to catch), and FORCE_MOUSE delivers every click with the
            # visible mouse at the UIA-exact coordinates. See config.FORCE_MOUSE.
            if (
                step.action_type == "click"
                and "uia" in (result.method or "")
                and (step.target or "").lower() not in self._invoke_dead
                and not self._force_mouse()
            ):
                from desktop import uia
                if self._stop_event.is_set():
                    return False
                # WebView2 providers often THROW from Invoke while the button
                # is still PRESSED. Retrying at the pre-invoke coordinates then
                # clicks whatever the actuation put there — invoking 'New
                # meeting' opened a form whose Save button sat 25px away, and
                # the fallback saved the empty form. So: hash before; if the
                # screen changed, the invoke landed and the stale click is
                # skipped. A provably inert invoke still falls through.
                _pre_invoke = None
                try:
                    from desktop.capture import frame_phash
                    _pre_invoke = frame_phash(self.capturer.capture())
                except Exception:
                    pass
                if uia.invoke_element(step.target, timeout_s=2.5):
                    self._last_was_invoke = True
                    return True
                if _pre_invoke is not None:
                    try:
                        time.sleep(0.4)   # let a real actuation repaint
                        if (frame_phash(self.capturer.capture()) - _pre_invoke) != 0:
                            self.log(
                                f"  [INVOKE-LANDED] Invoke on '{step.target}' "
                                "reported an error but the screen changed — the "
                                "button was pressed; skipping the stale-"
                                "coordinate pixel click"
                            )
                            self._last_was_invoke = True
                            return True
                    except Exception:
                        pass

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
                # mark the run degraded so it is not recorded as a clean
                # success — "read me the value" tasks that found no value must
                # not masquerade as fully-successful in the task history.
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
            # Ground the field label (UIA+OCR only, no VLM) so set_value has a
            # pixel fallback: when the accessibility tree cannot focus the
            # control (UIA disabled, or web widgets with no writable pattern),
            # the actor clicks the field and types instead of failing outright.
            if step.action_type == "set_value" and step.target:
                # 0.8: this hit gets CLICKED then ctrl+a-TYPED into — a wrong
                # match types into arbitrary UI. Live: 'date' fuzzy-matched
                # taskbar garbage 'Wd TE' at 0.67 and the date was typed into
                # a non-Teams surface. Real field labels score >= 0.95.
                _r = self.grounder.ground_fast(step.target, ocr_threshold=0.8)
                if _r.found and _r.confidence >= self.grounder.min_confidence:
                    x, y = _r.x, _r.y
                elif self.truth.value_on_screen(step.value):
                    # Filling a text field REMOVES its placeholder label
                    # ('Add title' vanishes once the title is typed), so a
                    # re-attempt can never ground the label again even though
                    # the work is DONE — live on the OCR path, the title was
                    # set once and three replans then failed 'Title' for
                    # minutes. The value being literally on screen is ground
                    # truth that the field already holds it.
                    self.log(
                        f"  [SET-CHECK] '{step.target}' label not on screen "
                        f"but the value is — field already set"
                    )
                    self._last_step_was_noop = True
                    return True
                else:
                    # Give the planner something actionable: combo fields
                    # (date/time) render only their CURRENT value, never a
                    # label OCR can find.
                    # Live: 'click the value, type the new one' got the field
                    # focused but typing did NOT replace the old value (the
                    # goal-check still read 7/19 after 07/29 was typed) — the
                    # planner's manual sequence had no select-all. Spell out
                    # the full replace sequence.
                    self._exec_fail_reason = (
                        f"field label '{step.target}' is not visible on this "
                        f"screen. Date/time combo fields show only their "
                        f"CURRENT value — click the currently displayed value "
                        f"text on the form, press ctrl+a to select the old "
                        f"value, type the new value, then press enter"
                    )

        # select shares set_value's pixel fallback and needs coordinates too
        # (and the same strict 0.8 OCR bar — its hit also gets clicked+typed)
        elif step.action_type == "select" and step.target:
            _r = self.grounder.ground_fast(step.target, ocr_threshold=0.8)
            if _r.found and _r.confidence >= self.grounder.min_confidence:
                x, y = _r.x, _r.y
            elif self.truth.value_on_screen(step.value):
                self.log(
                    f"  [SET-CHECK] '{step.target}' not groundable but its "
                    f"value is already on screen — field already set"
                )
                self._last_step_was_noop = True
                return True
            else:
                self._exec_fail_reason = (
                    f"'{step.target}' is not visible on this screen. Date/"
                    f"time combo fields show only their CURRENT value — click "
                    f"the currently displayed value text on the form, press "
                    f"ctrl+a to select the old value, type the new value, "
                    f"then press enter"
                )

        # Final stop gate: grounding above may have blocked for seconds on a
        # model call during which the user hit Stop — do not fire the action.
        if self._stop_event.is_set():
            return False
        return self.actor.execute(step, x=x, y=y)

    # ── Deterministic terminal-command verification ─────────────────────────────




















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
        if groundtruth.file_saved_fresh(target, started_at):
            return True

        self.log(f"  [SAVE-AS] Saving deterministically to '{typed_path}'")

        # 1. Open the Save dialog from the editor — unless one is already open
        #    (a prior step may have left it up), in which case reopening it is at
        #    best a no-op and at worst dismisses it.
        if not self.truth.save_dialog_visible():
            if not self._execute_step(
                _step("hotkey", key="ctrl+s", desc="Open Save dialog")
            ):
                return False
            self.truth.wait_for_settle(0.8, 3.0)

        # 2. Confirm a dialog is actually open before typing — otherwise ctrl+s may
        #    have saved silently (already-named file) and typing would corrupt the
        #    document body.
        if not self.truth.save_dialog_visible():
            if groundtruth.file_saved_fresh(target, started_at):
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
                from desktop import uia
                _field_val = (uia.focused_element_info() or {}).get("value")
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
            if groundtruth.file_saved_fresh(target, started_at):
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
            from core.firewall import Decision, Severity, decide, evaluate
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












    # ── Adaptive inter-subtask wait ───────────────────────────────────────────


    # ── Screen context ─────────────────────────────────────────────────────────

    def _get_screen_context(self) -> str:
        """Return a structured screen context string for the planner.

        Uses capture_snapshot() to separate foreground UI text from background
        window text so the planner can focus on interactable elements only.
        Falls back to a flat token list if snapshot capture fails.
        """
        try:
            self.mask.refresh()
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

