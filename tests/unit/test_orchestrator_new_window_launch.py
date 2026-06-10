# tests/unit/test_orchestrator_new_window_launch.py
"""
Regression tests for the "app already running" launch semantics.

Live failure this guards against: "open Windows Terminal" was satisfied by
clicking the taskbar button of the user's EXISTING terminal (which was running
another program), and the agent then typed shell commands into that session.

Rules under test:
  1. When the goal app's process pre-exists the subtask, the planner receives a
     [NOTE] instructing it to open a NEW window (never focus the existing one).
  2. _verify_launch with a recorded window-count baseline passes ONLY when the
     window count increased; bare process existence is no longer sufficient.
  3. Without a baseline (app was not running), behaviour is unchanged.
"""
import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

from core.orchestrator import TaskOrchestrator, OrchestratorConfig
from core.protocols.a2a import SubTask


def _sub(desc):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_orch():
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(plan_next_step=MagicMock(return_value=None)),
        grounder=MagicMock(min_confidence=0.5),
        actor=MagicMock(execute=MagicMock(return_value=True)),
        reflector=MagicMock(min_confidence=0.75),
        capturer=MagicMock(),
        task_memory=MagicMock(
            get_failure_hints=MagicMock(return_value=[]),
            store_failure_pattern=MagicMock(),
        ),
        config=OrchestratorConfig(max_retries_per_step=1, max_steps_per_subtask=3),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"Terminal"')
    return orch


class TestPreExistingAppNote:

    def test_note_injected_when_app_already_running(self):
        """Process pre-exists → planner's task_context gets the NEW-window NOTE."""
        orch = _make_orch()
        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            orch._execute_subtask(_sub("open windows terminal"))

        ctx = orch.planner.plan_next_step.call_args.kwargs.get("task_context")
        assert ctx, "task_context must be passed to the planner"
        joined = " ".join(ctx)
        assert "ALREADY running" in joined
        assert "NEW" in joined
        # Baseline must be recorded for _verify_launch
        assert orch._launch_window_baseline.get("WindowsTerminal.exe") == 1

    def test_no_note_when_app_not_running(self):
        orch = _make_orch()
        with patch.object(orch, "_is_process_running", return_value=False), \
             patch.object(orch, "_count_process_windows", return_value=0):
            orch._execute_subtask(_sub("open windows terminal"))

        ctx = orch.planner.plan_next_step.call_args.kwargs.get("task_context")
        assert not ctx or "ALREADY running" not in " ".join(ctx)
        assert "WindowsTerminal.exe" not in orch._launch_window_baseline

    def test_non_launch_subtask_records_no_baseline(self):
        orch = _make_orch()
        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            orch._execute_subtask(_sub("with the terminal already open, run: dir"))
        assert orch._launch_window_baseline == {}


class TestVerifyLaunchWithBaseline:

    def test_focusing_existing_window_does_not_pass(self):
        """Window count flat at baseline → launch NOT confirmed."""
        orch = _make_orch()
        orch._launch_window_baseline["WindowsTerminal.exe"] = 1
        with patch.object(orch, "_count_process_windows", return_value=1), \
             patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_launch_confirmed", return_value=True), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub("open windows terminal")) is False, (
                "bare process existence must NOT confirm a launch when the app "
                "pre-existed the subtask"
            )

    def test_new_window_passes(self):
        orch = _make_orch()
        orch._launch_window_baseline["WindowsTerminal.exe"] = 1
        with patch.object(orch, "_count_process_windows", return_value=2), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub("open windows terminal")) is True

    def test_no_baseline_falls_back_to_process_check(self):
        orch = _make_orch()
        with patch.object(orch, "_launch_confirmed", return_value=True), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub("open windows terminal")) is True


class TestCtrlCTerminalGuard:

    def test_ctrl_c_blocked_when_foreground_is_terminal(self):
        from core.protocols.a2a import ActionStep
        step = ActionStep(id=1, subtask_id=1, action_type="hotkey", target=None,
                          value=None, key="ctrl+c", description="copy error",
                          verification="")
        orch = _make_orch()
        with patch.object(orch, "_foreground_is_terminal", return_value=True):
            assert orch._execute_step(step) is False
        orch.actor.execute.assert_not_called()

    def test_ctrl_c_allowed_outside_terminal(self):
        from core.protocols.a2a import ActionStep
        step = ActionStep(id=1, subtask_id=1, action_type="hotkey", target=None,
                          value=None, key="ctrl+c", description="copy text",
                          verification="")
        orch = _make_orch()
        with patch.object(orch, "_foreground_is_terminal", return_value=False):
            orch._execute_step(step)
        orch.actor.execute.assert_called_once()


class TestLoopGuardCommandSubtask:

    def _looping_orch(self, desc):
        """Orchestrator where a key_press enter loops 5× after one failure."""
        from core.protocols.a2a import ActionStep
        from agents.reflection.reflection_agent import ReflectionResult

        enter = ActionStep(id=1, subtask_id=1, action_type="key_press",
                           target=None, value=None, key="enter",
                           description="Execute command", verification="")
        fail = ReflectionResult(success=False, confidence=0.9, observation="err",
                                error_description="Access denied",
                                should_retry=True, recovery_hint="", ocr_text="")
        ok = ReflectionResult(success=True, confidence=1.0, observation="ok",
                              error_description="", should_retry=False,
                              recovery_hint="", ocr_text="")
        orch = _make_orch()
        orch.config.max_steps_per_subtask = 10
        orch.config.max_retries_per_step = 1
        orch.config.consecutive_failures_limit = 10
        orch.config.visual_replan_after = 0
        orch.planner.plan_next_step = MagicMock(return_value=enter)
        # First execution fails, then the identical step "succeeds" repeatedly
        # until the loop guard fires (key_press dedup limit = 3).
        orch.reflector.verify = MagicMock(side_effect=[fail] + [ok] * 10)
        return orch

    def test_command_subtask_loop_after_failure_fails(self):
        orch = self._looping_orch("with the terminal already open, run: echo x > f.txt")
        result = orch._execute_subtask(
            _sub("with the terminal already open, run: echo x > f.txt"))
        assert result is False, (
            "a 'run:' subtask that looped after a failed execution must FAIL — "
            "dependents would otherwise build on state that doesn't exist"
        )

    def test_non_command_subtask_loop_still_returns_true(self):
        orch = self._looping_orch("press arrow keys")
        result = orch._execute_subtask(_sub("press arrow keys repeatedly"))
        assert result is True
        assert orch._degraded is True


class TestGoalCheckWithBaseline:

    def test_goal_check_requires_new_window_when_preexisting(self):
        """The in-loop GOAL-CHECK early exit must not fire while count is flat."""
        from core.protocols.a2a import ActionStep
        from agents.reflection.reflection_agent import ReflectionResult

        click = ActionStep(id=1, subtask_id=1, action_type="click",
                           target="Terminal", value=None, key=None,
                           description="click Terminal", verification="")
        ok = ReflectionResult(success=True, confidence=0.9, observation="ok",
                              error_description="", should_retry=False,
                              recovery_hint="", ocr_text="")

        orch = _make_orch()
        orch.planner.plan_next_step = MagicMock(side_effect=[click, None])
        orch.reflector.verify = MagicMock(return_value=ok)
        from agents.grounding.grounding_agent import GroundingResult
        orch.grounder.ground = MagicMock(return_value=GroundingResult(
            found=True, confidence=0.9, x=1, y=2, latency_ms=1.0,
            target="Terminal", element_type="foreground_interactive"))

        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            result = orch._execute_subtask(_sub("open windows terminal"))

        # Subtask still completes (planner returned None), but it must have been
        # the planner's decision — the GOAL-CHECK shortcut must not have fired
        # after the first step (count never rose above baseline), so the planner
        # must have been called a second time.
        assert result is True
        assert orch.planner.plan_next_step.call_count == 2
