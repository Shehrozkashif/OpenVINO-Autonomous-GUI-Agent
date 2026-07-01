# tests/unit/test_burst_executor.py
"""Unit tests for Fix 1.1 – ActionBurst.

Three test classes:

1. TestDetectBurst
   Pattern matching: verify detect_burst() returns the right ActionBurst (or None)
   for each supported description pattern.

2. TestBurstExecutorRun
   Execution behaviour: order, 80 ms inter-step delay (mock sleep), grounding
   failure path, actor failure path, timeout path, verify_at_end paths.

3. TestOrchestratorBurstIntegration
   Orchestrator wiring: matching subtask runs burst first; if burst fails the
   planning loop runs; if no pattern matches the planning loop runs directly.
"""
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

import pytest

from agents.reflection import ReflectionResult
from core.burst_executor import _INTER_STEP_DELAY_S, BurstExecutor, detect_burst
from core.protocols import ActionBurst, ActionStep, BurstResult, SubTask

# ── helpers ───────────────────────────────────────────────────────────────────

def _subtask(desc: str, sid: int = 1):
    return SubTask(id=sid, description=desc, depends_on=[])


def _step(action_type="click", target="Btn", key=None, value=None, sid=1, step_id=1):
    return ActionStep(
        id=step_id, subtask_id=sid,
        action_type=action_type, target=target, value=value, key=key,
        description=f"{action_type} {target or key or value}", verification="",
    )


_OK_GROUND = MagicMock(found=True, confidence=0.9, x=100, y=200)
_FAIL_GROUND = MagicMock(found=False, confidence=0.0, x=0, y=0)

_SUCCESS_REFLECTION = ReflectionResult(
    success=True, confidence=0.9, observation="ok",
    error_description="", should_retry=False, recovery_hint="", ocr_text="",
)
_FAIL_REFLECTION = ReflectionResult(
    success=False, confidence=0.8, observation="nothing changed",
    error_description="step had no effect", should_retry=True, recovery_hint="", ocr_text="",
)


def _make_executor(ground_ok=True, actor_ok=True, reflector=None):
    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(return_value=_OK_GROUND if ground_ok else _FAIL_GROUND)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=actor_ok)

    return BurstExecutor(grounder=grounder, actor=actor, reflector=reflector), grounder, actor


# ── 1. Pattern detection ──────────────────────────────────────────────────────

class TestDetectBurst:
    """detect_burst() maps description text → ActionBurst or None."""

    # ── Pattern 1: new folder ─────────────────────────────────────────────────

    def test_new_folder_detected(self):
        burst = detect_burst(_subtask("create a new folder on the desktop"))
        assert burst is not None

    def test_new_folder_has_five_steps(self):
        burst = detect_burst(_subtask("create new folder"))
        assert len(burst.steps) == 5

    def test_new_folder_step_types(self):
        burst = detect_burst(_subtask("make a folder on the desktop"))
        types = [s.action_type for s in burst.steps]
        assert types == ["right_click", "click", "click", "type", "key_press"]

    def test_new_folder_targets(self):
        burst = detect_burst(_subtask("create a new folder"))
        # right_click uses explicit coords (no UIA target); click steps target
        # menu items by name for OCR-based late grounding at runtime.
        assert burst.steps[0].target is None
        assert burst.steps[1].target == "New"
        assert burst.steps[2].target == "Folder"
        assert burst.steps[3].target is None   # type step
        assert burst.steps[4].target is None   # key_press step

    def test_new_folder_right_click_has_explicit_coords(self):
        """right_click step must carry explicit pixel coords in value to bypass UIA."""
        burst = detect_burst(_subtask("create new folder"))
        step0 = burst.steps[0]
        assert step0.value is not None
        parts = step0.value.split(",")
        assert len(parts) == 2
        assert all(p.strip().isdigit() for p in parts), f"Expected 'x,y' digits, got '{step0.value}'"

    def test_new_folder_menu_clicks(self):
        """Steps 2 and 3 target 'New' and 'Folder' — grounded visually at runtime."""
        burst = detect_burst(_subtask("create new folder"))
        assert burst.steps[1].action_type == "click"
        assert burst.steps[1].target == "New"
        assert burst.steps[2].action_type == "click"
        assert burst.steps[2].target == "Folder"

    def test_new_folder_type_and_confirm(self):
        burst = detect_burst(_subtask("create new folder"))
        assert burst.steps[3].value == "NewFolder"
        assert burst.steps[4].key == "enter"

    def test_new_folder_verify_at_end_true(self):
        burst = detect_burst(_subtask("create new folder"))
        assert burst.verify_at_end is True

    def test_new_folder_uses_subtask_id(self):
        burst = detect_burst(_subtask("create new folder", sid=7))
        assert all(s.subtask_id == 7 for s in burst.steps)

    # ── Pattern 2: right-click + select ──────────────────────────────────────

    def test_right_click_and_select_detected(self):
        burst = detect_burst(_subtask("right click on the file and select open"))
        assert burst is not None
        assert len(burst.steps) == 2

    def test_right_click_step_types(self):
        burst = detect_burst(_subtask("right click on the file and select open"))
        assert burst.steps[0].action_type == "right_click"
        assert burst.steps[1].action_type == "click"

    def test_right_click_target_extracted(self):
        burst = detect_burst(_subtask("right click on the file and click rename"))
        assert "file" in burst.steps[0].target
        assert "rename" in burst.steps[1].target

    # ── Pattern 3: type + enter ───────────────────────────────────────────────

    def test_type_enter_detected(self):
        burst = detect_burst(_subtask("type 'TestFolder' and press enter"))
        assert burst is not None
        assert len(burst.steps) == 2

    def test_type_enter_step_types(self):
        burst = detect_burst(_subtask("type MyFile and press enter"))
        assert burst.steps[0].action_type == "type"
        assert burst.steps[1].action_type == "key_press"

    def test_type_enter_value_extracted(self):
        burst = detect_burst(_subtask("type HelloWorld and press enter"))
        assert burst.steps[0].value == "helloworld"  # lowercased desc

    def test_type_enter_key_is_enter(self):
        burst = detect_burst(_subtask("type SomeName then press enter"))
        assert burst.steps[1].key == "enter"

    def test_type_enter_verify_at_end_false(self):
        burst = detect_burst(_subtask("type TestFolder and press enter"))
        assert burst.verify_at_end is False

    # ── Pattern 4: select all + type ─────────────────────────────────────────

    def test_ctrl_a_type_detected(self):
        burst = detect_burst(_subtask("ctrl+a then type NewContent"))
        assert burst is not None
        assert len(burst.steps) == 2

    def test_select_all_type_detected(self):
        burst = detect_burst(_subtask("select all and type replacement text"))
        assert burst is not None

    def test_ctrl_a_step_types(self):
        burst = detect_burst(_subtask("ctrl+a then type NewContent"))
        assert burst.steps[0].action_type == "hotkey"
        assert burst.steps[1].action_type == "type"

    def test_ctrl_a_key_is_ctrl_a(self):
        burst = detect_burst(_subtask("ctrl+a then type NewContent"))
        assert burst.steps[0].key == "ctrl+a"

    # ── No match ──────────────────────────────────────────────────────────────

    def test_open_browser_no_burst(self):
        assert detect_burst(_subtask("open firefox browser")) is None

    def test_click_button_no_burst(self):
        assert detect_burst(_subtask("click the save button")) is None

    def test_empty_description_no_burst(self):
        assert detect_burst(_subtask("")) is None

    def test_generic_task_no_burst(self):
        assert detect_burst(_subtask("launch the calculator application")) is None


# ── 2. BurstExecutor.run() ────────────────────────────────────────────────────

class TestBurstExecutorRun:

    # ── Execution order ───────────────────────────────────────────────────────

    def test_steps_executed_in_order(self):
        s1 = _step("right_click", target="desktop", step_id=1)
        s2 = _step("click",       target="New",     step_id=2)
        s3 = _step("click",       target="Folder",  step_id=3)
        burst = ActionBurst(steps=[s1, s2, s3], verify_at_end=False)

        executor, grounder, actor = _make_executor()
        result = executor.run(burst)

        assert result.success is True
        executed = [c[0][0] for c in actor.execute.call_args_list]
        assert executed == [s1, s2, s3]

    def test_pre_groundable_steps_grounded_before_execution(self):
        """Targets visible before the burst starts are all grounded before any step fires."""
        s1 = _step("right_click", target="A", step_id=1)
        s2 = _step("click",       target="B", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        ground_calls = []
        execute_calls = []

        executor, grounder, actor = _make_executor()
        grounder.ground = MagicMock(side_effect=lambda t: (ground_calls.append(t), _OK_GROUND)[1])
        actor.execute   = MagicMock(side_effect=lambda s, **kw: (execute_calls.append(s), True)[1])

        executor.run(burst)

        # Both targets found pre-execution → both grounded before first actor call.
        assert set(ground_calls) == {"A", "B"}
        assert len(execute_calls) == 2

    def test_transient_target_grounded_at_execution_time(self):
        """A target not found pre-execution is late-grounded during step execution."""
        s1 = _step("right_click", target="A", step_id=1)  # succeeds pre-ground
        s2 = _step("click", target="Transient", step_id=2)  # fails pre, late-grounds ok

        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        ground_call_times: list = []

        counter = {"t": 0}
        def _inc_and_return_ok(t):
            counter["t"] += 1
            ground_call_times.append(counter["t"])
            return _OK_GROUND if t == "A" else _FAIL_GROUND

        executor, grounder, actor = _make_executor()
        # "A" succeeds pre-ground; "Transient" fails pre-ground (late-grounded later)
        grounder.ground = MagicMock(side_effect=lambda t: _OK_GROUND if t == "A" else _FAIL_GROUND)

        # Second call to ground("Transient") happens during execution — override to succeed
        call_count: dict = {"Transient": 0}
        def _smart_ground(t):
            if t == "Transient":
                call_count["Transient"] += 1
                # First call (Phase 1): fail — not visible yet
                # Second call (Phase 2): succeed — context menu now open
                return _FAIL_GROUND if call_count["Transient"] == 1 else _OK_GROUND
            return _OK_GROUND
        grounder.ground = MagicMock(side_effect=_smart_ground)
        actor.execute = MagicMock(return_value=True)

        result = executor.run(burst)

        assert result.success is True
        # ground("Transient") called twice: once in Phase 1 (fail), once in Phase 2 (ok)
        assert call_count["Transient"] == 2

    def test_coordinates_passed_to_actor(self):
        """Coordinates from grounding must reach actor.execute as x/y kwargs."""
        s1 = _step("click", target="Btn", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=False)

        grounder = MagicMock()
        grounder.min_confidence = 0.5
        grounder.ground = MagicMock(return_value=MagicMock(found=True, confidence=0.9, x=42, y=77))

        actor = MagicMock()
        actor.execute = MagicMock(return_value=True)

        executor = BurstExecutor(grounder=grounder, actor=actor)
        executor.run(burst)

        actor.execute.assert_called_once_with(s1, x=42, y=77)

    def test_no_grounding_for_type_steps(self):
        """Type and key_press steps do not trigger grounding."""
        s1 = _step("type", target=None, value="hello", step_id=1)
        s2 = _step("key_press", target=None, key="enter", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        executor, grounder, actor = _make_executor()
        executor.run(burst)

        grounder.ground.assert_not_called()

    def test_explicit_coords_in_value_bypass_grounder(self):
        """If value encodes 'x,y', grounder.ground() must NOT be called for that step."""
        s1 = ActionStep(
            id=1, subtask_id=1,
            action_type="right_click", target="desktop", value="960,540",
            key=None, description="right click desktop", verification="",
        )
        burst = ActionBurst(steps=[s1], verify_at_end=False)
        executor, grounder, actor = _make_executor()
        actor.execute = MagicMock(return_value=True)

        result = executor.run(burst)

        assert result.success is True
        grounder.ground.assert_not_called()
        # Actor receives the parsed explicit coordinates
        actor.execute.assert_called_once_with(s1, x=960, y=540)

    # ── Inter-step delay (80 ms) ──────────────────────────────────────────────

    @patch("core.burst_executor.time")
    def test_sleep_called_between_steps_not_after_last(self, mock_time):
        mock_time.time.return_value = 0.0
        s1 = _step("type", value="a", step_id=1)
        s2 = _step("type", value="b", step_id=2)
        s3 = _step("type", value="c", step_id=3)
        burst = ActionBurst(steps=[s1, s2, s3], verify_at_end=False, timeout_ms=9999)

        executor, _, actor = _make_executor()
        actor.execute = MagicMock(return_value=True)
        executor.run(burst)

        # N-1 = 2 sleep calls (between steps 1→2 and 2→3, not after 3)
        assert mock_time.sleep.call_count == 2

    @patch("core.burst_executor.time")
    def test_sleep_duration_is_80ms_for_type_steps(self, mock_time):
        """Type steps use the default 80 ms inter-step delay."""
        mock_time.time.return_value = 0.0
        s1 = _step("type", value="x", step_id=1)
        s2 = _step("type", value="y", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False, timeout_ms=9999)

        executor, _, actor = _make_executor()
        actor.execute = MagicMock(return_value=True)
        executor.run(burst)

        for c in mock_time.sleep.call_args_list:
            assert c[0][0] == pytest.approx(_INTER_STEP_DELAY_S)

    @patch("core.burst_executor.time")
    def test_sleep_duration_is_300ms_for_key_press(self, mock_time):
        """key_press steps use a longer 300 ms delay to allow submenu animations."""
        mock_time.time.return_value = 0.0
        s1 = _step("key_press", key="w", step_id=1)
        s2 = _step("key_press", key="f", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False, timeout_ms=9999)

        executor, _, actor = _make_executor()
        actor.execute = MagicMock(return_value=True)
        executor.run(burst)

        for c in mock_time.sleep.call_args_list:
            assert c[0][0] == pytest.approx(0.30)

    @patch("core.burst_executor.time")
    def test_single_step_burst_no_sleep(self, mock_time):
        mock_time.time.return_value = 0.0
        burst = ActionBurst(steps=[_step("type", value="x")], verify_at_end=False, timeout_ms=9999)
        executor, _, actor = _make_executor()
        actor.execute = MagicMock(return_value=True)
        executor.run(burst)
        mock_time.sleep.assert_not_called()

    # ── Grounding failure path ────────────────────────────────────────────────

    def test_grounding_failure_aborts_burst(self):
        s1 = _step("click", target="Missing", step_id=1)
        s2 = _step("click", target="Also",    step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        executor, _, actor = _make_executor(ground_ok=False)
        result = executor.run(burst)

        assert result.success is False
        assert result.failed_at_step == 0
        assert "grounding" in result.reason
        actor.execute.assert_not_called()

    def test_grounding_failure_reports_correct_step_index(self):
        """Grounding fails on step 1 (second step). failed_at_step must be 1."""
        s1 = _step("type", value="ok",      step_id=1)   # no grounding
        s2 = _step("click", target="Gone",  step_id=2)   # grounding fails

        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        grounder = MagicMock()
        grounder.min_confidence = 0.5
        grounder.ground = MagicMock(return_value=_FAIL_GROUND)

        actor = MagicMock()
        actor.execute = MagicMock(return_value=True)

        executor = BurstExecutor(grounder=grounder, actor=actor)
        result = executor.run(burst)

        assert result.success is False
        assert result.failed_at_step == 1
        actor.execute.assert_not_called()  # pre-ground phase, no execution yet

    # ── Actor failure path ────────────────────────────────────────────────────

    def test_actor_failure_returns_failure_result(self):
        s1 = _step("type", value="x", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=False)

        executor, _, actor = _make_executor(actor_ok=False)
        result = executor.run(burst)

        assert result.success is False
        assert result.failed_at_step == 0
        assert "actor.execute failed" in result.reason

    def test_actor_failure_at_second_step(self):
        s1 = _step("type", value="a", step_id=1)
        s2 = _step("type", value="b", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False)

        executor, _, actor = _make_executor()
        actor.execute = MagicMock(side_effect=[True, False])

        result = executor.run(burst)
        assert result.success is False
        assert result.failed_at_step == 1

    # ── Timeout path ─────────────────────────────────────────────────────────

    @patch("core.burst_executor.time")
    def test_timeout_aborts_before_first_step(self, mock_time):
        # deadline = 100.0 + 0.005 = 100.005; first check is 200.0 > 100.005 → timeout
        mock_time.time.side_effect = [100.0, 200.0]

        s1 = _step("type", value="a", step_id=1)
        s2 = _step("type", value="b", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=False, timeout_ms=5)

        executor, _, actor = _make_executor()
        result = executor.run(burst)

        assert result.success is False
        assert result.failed_at_step == 0
        assert "timeout" in result.reason
        actor.execute.assert_not_called()

    # ── verify_at_end paths ───────────────────────────────────────────────────

    def test_verify_at_end_true_calls_reflector_once(self):
        reflector = MagicMock()
        reflector.verify = MagicMock(return_value=_SUCCESS_REFLECTION)

        s1 = _step("type", value="x", step_id=1)
        s2 = _step("key_press", key="enter", step_id=2)
        burst = ActionBurst(steps=[s1, s2], verify_at_end=True)

        executor, _, actor = _make_executor(reflector=reflector)
        actor.execute = MagicMock(return_value=True)
        result = executor.run(burst)

        assert result.success is True
        reflector.verify.assert_called_once()
        # Called on the LAST step
        assert reflector.verify.call_args[0][0] is s2

    def test_verify_at_end_false_does_not_call_reflector(self):
        reflector = MagicMock()
        s1 = _step("type", value="x", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=False)

        executor, _, actor = _make_executor(reflector=reflector)
        actor.execute = MagicMock(return_value=True)
        executor.run(burst)

        reflector.verify.assert_not_called()

    def test_verify_at_end_failure_returns_failure_result(self):
        reflector = MagicMock()
        reflector.verify = MagicMock(return_value=_FAIL_REFLECTION)

        s1 = _step("type", value="x", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=True)

        executor, _, actor = _make_executor(reflector=reflector)
        actor.execute = MagicMock(return_value=True)
        result = executor.run(burst)

        assert result.success is False
        assert result.failed_at_step == 0  # last (and only) step

    def test_verify_at_end_reflection_exception_treated_as_success(self):
        """Infrastructure failure in reflection → burst still succeeds (same policy as main loop)."""
        reflector = MagicMock()
        reflector.verify = MagicMock(side_effect=RuntimeError("model crashed"))

        s1 = _step("type", value="x", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=True)

        executor, _, actor = _make_executor(reflector=reflector)
        actor.execute = MagicMock(return_value=True)
        result = executor.run(burst)

        assert result.success is True

    def test_no_reflector_and_verify_at_end_true_still_succeeds(self):
        """verify_at_end=True but reflector=None → no crash, just skip."""
        s1 = _step("type", value="x", step_id=1)
        burst = ActionBurst(steps=[s1], verify_at_end=True)

        executor = BurstExecutor(grounder=MagicMock(min_confidence=0.5), actor=MagicMock())
        executor.actor.execute = MagicMock(return_value=True)
        result = executor.run(burst)

        assert result.success is True


# ── 3. Orchestrator integration ───────────────────────────────────────────────

class TestOrchestratorBurstIntegration:
    """Verify that the orchestrator calls burst_executor.run() for matching
    subtasks, returns True on success, and falls back to the planning loop
    on failure or when no pattern matches.
    """

    def _make_orch(self, planner_side_effect=None):
        from agents.reflection import ReflectionResult
        from core.orchestrator import OrchestratorConfig, TaskOrchestrator

        reflector = MagicMock()
        reflector.min_confidence = 0.75
        reflector.verify = MagicMock(return_value=ReflectionResult(
            success=True, confidence=0.9, observation="ok",
            error_description="", should_retry=False, recovery_hint="", ocr_text="",
        ))

        planner = MagicMock()
        planner.plan_next_step = MagicMock(
            side_effect=planner_side_effect if planner_side_effect else [None]
        )

        orch = TaskOrchestrator(
            router=MagicMock(),
            planner=planner,
            grounder=MagicMock(min_confidence=0.5,
                               ground=MagicMock(return_value=_OK_GROUND)),
            actor=MagicMock(execute=MagicMock(return_value=True)),
            reflector=reflector,
            capturer=MagicMock(),
            task_memory=MagicMock(get_failure_hints=MagicMock(return_value=[]),
                                  store_failure_pattern=MagicMock()),
            config=OrchestratorConfig(max_retries_per_step=1,
                                       max_steps_per_subtask=5,
                                       consecutive_failures_limit=3),
            on_step_log=lambda _: None,
        )
        orch._get_screen_context = MagicMock(return_value='"desktop"')
        # Replace burst_executor with a controllable mock
        orch.burst_executor = MagicMock()
        return orch

    def test_matching_subtask_runs_burst_not_planner(self):
        """A 'create new folder' subtask triggers burst; planner is never called."""
        orch = self._make_orch()
        orch.burst_executor.run.return_value = BurstResult(True, None, "ok")

        result = orch._execute_subtask(SubTask(id=1, description="create a new folder", depends_on=[]))

        assert result is True
        orch.burst_executor.run.assert_called_once()
        orch.planner.plan_next_step.assert_not_called()

    def test_burst_success_returns_true(self):
        orch = self._make_orch()
        orch.burst_executor.run.return_value = BurstResult(True, None, "ok")
        result = orch._execute_subtask(SubTask(id=1, description="create new folder", depends_on=[]))
        assert result is True

    def test_failed_burst_falls_back_to_planning_loop(self):
        """Burst failure → orchestrator continues with the normal planning loop."""
        orch = self._make_orch(planner_side_effect=[None])
        orch.burst_executor.run.return_value = BurstResult(False, 0, "grounding failed")

        result = orch._execute_subtask(SubTask(id=1, description="create new folder", depends_on=[]))

        assert result is True  # planner returned None → goal achieved
        orch.planner.plan_next_step.assert_called()

    def test_no_burst_pattern_goes_directly_to_planner(self):
        """Subtask with no burst pattern → burst_executor.run is never called."""
        orch = self._make_orch(planner_side_effect=[None])
        result = orch._execute_subtask(SubTask(id=1, description="open the file manager", depends_on=[]))
        assert result is True
        orch.burst_executor.run.assert_not_called()
        orch.planner.plan_next_step.assert_called()

    def test_burst_executor_receives_correct_burst_object(self):
        """The ActionBurst passed to burst_executor.run() uses visual click steps for menu items."""
        orch = self._make_orch()
        orch.burst_executor.run.return_value = BurstResult(True, None, "ok")

        orch._execute_subtask(SubTask(id=3, description="create a new folder", depends_on=[]))

        burst_arg = orch.burst_executor.run.call_args[0][0]
        assert isinstance(burst_arg, ActionBurst)
        assert len(burst_arg.steps) == 5
        action_types = [s.action_type for s in burst_arg.steps]
        assert action_types == ["right_click", "click", "click", "type", "key_press"]
        # Menu items grounded visually by target name
        assert burst_arg.steps[1].target == "New"
        assert burst_arg.steps[2].target == "Folder"
