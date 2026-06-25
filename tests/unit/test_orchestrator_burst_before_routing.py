# tests/unit/test_orchestrator_burst_before_routing.py
"""
Unit tests for Fix C — instruction-level burst detection in orchestrator.execute().

Verifies:
  1. When detect_burst_from_instruction matches, router.decompose is NOT called.
  2. The synthetic SubTask passed to _execute_subtask has burst set to the detected burst.
  3. When no burst matches, the router IS called normally.
  4. detect_burst_from_instruction correctly recognises the full folder-creation instruction.
  5. _execute_subtask uses subtask.burst directly when it is set (no second detect_burst call).
"""
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch

from core.executor.burst_executor import detect_burst_from_instruction
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.protocols.a2a import ActionBurst, ActionStep, SubTask

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_orch() -> TaskOrchestrator:
    """Minimal orchestrator with collaborators fully mocked."""
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(),
        grounder=MagicMock(),
        actor=MagicMock(),
        reflector=MagicMock(),
        capturer=MagicMock(),
        task_memory=MagicMock(),
        config=OrchestratorConfig(max_steps_per_subtask=1),
        on_step_log=lambda _: None,
        ocr=MagicMock(),
    )
    orch.router.summarize_completion = MagicMock(return_value="done")
    orch.memory.find_similar = MagicMock(return_value=None)
    orch.memory.store_successful_task = MagicMock()
    orch._get_screen_context = MagicMock(return_value='"desktop"')
    # Patch out the real subtask execution and launch check so tests stay fast
    orch._execute_subtask = MagicMock(return_value=True)
    orch._verify_launch = MagicMock(return_value=True)
    orch._wait_for_settle = MagicMock()
    return orch


def _fake_burst() -> ActionBurst:
    step = ActionStep(
        id=1, subtask_id=1, action_type="right_click",
        target="desktop", description="test", verification="",
    )
    return ActionBurst(steps=[step], verify_at_end=False, timeout_ms=3000)


# ── tests: routing skipped when burst detected ────────────────────────────────

class TestBurstSkipsRouter:

    def test_router_decompose_not_called_when_burst_detected(self):
        """router.decompose must not be called when a burst is found at instruction level."""
        orch = _make_orch()
        burst = _fake_burst()
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")
        orch.router.decompose.assert_not_called()

    def test_router_decompose_called_when_no_burst(self):
        """router.decompose IS called when detect_burst_from_instruction returns None."""
        orch = _make_orch()
        mock_subtask = SubTask(id=1, description="open Notepad", depends_on=[])
        orch.router.decompose = MagicMock(return_value=("task_1", [mock_subtask]))
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=None):
            orch.execute("open Notepad")
        orch.router.decompose.assert_called_once()

    def test_only_one_subtask_created_for_burst_instruction(self):
        """Instruction-level burst produces exactly one synthetic subtask."""
        orch = _make_orch()
        burst = _fake_burst()
        captured = []
        def _capture(subtask, **kwargs):
            captured.append(subtask)
            return True
        orch._execute_subtask = _capture

        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")

        assert len(captured) == 1, "Exactly one synthetic subtask must be created"

    def test_synthetic_subtask_has_burst_attached(self):
        """The synthetic subtask passed to _execute_subtask carries burst=<detected>."""
        orch = _make_orch()
        burst = _fake_burst()
        captured_subtask = []

        def _capture(subtask, **kwargs):
            captured_subtask.append(subtask)
            return True
        orch._execute_subtask = _capture

        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")

        assert captured_subtask[0].burst is burst, (
            "subtask.burst must be the ActionBurst returned by detect_burst_from_instruction"
        )

    def test_synthetic_subtask_description_is_full_instruction(self):
        """The synthetic subtask description is the full instruction, not a router fragment."""
        orch = _make_orch()
        instruction = "Right click on the desktop, click New, click Folder, type MyFolder, press Enter."
        captured_subtask = []

        def _capture(subtask, **kwargs):
            captured_subtask.append(subtask)
            return True
        orch._execute_subtask = _capture

        burst = _fake_burst()
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute(instruction)

        assert captured_subtask[0].description == instruction


# ── tests: detect_burst_from_instruction patterns ────────────────────────────

class TestDetectBurstFromInstruction:

    def test_full_folder_creation_instruction_detected(self):
        """The canonical test instruction matches the 5-step shortcut-key folder burst."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, "
            "type TestFolder, press Enter."
        )
        assert burst is not None, "Should detect burst for full folder creation instruction"
        assert len(burst.steps) == 5

    def test_full_folder_creation_step_types(self):
        """5-step visual burst has the expected action types in order."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, type MyDir, press Enter."
        )
        types = [s.action_type for s in burst.steps]
        assert types == ["right_click", "click", "click", "type", "key_press"]

    def test_full_folder_creation_captures_name(self):
        """The type step carries the exact folder name from the instruction."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, type Projects, press Enter."
        )
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Projects"

    def test_create_folder_called_detected(self):
        """'create a folder called X' matches pattern 1c → 5-step shortcut-key burst."""
        burst = detect_burst_from_instruction("create a folder called Reports")
        assert burst is not None
        assert len(burst.steps) == 5
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Reports"

    def test_create_new_folder_named_detected(self):
        """'new folder named X' matches pattern 1c."""
        burst = detect_burst_from_instruction("new folder named Archive")
        assert burst is not None
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Archive"

    def test_unrelated_instruction_returns_none(self):
        """An instruction that has nothing to do with burst patterns returns None."""
        burst = detect_burst_from_instruction("open Firefox and navigate to gmail.com")
        assert burst is None

    def test_short_new_folder_keywords_still_match_pattern1a(self):
        """'new folder' keyword still fires (5-step shortcut-key burst with default name)."""
        burst = detect_burst_from_instruction("create new folder on the desktop")
        assert burst is not None
        assert len(burst.steps) == 5


# ── tests: _execute_subtask uses subtask.burst directly ──────────────────────

class TestExecuteSubtaskUsesBurstField:

    def _make_subtask_orch(self):
        """Build an orchestrator where all collaborators are stubbed for burst execution."""
        from agents.grounding.grounding_agent import GroundingResult

        orch = TaskOrchestrator(
            router=MagicMock(),
            planner=MagicMock(),
            grounder=MagicMock(),
            actor=MagicMock(),
            reflector=MagicMock(),
            capturer=MagicMock(),
            task_memory=MagicMock(),
            config=OrchestratorConfig(max_steps_per_subtask=5),
            on_step_log=lambda _: None,
            ocr=MagicMock(),
        )
        gr = GroundingResult(
            found=True, confidence=0.9, x=500, y=400,
            latency_ms=5.0, target="desktop", element_type="foreground_interactive",
        )
        orch.grounder.ground = MagicMock(return_value=gr)
        orch.grounder.min_confidence = 0.5
        orch.actor.execute = MagicMock(return_value=True)
        orch.reflector.verify = MagicMock(return_value=MagicMock(
            success=True, confidence=0.95, ocr_text="", error_description="",
            should_retry=False,
        ))
        orch.memory.get_failure_hints = MagicMock(return_value=[])
        orch._get_screen_context = MagicMock(return_value='"desktop"')
        return orch

    def test_subtask_with_burst_runs_burst_not_planning_loop(self):
        """
        When subtask.burst is set, _execute_subtask must use the burst executor
        and NOT call planner.plan_next_step.
        """
        orch = self._make_subtask_orch()
        burst = _fake_burst()
        subtask = SubTask(id=1, description="folder task", depends_on=[], burst=burst)

        with patch.object(orch.burst_executor, 'run',
                          return_value=MagicMock(success=True, failed_at_step=None, reason="ok")) \
             as mock_burst_run:
            result = orch._execute_subtask(subtask)

        assert result is True
        mock_burst_run.assert_called_once_with(burst)
        orch.planner.plan_next_step.assert_not_called()

    def test_subtask_without_burst_falls_back_to_planning_loop(self):
        """
        When subtask.burst is None and detect_burst returns None,
        _execute_subtask must fall through to the LLM planning loop.
        """
        orch = self._make_subtask_orch()
        subtask = SubTask(id=1, description="click the OK button", depends_on=[])
        # Planner returns one step then done
        from core.protocols.a2a import ActionStep
        step = ActionStep(
            id=1, subtask_id=1, action_type="key_press",
            target=None, value=None, key="escape",
            description="Press Escape", verification="dialog dismissed",
        )
        orch.planner.plan_next_step = MagicMock(side_effect=[step, None])

        with patch("core.orchestrator.detect_burst", return_value=None):
            result = orch._execute_subtask(subtask)

        assert result is True
        orch.planner.plan_next_step.assert_called()
