# tests/unit/test_replanning.py
"""Tests for the long-task mechanisms in TaskOrchestrator and RouterAgent:

  1. Adaptive wall-clock budgets (_effective_task_deadline)
  2. Task-level replanning on subtask failure (router.replan + queue swap)
  3. Checkpointing (save per subtask, clear on success, keep on failure)
  4. Missing-parameter elicitation (on_ask hook enriches the instruction)

Live failure these guard against: a long meeting-scheduling task dies at the
flat 600 s cap, or one failed subtask throws away 15 minutes of correct
progress, or the agent invents a meeting time the user never gave.
"""
import json
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

from agents.router import RouterAgent
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.protocols import SubTask


def _make_orch(config: OrchestratorConfig | None = None) -> TaskOrchestrator:
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(),
        grounder=MagicMock(),
        actor=MagicMock(),
        reflector=MagicMock(),
        capturer=MagicMock(),
        task_memory=MagicMock(),
        config=config or OrchestratorConfig(),
        on_step_log=lambda _: None,
        ocr=MagicMock(),
    )
    orch.router.summarize_completion = MagicMock(return_value="done")
    orch.memory.find_similar = MagicMock(return_value=None)
    orch.memory.load_checkpoint = MagicMock(return_value=None)
    orch._get_screen_context = MagicMock(return_value='"desktop"')
    orch._verify_launch = MagicMock(return_value=True)
    orch._wait_for_settle = MagicMock()
    return orch


def _no_burst():
    return patch("core.orchestrator.detect_burst_from_instruction", return_value=None)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Adaptive wall-clock budgets
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveDeadline:

    def test_floor_applies_for_small_plans(self):
        orch = _make_orch(OrchestratorConfig(task_deadline_s=600, subtask_deadline_s=240))
        assert orch._effective_task_deadline(1) == 600
        assert orch._effective_task_deadline(2) == 600

    def test_budget_scales_with_subtask_count(self):
        orch = _make_orch(OrchestratorConfig(task_deadline_s=600, subtask_deadline_s=240))
        assert orch._effective_task_deadline(5) == 1200
        assert orch._effective_task_deadline(8) == 1920

    def test_zero_task_deadline_disables_budget(self):
        orch = _make_orch(OrchestratorConfig(task_deadline_s=0, subtask_deadline_s=240))
        assert orch._effective_task_deadline(8) == 0.0

    def test_zero_subtask_deadline_keeps_flat_budget(self):
        orch = _make_orch(OrchestratorConfig(task_deadline_s=600, subtask_deadline_s=0))
        assert orch._effective_task_deadline(8) == 600


# ═══════════════════════════════════════════════════════════════════════════
# 2. Task-level replanning
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskReplanning:

    def test_failed_subtask_triggers_replan_and_task_succeeds(self):
        orch = _make_orch()
        plan_a = [
            SubTask(id=1, description="open Zoom", depends_on=[]),
            SubTask(id=2, description="schedule the meeting", depends_on=[1]),
        ]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        recovery = [
            SubTask(id=1, description="open Zoom via the search launcher", depends_on=[]),
            SubTask(id=2, description="schedule the meeting", depends_on=[1]),
        ]
        orch.router.replan = MagicMock(return_value=recovery)
        # First subtask fails; every recovery subtask succeeds.
        orch._execute_subtask = MagicMock(side_effect=[False, True, True])

        with _no_burst():
            result = orch.execute("schedule a zoom meeting")

        assert result["success"] is True
        orch.router.replan.assert_called_once()
        args, kwargs = orch.router.replan.call_args
        assert args[0] == "schedule a zoom meeting"
        assert args[1] == []                      # nothing completed yet
        assert args[2] == "open Zoom"             # the failed subtask

    def test_replanned_ids_are_renumbered_above_existing(self):
        orch = _make_orch()
        plan_a = [
            SubTask(id=1, description="a", depends_on=[]),
            SubTask(id=2, description="b", depends_on=[1]),
        ]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        recovery = [
            SubTask(id=1, description="a2", depends_on=[]),
            SubTask(id=2, description="b2", depends_on=[1]),
        ]
        orch.router.replan = MagicMock(return_value=recovery)
        orch._execute_subtask = MagicMock(side_effect=[False, True, True])

        with _no_burst():
            result = orch.execute("do the thing")

        # Watermark is 2 (ids 1 and 2 already used) → recovery ids become 3, 4.
        assert result["subtasks_completed"] == [3, 4]

    def test_completed_work_is_preserved_across_replan(self):
        orch = _make_orch()
        plan_a = [
            SubTask(id=1, description="open Zoom", depends_on=[]),
            SubTask(id=2, description="fill the form", depends_on=[1]),
        ]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        recovery = [SubTask(id=1, description="fill the form via keyboard", depends_on=[])]
        orch.router.replan = MagicMock(return_value=recovery)
        orch._execute_subtask = MagicMock(side_effect=[True, False, True])

        with _no_burst():
            result = orch.execute("schedule a zoom meeting")

        assert result["success"] is True
        # Replan sees what already completed, so it never repeats it.
        args, _ = orch.router.replan.call_args
        assert args[1] == ["open Zoom"]

    def test_replan_budget_is_capped(self):
        orch = _make_orch(OrchestratorConfig(max_task_replans=1))
        plan_a = [SubTask(id=1, description="a", depends_on=[])]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        recovery = [SubTask(id=1, description="a again", depends_on=[])]
        orch.router.replan = MagicMock(return_value=recovery)
        orch._execute_subtask = MagicMock(return_value=False)  # everything fails

        with _no_burst():
            result = orch.execute("do the thing")

        assert result["success"] is False
        assert orch.router.replan.call_count == 1   # capped, no infinite loop

    def test_empty_replan_fails_the_task(self):
        orch = _make_orch()
        plan_a = [SubTask(id=1, description="a", depends_on=[])]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        orch.router.replan = MagicMock(return_value=[])
        orch._execute_subtask = MagicMock(return_value=False)

        with _no_burst():
            result = orch.execute("do the thing")

        assert result["success"] is False

    def test_replanning_disabled_by_config(self):
        orch = _make_orch(OrchestratorConfig(max_task_replans=0))
        plan_a = [SubTask(id=1, description="a", depends_on=[])]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        orch._execute_subtask = MagicMock(return_value=False)

        with _no_burst():
            result = orch.execute("do the thing")

        assert result["success"] is False
        orch.router.replan.assert_not_called()

    def test_executed_subtasks_stored_in_memory_not_original_plan(self):
        # After a replan, memory must record the plan that actually worked.
        orch = _make_orch()
        plan_a = [SubTask(id=1, description="a", depends_on=[])]
        orch.router.decompose = MagicMock(return_value=("t1", plan_a))
        recovery = [SubTask(id=1, description="a via other route", depends_on=[])]
        orch.router.replan = MagicMock(return_value=recovery)
        orch._execute_subtask = MagicMock(side_effect=[False, True])

        with _no_burst():
            orch.execute("do the thing")

        stored = orch.memory.store_successful_task.call_args[0][1]
        assert [s.description for s in stored] == ["a via other route"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckpointing:

    def _two_subtask_orch(self):
        orch = _make_orch()
        plan = [
            SubTask(id=1, description="a", depends_on=[]),
            SubTask(id=2, description="b", depends_on=[1]),
        ]
        orch.router.decompose = MagicMock(return_value=("t1", plan))
        return orch

    def test_checkpoint_saved_after_each_completed_subtask(self):
        orch = self._two_subtask_orch()
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("do the thing")
        saved = [c.args[1] for c in orch.memory.save_checkpoint.call_args_list]
        assert saved == [["a"], ["a", "b"]]

    def test_checkpoint_cleared_on_success(self):
        orch = self._two_subtask_orch()
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("do the thing")
        orch.memory.clear_checkpoint.assert_called_once_with("do the thing")

    def test_checkpoint_kept_on_failure_for_resume(self):
        orch = self._two_subtask_orch()
        orch.config.max_task_replans = 0
        orch._execute_subtask = MagicMock(side_effect=[True, False])
        with _no_burst():
            result = orch.execute("do the thing")
        assert result["success"] is False
        orch.memory.clear_checkpoint.assert_not_called()

    def test_resume_hint_passed_to_router(self):
        orch = self._two_subtask_orch()
        orch.memory.load_checkpoint = MagicMock(return_value=["a"])
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("do the thing")
        _, kwargs = orch.router.decompose.call_args
        assert "do NOT repeat" in (kwargs.get("memory_hint") or "")
        assert "a" in kwargs["memory_hint"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Missing-parameter elicitation
# ═══════════════════════════════════════════════════════════════════════════

class TestElicitation:

    def test_answers_are_folded_into_the_instruction(self):
        orch = _make_orch()
        orch.on_ask = MagicMock(return_value="tomorrow at 3pm")
        orch.router.missing_parameters = MagicMock(
            return_value=["What date and time should the meeting be?"]
        )
        orch.router.decompose = MagicMock(
            return_value=("t1", [SubTask(id=1, description="a", depends_on=[])])
        )
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("schedule a zoom meeting")
        routed_instruction = orch.router.decompose.call_args[0][0]
        assert "details provided by the user" in routed_instruction
        assert "tomorrow at 3pm" in routed_instruction

    def test_no_handler_means_no_llm_check_and_no_change(self):
        orch = _make_orch()
        assert orch.on_ask is None
        orch.router.decompose = MagicMock(
            return_value=("t1", [SubTask(id=1, description="a", depends_on=[])])
        )
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("schedule a zoom meeting")
        orch.router.missing_parameters.assert_not_called()
        assert orch.router.decompose.call_args[0][0] == "schedule a zoom meeting"

    def test_declined_answer_leaves_instruction_unchanged(self):
        orch = _make_orch()
        orch.on_ask = MagicMock(return_value=None)   # user dismissed the dialog
        orch.router.missing_parameters = MagicMock(return_value=["When?"])
        orch.router.decompose = MagicMock(
            return_value=("t1", [SubTask(id=1, description="a", depends_on=[])])
        )
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("schedule a zoom meeting")
        assert orch.router.decompose.call_args[0][0] == "schedule a zoom meeting"


# ═══════════════════════════════════════════════════════════════════════════
# 5. RouterAgent.replan / missing_parameters parsing
# ═══════════════════════════════════════════════════════════════════════════

def _router_with_response(content: str) -> RouterAgent:
    client = MagicMock()
    client.query_llm = MagicMock(return_value=MagicMock(content=content))
    return RouterAgent(client)


class TestRouterReplan:

    def test_replan_parses_subtasks(self):
        router = _router_with_response(json.dumps([
            {"id": 1, "description": "open Zoom via search", "depends_on": []},
            {"id": 2, "description": "schedule the meeting", "depends_on": [1]},
        ]))
        subs = router.replan("schedule a zoom meeting", ["nothing"], "open Zoom")
        assert [s.description for s in subs] == [
            "open Zoom via search", "schedule the meeting",
        ]

    def test_replan_prompt_contains_completed_and_failed(self):
        router = _router_with_response("[]")
        router.replan("instr", ["step one done"], "the failed step")
        user_msg = router.client.query_llm.call_args[0][0][-1]["content"]
        assert "step one done" in user_msg
        assert "the failed step" in user_msg
        assert "DIFFERENT approach" in user_msg

    def test_replan_returns_empty_on_garbage(self):
        router = _router_with_response("no json here at all")
        assert router.replan("instr", [], "failed") == []


class TestMissingParameters:

    def test_non_trigger_instruction_skips_llm(self):
        router = _router_with_response('["should not be used"]')
        assert router.missing_parameters("open notepad and type hello") == []
        router.client.query_llm.assert_not_called()

    def test_trigger_instruction_returns_questions(self):
        router = _router_with_response(
            '["What date and time?", "Who should be invited?"]'
        )
        qs = router.missing_parameters("schedule a zoom meeting")
        assert qs == ["What date and time?", "Who should be invited?"]

    def test_questions_capped_at_three(self):
        router = _router_with_response('["a?", "b?", "c?", "d?", "e?"]')
        assert len(router.missing_parameters("schedule a meeting")) == 3

    def test_garbage_output_returns_empty(self):
        router = _router_with_response("I think you should provide a time")
        assert router.missing_parameters("schedule a meeting") == []

    def test_complete_instruction_with_empty_array(self):
        router = _router_with_response("[]")
        assert router.missing_parameters(
            "schedule a zoom meeting tomorrow 3pm titled Sync with bob@x.com"
        ) == []
