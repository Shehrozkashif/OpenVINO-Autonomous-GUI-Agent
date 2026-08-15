# tests/unit/test_replanning.py
"""Tests for the long-task mechanisms in TaskOrchestrator and RouterAgent:

  1. Adaptive wall-clock budgets (_effective_task_deadline)
  2. Task-level replanning on subtask failure (router.replan + queue swap)
  3. Missing-parameter elicitation (on_ask hook enriches the instruction)

Live failure these guard against: a long meeting-scheduling task dies at the
flat 600 s cap, or one failed subtask throws away 15 minutes of correct
progress, or the agent invents a meeting time the user never gave.
"""
import json
import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from agents.router import RouterAgent
from core.orchestrator import TaskOrchestrator
from core.runstate import OrchestratorConfig
from core.types import SubTask
from tests.unit.conftest import make_history


def _make_orch(config: OrchestratorConfig | None = None) -> TaskOrchestrator:
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(),
        grounder=MagicMock(),
        actor=MagicMock(),
        reflector=MagicMock(),
        capturer=MagicMock(),
        history=make_history(),
        config=config or OrchestratorConfig(),
        on_step_log=lambda _: None,
        ocr=MagicMock(),
    )
    orch.router.summarize_completion = MagicMock(return_value="done")
    orch._get_screen_context = MagicMock(return_value='"desktop"')
    orch.truth.verify_launch = MagicMock(return_value=True)
    orch.truth.wait_for_settle = MagicMock()
    return orch


def _no_burst():
    # Historical guard: the orchestrator once had a burst fast path that could
    # skip the router. That path is gone (the router always runs now), so this
    # is a no-op context kept only so the call sites below read unchanged.
    import contextlib
    return contextlib.nullcontext()


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
        # First subtask fails; recovery subtasks AND the preserved queued
        # subtask ("schedule the meeting") all succeed.
        orch._execute_subtask = MagicMock(side_effect=[False, True, True, True])

        with _no_burst():
            result = orch.execute("schedule a zoom meeting")

        assert result["success"] is True
        orch.router.replan.assert_called_once()
        args, kwargs = orch.router.replan.call_args
        assert args[0] == "schedule a zoom meeting"
        assert args[1] == []                      # nothing completed yet
        assert args[2] == "open Zoom"             # the failed subtask
        # The still-queued downstream work is named so the router excludes it.
        assert kwargs["pending_descs"] == ["schedule the meeting"]

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
        orch._execute_subtask = MagicMock(side_effect=[False, True, True, True])

        with _no_burst():
            result = orch.execute("do the thing")

        # Watermark is 2 (ids 1 and 2 already used) → recovery ids become 3, 4;
        # the preserved queued subtask "b" is renumbered above them (5).
        assert result["subtasks_completed"] == [3, 4, 5]

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

        stored = orch.history.store_successful_task.call_args[0][1]
        assert [s.description for s in stored] == ["a via other route"]


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

    def test_history_stores_what_the_user_typed_not_the_answers(self):
        """The agent runs on the enriched text; history keeps the original.

        Instruction is the primary key of the task table, so filing a run under
        its own answers gives every run its own row — the same task appears
        several times in Recent Automations, and replaying one from a chip feeds
        last time's answers ("time zone -> PST") back into the router.
        """
        orch = _make_orch()
        orch.on_ask = MagicMock(return_value="PST")
        orch.router.missing_parameters = MagicMock(return_value=["Time zone?"])
        orch.router.decompose = MagicMock(
            return_value=("t1", [SubTask(id=1, description="a", depends_on=[])])
        )
        orch._execute_subtask = MagicMock(return_value=True)
        with _no_burst():
            orch.execute("schedule a zoom meeting")

        # the agent still acted on the enriched instruction
        assert "PST" in orch.router.decompose.call_args[0][0]
        # but what got remembered is what the user actually asked for
        stored = orch.history.store_successful_task.call_args[0][0]
        assert stored == "schedule a zoom meeting"

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


class TestReplanPreservesDownstream:
    """Regression (live AI-PC 08:20 run): the form was filled perfectly, then
    a replan re-derived 'remaining work' and emitted only the fill subtask —
    'click Save and send the invitation' vanished and the run declared success
    with the form open and unsaved. Queued downstream subtasks must survive a
    replan verbatim; only the failed subtask's work is rewritten.
    """

    def test_downstream_subtasks_survive_replan(self):
        orch = _make_orch()
        orch._clickable_controls_block = MagicMock(return_value="")
        orch.router.replan = MagicMock(return_value=[
            SubTask(id=1, description="fill the form via set_value", depends_on=[]),
        ])
        failed = SubTask(id=3, description="fill the form", depends_on=[2])
        downstream = [
            SubTask(id=4, description="click Save and send the invitation",
                    depends_on=[3]),
        ]
        executed = [
            SubTask(id=1, description="open Teams", depends_on=[]),
            SubTask(id=2, description="open the form", depends_on=[1]),
        ]
        out = orch._replan_remaining(
            "schedule a meeting", ["open Teams", "open the form"],
            failed, executed, downstream,
        )
        assert [s.description for s in out] == [
            "fill the form via set_value",
            "click Save and send the invitation",
        ]
        # Save depends on the rewrite and keeps a higher id (ID-order safe).
        assert out[1].depends_on == [out[0].id]
        assert out[1].id > out[0].id

    def test_empty_downstream_behaves_as_before(self):
        orch = _make_orch()
        orch._clickable_controls_block = MagicMock(return_value="")
        orch.router.replan = MagicMock(return_value=[
            SubTask(id=1, description="add attendee", depends_on=[]),
            SubTask(id=2, description="click Save", depends_on=[1]),
        ])
        failed = SubTask(id=4, description="click Save", depends_on=[3])
        out = orch._replan_remaining("instr", [], failed, [], [])
        assert [s.description for s in out] == ["add attendee", "click Save"]

    def test_router_prompt_names_queued_work_as_off_limits(self):
        client = MagicMock()
        client.query_llm = MagicMock(return_value=MagicMock(
            content='[{"id":1,"description":"fix the fill","depends_on":[]}]'
        ))
        router = RouterAgent(client)
        router.replan(
            "schedule a meeting", ["open Teams"], "fill the form",
            pending_descs=["click Save and send the invitation"],
        )
        prompt = client.query_llm.call_args[0][0][1]["content"]
        assert "click Save and send the invitation" in prompt
        assert "do NOT include them" in prompt


class TestSkipExhaustedSubtask:
    """Regression (live 14:12 run): the GST time-zone hunt burned the whole
    replan budget and the task died with attendees + Save still queued — the
    filled form was thrown away. A permanently-failed subtask with work
    queued behind it is skipped (degraded, named in the summary); a failed
    LAST subtask still fails the task (skipping it would fake success).
    """

    def _orch(self):
        orch = _make_orch(OrchestratorConfig(max_task_replans=0))
        plan = [
            SubTask(id=1, description="set the time zone to GST", depends_on=[]),
            SubTask(id=2, description="click Save to create the meeting",
                    depends_on=[1]),
        ]
        orch.router.decompose = MagicMock(return_value=("t1", plan))
        return orch

    def test_failed_subtask_with_downstream_is_skipped(self):
        orch = self._orch()
        orch._execute_subtask = MagicMock(side_effect=[False, True])

        with _no_burst():
            result = orch.execute("schedule a meeting")

        assert result["success"] is True          # Save ran and succeeded
        assert result["subtasks_completed"] == [2]
        assert orch._degraded is True             # never stored as clean
        orch.history.store_successful_task.assert_not_called()
        # The summary is told exactly what was skipped.
        _, kwargs = orch.router.summarize_completion.call_args
        assert kwargs["skipped"] == ["set the time zone to GST"]

    def test_failed_last_subtask_still_fails_task(self):
        orch = self._orch()
        orch._execute_subtask = MagicMock(side_effect=[True, False])

        with _no_burst():
            result = orch.execute("schedule a meeting")

        assert result["success"] is False
