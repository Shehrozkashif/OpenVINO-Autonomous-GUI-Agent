# tests/unit/test_orchestrator_idempotency.py
"""
Unit tests for Fix C5 — non-idempotent actions are not blind-retried.

A `type` step (and Enter/paste) changes state every time it runs. When such a
step physically executes but verification comes back uncertain or failed, the
orchestrator must NOT re-execute it (which would type the text twice). It hands
control back to the planner instead.

Contrast: an idempotent step (click) may still be retried on an uncertain result.
"""
import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock

from core.orchestrator import TaskOrchestrator, OrchestratorConfig
from core.protocols.a2a import ActionStep, SubTask
from agents.grounding.grounding_agent import GroundingResult
from agents.reflection.reflection_agent import ReflectionResult


_UNCERTAIN = ReflectionResult(
    success=False, confidence=0.40, observation="unclear",
    error_description="", should_retry=True, recovery_hint="", ocr_text="",
)


def _make_orch(plan_steps, reflection):
    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(return_value=reflection)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(return_value=GroundingResult(
        found=True, confidence=0.9, x=10, y=20, latency_ms=1.0,
        target="x", element_type="foreground_interactive"))

    planner = MagicMock()
    planner.plan_next_step = MagicMock(side_effect=list(plan_steps) + [None] * 10)

    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(), planner=planner, grounder=grounder, actor=actor,
        reflector=reflector, capturer=MagicMock(), task_memory=memory,
        config=OrchestratorConfig(max_retries_per_step=3, max_steps_per_subtask=1,
                                  consecutive_failures_limit=10),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value="")
    orch._firewall_allows = MagicMock(return_value=True)
    return orch, actor


def _type_step():
    return ActionStep(id=1, subtask_id=1, action_type="type", target=None,
                      value="hello", key=None, description="type hello",
                      verification="hello visible")


def _click_step():
    return ActionStep(id=1, subtask_id=1, action_type="click", target="Btn",
                      value=None, key=None, description="click Btn",
                      verification="changed")


def test_type_executes_once_on_uncertain():
    """An uncertain `type` verdict must not cause a second type within one step."""
    orch, actor = _make_orch([_type_step()], _UNCERTAIN)
    orch._execute_subtask(SubTask(id=1, description="do it", depends_on=[]))
    type_calls = [c for c in actor.execute.call_args_list
                  if c.args and c.args[0].action_type == "type"]
    assert len(type_calls) == 1, f"expected 1 type execution, got {len(type_calls)}"


def test_click_may_retry_on_uncertain():
    """An idempotent click is allowed to retry on an uncertain verdict."""
    orch, actor = _make_orch([_click_step()], _UNCERTAIN)
    orch._execute_subtask(SubTask(id=1, description="do it", depends_on=[]))
    click_calls = [c for c in actor.execute.call_args_list
                   if c.args and c.args[0].action_type == "click"]
    assert len(click_calls) >= 2, "click should retry at least once on uncertain"
