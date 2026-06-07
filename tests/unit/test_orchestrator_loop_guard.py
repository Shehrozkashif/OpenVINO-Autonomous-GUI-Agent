# tests/unit/test_orchestrator_loop_guard.py
"""
Unit tests for action-type-aware step deduplication (loop guard) in the orchestrator.

The loop guard fires when _same_step_streak > DEDUP_LIMIT_BY_ACTION_TYPE[action_type].
Streak starts at 0 on the first success of a given signature, increments on each
consecutive repeat. Trigger semantics (limit N = allow N repeats):

  type        limit=1 → fires on 3rd total success (2nd repeat)
  click       limit=2 → fires on 4th total success (3rd repeat)
  right_click limit=1 → fires on 3rd total success (2nd repeat)
  key_press   limit=3 → fires on 5th total success (4th repeat)

Tests verify:
  1. Each allowed boundary does NOT fire early.
  2. One step beyond the boundary DOES fire.
  3. A different action type resets the streak so limits start fresh.
  4. For click/right_click the guard injects an Escape step before returning.
  5. The returned value is always True (declares goal achieved, not failure).
"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock

from core.orchestrator import TaskOrchestrator, OrchestratorConfig, DEDUP_LIMIT_BY_ACTION_TYPE
from core.protocols.a2a import ActionStep, SubTask
from agents.reflection.reflection_agent import ReflectionResult


# ── helpers ───────────────────────────────────────────────────────────────────

def _subtask(desc="do something"):
    return SubTask(id=1, description=desc, depends_on=[])


def _step(action_type="click", target="Button", key=None, value=None):
    return ActionStep(
        id=1, subtask_id=1,
        action_type=action_type,
        target=target, value=value, key=key,
        description=f"{action_type} {target or key or value}",
        verification="expected result",
    )


_SUCCESS = ReflectionResult(
    success=True, confidence=0.9, observation="ok",
    error_description="", should_retry=False, recovery_hint="", ocr_text="",
)


def _make_orch(plan_steps):
    """
    Build an orchestrator where:
    - planner returns plan_steps in order, then None (goal achieved).
    - every reflection returns success.
    - actor.execute always returns True.
    - grounding always finds the target.
    """
    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(return_value=_SUCCESS)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    from agents.grounding.grounding_agent import GroundingResult
    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(
        return_value=GroundingResult(
            found=True, confidence=0.9, x=100, y=200,
            latency_ms=5.0, target="Button",
            element_type="foreground_interactive",
        )
    )

    planner = MagicMock()
    planner.plan_next_step = MagicMock(side_effect=list(plan_steps) + [None])

    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=planner,
        grounder=grounder,
        actor=actor,
        reflector=reflector,
        capturer=MagicMock(),
        task_memory=memory,
        config=OrchestratorConfig(
            max_retries_per_step=1,
            max_steps_per_subtask=25,
            consecutive_failures_limit=10,
        ),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"Button"')
    return orch


# ── 1. Sanity check: DEDUP_LIMIT_BY_ACTION_TYPE values ───────────────────────

class TestDedupLimitValues:
    """The dict must contain the exact values specified in the design document."""

    def test_type_limit_is_1(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["type"] == 1

    def test_click_limit_is_2(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["click"] == 2

    def test_right_click_limit_is_1(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["right_click"] == 1

    def test_key_press_limit_is_3(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["key_press"] == 3


# ── 2. type (limit=1): 1 repeat allowed, 2nd repeat triggers ─────────────────

class TestTypeDedupLimit:
    """
    type limit=1 → streak 0→1: allowed; streak 1→2: trigger.
    Total appearances: 2 = allowed, 3 = triggers.
    """

    def test_type_appears_twice_is_allowed(self):
        """
        type_step × 2, then a different step, then None.
        Streak after 2nd type = 1 (1 > 1 is False) → no trigger.
        Planner is called 4 times (2 type + 1 diff + None).
        """
        ts = _step("type", target=None, value="hello")
        ds = _step("key_press", key="enter")
        orch = _make_orch([ts, ts, ds])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 4  # all 3 steps + None

    def test_type_appears_three_times_triggers(self):
        """
        type_step × 3 → streak reaches 2 (2 > 1) → loop guard fires after step 3.
        Planner is called exactly 3 times (4th call never happens).
        """
        ts = _step("type", target=None, value="hello")
        orch = _make_orch([ts, ts, ts])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 3

    def test_type_trigger_returns_true(self):
        """Loop guard always returns True (declares goal achieved, not failure)."""
        ts = _step("type", target=None, value="x")
        orch = _make_orch([ts, ts, ts])
        assert orch._execute_subtask(_subtask()) is True


# ── 3. click (limit=2): 2 repeats allowed, 3rd repeat triggers ───────────────

class TestClickDedupLimit:
    """
    click limit=2 → streak 0→1→2: allowed; streak 2→3: trigger.
    Total appearances: 3 = allowed, 4 = triggers.
    """

    def test_click_appears_three_times_is_allowed(self):
        """
        click × 3, then different, then None.
        Streak after 3rd click = 2 (2 > 2 is False) → no trigger.
        Planner called 5 times.
        """
        cs = _step("click", target="Btn")
        ds = _step("key_press", key="escape")
        orch = _make_orch([cs, cs, cs, ds])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 5

    def test_click_appears_four_times_triggers(self):
        """
        click × 4 → streak = 3 (3 > 2) → loop guard fires after step 4.
        Planner called exactly 4 times.
        """
        cs = _step("click", target="Btn")
        orch = _make_orch([cs, cs, cs, cs])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 4

    def test_click_trigger_injects_escape(self):
        """
        When the click loop guard fires, actor.execute must be called with an
        Escape key_press step (in addition to the 4 regular click executions).
        """
        cs = _step("click", target="Btn")
        orch = _make_orch([cs, cs, cs, cs])
        orch._execute_subtask(_subtask())
        # 4 click executions + 1 escape injection = 5 actor.execute calls
        assert orch.actor.execute.call_count == 5
        # Verify the last call was the escape step
        last_call_step = orch.actor.execute.call_args_list[-1][0][0]
        assert last_call_step.action_type == "key_press"
        assert last_call_step.key == "escape"


# ── 4. right_click (limit=1): same threshold as type ─────────────────────────

class TestRightClickDedupLimit:

    def test_right_click_appears_twice_is_allowed(self):
        rc = _step("right_click", target="Desktop")
        ds = _step("key_press", key="escape")
        orch = _make_orch([rc, rc, ds])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 4

    def test_right_click_appears_three_times_triggers(self):
        rc = _step("right_click", target="Desktop")
        orch = _make_orch([rc, rc, rc])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 3

    def test_right_click_trigger_injects_escape(self):
        rc = _step("right_click", target="Desktop")
        orch = _make_orch([rc, rc, rc])
        orch._execute_subtask(_subtask())
        # 3 right_click executions + 1 escape = 4 actor calls
        assert orch.actor.execute.call_count == 4
        last_step = orch.actor.execute.call_args_list[-1][0][0]
        assert last_step.action_type == "key_press"
        assert last_step.key == "escape"


# ── 5. Streak reset when action type changes ──────────────────────────────────

class TestStreakReset:
    """A different action signature resets the streak; subsequent runs start fresh."""

    def test_type_streak_resets_after_click(self):
        """
        type × 2 → streak = 1 (at limit, but NOT >1 so not triggered).
        click (different) → streak resets to 0.
        type × 2 → streak = 1 again (fresh start, NOT triggered).
        None → done normally.
        Planner called 6 times (5 steps + None).
        """
        ts = _step("type", target=None, value="hello")
        cs = _step("click", target="Foo")
        orch = _make_orch([ts, ts, cs, ts, ts])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 6

    def test_click_streak_resets_after_type(self):
        """
        click × 3 → streak = 2 (at limit, NOT triggered).
        type → streak resets.
        click × 3 → streak = 2 again (NOT triggered).
        """
        cs = _step("click", target="Btn")
        ts = _step("type", target=None, value="x")
        orch = _make_orch([cs, cs, cs, ts, cs, cs, cs])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 8  # 7 steps + None

    def test_different_target_same_type_resets_streak(self):
        """
        Two clicks on the same target, then a click on a different target,
        then two more clicks on the original target — streak for the original
        resets after the different target, so no trigger.
        """
        ca = _step("click", target="Alpha")
        cb = _step("click", target="Beta")
        orch = _make_orch([ca, ca, cb, ca, ca])
        result = orch._execute_subtask(_subtask())
        assert result is True
        assert orch.planner.plan_next_step.call_count == 6

    def test_only_type_no_escape_on_trigger(self):
        """type loop guard does NOT inject Escape (only click/right_click do)."""
        ts = _step("type", target=None, value="hello")
        orch = _make_orch([ts, ts, ts])
        orch._execute_subtask(_subtask())
        # Exactly 3 type executions, no escape step
        assert orch.actor.execute.call_count == 3
        for call in orch.actor.execute.call_args_list:
            assert call[0][0].action_type == "type"
