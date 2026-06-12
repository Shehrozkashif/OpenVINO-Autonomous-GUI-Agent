# tests/unit/test_planning_visual.py
"""
Unit tests for the visual planning recovery path and the planner parse-error fix.

1. _parse_visual_action — UI-TARS native action strings → ActionStep, with
   0-1000 coordinates scaled to screen pixels and carried in step.value ("x,y").
2. plan_next_step — a parse error is retried once, then raises
   PlanningParseError (never silently returns None / "goal achieved").
3. Orchestrator — escalates to plan_next_step_visual after
   visual_replan_after consecutive failures; explicit-coordinate click steps
   execute directly without grounding.
"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock

from agents.planning.planning_agent import (
    PlanningAgent,
    PlanningParseError,
    _parse_visual_action,
)
from agents.reflection.reflection_agent import ReflectionResult
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.protocols.a2a import ActionStep, SubTask


_W, _H = 1000, 1000   # identity scaling — 0-1000 coords map 1:1 to pixels


# ── 1. _parse_visual_action ───────────────────────────────────────────────────

class TestParseVisualAction:

    def test_click_bbox_scales_to_screen_center(self):
        step = _parse_visual_action(
            "click(start_box='[[100, 200, 300, 400]]')", 1, 1920, 1080)
        assert step.action_type == "click"
        x, y = (int(p) for p in step.value.split(","))
        assert x == int(200 / 1000 * 1920)
        assert y == int(300 / 1000 * 1080)

    def test_right_single_maps_to_right_click(self):
        step = _parse_visual_action(
            "right_single(start_box='[[500, 500, 500, 500]]')", 1, _W, _H)
        assert step.action_type == "right_click"
        assert step.value == "500,500"

    def test_left_double_maps_to_double_click(self):
        step = _parse_visual_action(
            "left_double(start_box='[[10, 20, 30, 40]]')", 1, _W, _H)
        assert step.action_type == "double_click"

    def test_click_point_form_two_values(self):
        step = _parse_visual_action("click(start_box='[[250, 750]]')", 1, _W, _H)
        assert step.action_type == "click"
        assert step.value == "250,750"

    def test_click_triple_bracket_point_form(self):
        """Regression: UI-TARS emits 'click(start_box='[[[287, 569]')' in practice."""
        step = _parse_visual_action("click(start_box='[[[287, 569]')", 1, _W, _H)
        assert step.action_type == "click"
        assert step.value == "287,569"

    def test_type_content(self):
        step = _parse_visual_action("type(content='hello world')", 1, _W, _H)
        assert step.action_type == "type"
        assert step.value == "hello world"

    def test_type_unescapes_quotes_and_newlines(self):
        step = _parse_visual_action(r"type(content='it\'s a line\ntwo')", 1, _W, _H)
        assert step.value == "it's a line\ntwo"

    def test_hotkey_space_separated_becomes_plus(self):
        step = _parse_visual_action("hotkey(key='ctrl s')", 1, _W, _H)
        assert step.action_type == "hotkey"
        assert step.key == "ctrl+s"

    def test_press_single_key(self):
        step = _parse_visual_action("press(key='enter')", 1, _W, _H)
        assert step.action_type == "key_press"
        assert step.key == "enter"

    def test_scroll_direction(self):
        step = _parse_visual_action("scroll(direction='down')", 1, _W, _H)
        assert step.action_type == "scroll"
        assert step.value == "down"

    def test_wait(self):
        step = _parse_visual_action("wait()", 1, _W, _H)
        assert step.action_type == "wait"

    def test_finished_returns_none(self):
        assert _parse_visual_action("finished()", 1, _W, _H) is None

    def test_think_tags_stripped(self):
        step = _parse_visual_action(
            "<think>reasoning here</think>\nclick(start_box='[[100,100,100,100]]')",
            1, _W, _H)
        assert step.action_type == "click"

    def test_garbage_raises_planning_parse_error(self):
        with pytest.raises(PlanningParseError):
            _parse_visual_action("I am not sure what to do next.", 1, _W, _H)


# ── 2. plan_next_step parse-error handling ────────────────────────────────────

def _resp(content):
    r = MagicMock()
    r.content = content
    return r


_VALID_STEP_JSON = (
    '[{"id":1,"action_type":"click","target":"OK","value":null,"key":null,'
    '"description":"click OK","verification":"dialog closes"}]'
)


class TestPlanNextStepParseError:

    def test_parse_error_retries_once_then_succeeds(self):
        client = MagicMock()
        client.query_llm = MagicMock(side_effect=[
            _resp("garbage with no json"),
            _resp(_VALID_STEP_JSON),
        ])
        agent = PlanningAgent(client)
        step = agent.plan_next_step(SubTask(id=1, description="click OK", depends_on=[]))
        assert step is not None
        assert step.action_type == "click"
        assert client.query_llm.call_count == 2

    def test_double_parse_error_raises_not_none(self):
        """Unparseable output twice must raise — returning None would be read
        by the orchestrator as 'goal achieved'."""
        client = MagicMock()
        client.query_llm = MagicMock(side_effect=[
            _resp("garbage"), _resp("more garbage"),
        ])
        agent = PlanningAgent(client)
        with pytest.raises(PlanningParseError):
            agent.plan_next_step(SubTask(id=1, description="click OK", depends_on=[]))

    def test_empty_array_still_means_goal_achieved(self):
        client = MagicMock()
        client.query_llm = MagicMock(return_value=_resp("[]"))
        agent = PlanningAgent(client)
        assert agent.plan_next_step(
            SubTask(id=1, description="click OK", depends_on=[])) is None


# ── 3. Orchestrator integration ───────────────────────────────────────────────

_FAIL = ReflectionResult(
    success=False, confidence=0.90, observation="no change",
    error_description="nothing happened", should_retry=True,
    recovery_hint="", ocr_text="",
)
_SUCCESS = ReflectionResult(
    success=True, confidence=0.95, observation="ok",
    error_description="", should_retry=False, recovery_hint="", ocr_text="",
)


def _click_step(target="Btn"):
    return ActionStep(id=1, subtask_id=1, action_type="click", target=target,
                      value=None, key=None, description=f"click {target}",
                      verification="changed")


def _make_orch(plan_steps, reflections, visual_replan_after=2):
    from agents.grounding.grounding_agent import GroundingResult

    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(side_effect=list(reflections) + [_SUCCESS] * 10)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(return_value=GroundingResult(
        found=True, confidence=0.9, x=10, y=20, latency_ms=1.0,
        target="Btn", element_type="foreground_interactive"))

    planner = MagicMock()
    planner.plan_next_step = MagicMock(side_effect=list(plan_steps) + [None] * 10)

    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(), planner=planner, grounder=grounder, actor=actor,
        reflector=reflector, capturer=MagicMock(), task_memory=memory,
        config=OrchestratorConfig(
            max_retries_per_step=1, max_steps_per_subtask=10,
            consecutive_failures_limit=5,
            visual_replan_after=visual_replan_after,
        ),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"Btn"')
    orch._firewall_allows = MagicMock(return_value=True)
    return orch


class TestVisualReplanEscalation:

    def test_visual_replan_fires_after_threshold_failures(self):
        """Two consecutive step failures → third planning call goes visual."""
        orch = _make_orch(
            plan_steps=[_click_step("A"), _click_step("B")],
            reflections=[_FAIL, _FAIL],
        )
        orch._plan_visual = MagicMock(return_value=None)   # VLM says finished()

        result = orch._execute_subtask(SubTask(id=1, description="do thing", depends_on=[]))

        assert result is True
        orch._plan_visual.assert_called_once()
        assert orch._degraded is True, "visual finished() must mark the run degraded"

    def test_visual_step_executes_with_explicit_coords(self):
        """A visual click step carries 'x,y' in value — executed without grounding."""
        visual_step = ActionStep(
            id=3, subtask_id=1, action_type="click", target=None,
            value="640,360", key=None, description="[visual] click at (640,360)",
            verification="")
        orch = _make_orch(
            plan_steps=[_click_step("A"), _click_step("B")],
            reflections=[_FAIL, _FAIL, _SUCCESS],
        )
        orch._plan_visual = MagicMock(side_effect=[visual_step, None])

        result = orch._execute_subtask(SubTask(id=1, description="do thing", depends_on=[]))

        assert result is True
        coords = [(c.kwargs.get("x"), c.kwargs.get("y"))
                  for c in orch.actor.execute.call_args_list
                  if c.args and c.args[0].action_type == "click"]
        assert (640, 360) in coords
        # Grounding must not have been asked to resolve the visual step
        grounded_targets = [c.args[0] for c in orch.grounder.ground.call_args_list]
        assert None not in grounded_targets

    def test_visual_disabled_when_threshold_zero(self):
        orch = _make_orch(
            plan_steps=[_click_step("A"), _click_step("B"), _click_step("C")],
            reflections=[_FAIL, _FAIL, _FAIL],
            visual_replan_after=0,
        )
        orch._plan_visual = MagicMock()
        orch._execute_subtask(SubTask(id=1, description="do thing", depends_on=[]))
        orch._plan_visual.assert_not_called()

    def test_vlm_infrastructure_error_falls_back_to_text_planner(self):
        """A VLM error during visual replan degrades to the text planner."""
        orch = _make_orch(
            plan_steps=[_click_step("A"), _click_step("B"), _click_step("C")],
            reflections=[_FAIL, _FAIL, _SUCCESS],
        )
        orch._plan_visual = MagicMock(side_effect=RuntimeError("model swap timeout"))

        result = orch._execute_subtask(SubTask(id=1, description="do thing", depends_on=[]))

        assert result is True
        # Text planner was used for the step after the VLM error
        assert orch.planner.plan_next_step.call_count >= 3


class TestPlannerParseErrorInOrchestrator:

    def test_parse_error_is_failure_not_goal_achieved(self):
        """PlanningParseError on every call → subtask FAILS (old code returned True)."""
        orch = _make_orch(plan_steps=[], reflections=[])
        orch.planner.plan_next_step = MagicMock(
            side_effect=PlanningParseError("unparseable"))
        # keep visual replan out of the way for this test
        orch.config.visual_replan_after = 0

        result = orch._execute_subtask(SubTask(id=1, description="do thing", depends_on=[]))

        assert result is False


class TestExplicitCoordParsing:

    def test_valid_coords(self):
        assert TaskOrchestrator._explicit_coords("123,456") == (123, 456)

    def test_coords_with_spaces(self):
        assert TaskOrchestrator._explicit_coords(" 12 , 34 ") == (12, 34)

    def test_text_value_is_not_coords(self):
        assert TaskOrchestrator._explicit_coords("hello,world") is None

    def test_none_and_plain_text(self):
        assert TaskOrchestrator._explicit_coords(None) is None
        assert TaskOrchestrator._explicit_coords("down") is None
