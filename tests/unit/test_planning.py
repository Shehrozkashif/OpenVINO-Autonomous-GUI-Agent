# tests/unit/test_planning.py
"""Unit tests for agents/planning.py — PlanningAgent.

Organised by concern (each section was originally its own file):

  1. Planning prompt safety — "when in doubt, stop" bias and LOOP PREVENTION rule
  2. right_click override — forces action_type="right_click" when the subtask
     description explicitly says "right click" even if the LLM outputs "click"
  3. Visual planning recovery path and the planner parse-error fix
"""
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock

import pytest

from agents.planning import PlanningAgent, PlanningParseError, _parse_visual_action
from agents.reflection import ReflectionResult
from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.protocols import ActionStep, SubTask

# ═══════════════════════════════════════════════════════════════════════════
# 1. Planning prompt safety — "when in doubt, stop" bias and LOOP PREVENTION rule
#
# Two properties are verified:
#
# PROMPT BIAS    — the old "When in doubt, return the action step" instruction
#                 is gone; the new instruction biases the planner toward
#                 stopping (returning []) on uncertainty.
# LOOP PREVENTION — the new LOOP PREVENTION rule is present in the message
#                  sent to the LLM on every call to plan_next_step(), telling
#                  the model not to repeat the immediately preceding step.
#
# Tests inspect the actual messages[] passed to client.query_llm so they catch
# regressions in both the prompt text and the call path.
# ═══════════════════════════════════════════════════════════════════════════

def _subtask(desc="click a button"):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_agent(llm_response: str = "[]"):
    """PlanningAgent with a mocked LLM that records call args and returns llm_response."""
    client = MagicMock()
    client.query_llm = MagicMock(return_value=MagicMock(content=llm_response))
    return PlanningAgent(client=client), client


def _user_msg(client) -> str:
    """Extract the user-role message content from the last query_llm call."""
    call_args = client.query_llm.call_args
    messages: list = call_args[0][0]   # first positional arg
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    raise AssertionError("No user-role message found in query_llm call")


class TestPromptBias:
    """The old 'act on doubt' instruction must be gone; the new 'stop on doubt' must be present."""

    def test_old_act_on_doubt_instruction_removed(self):
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        assert "when in doubt, return the action step" not in msg.lower(), (
            "Old 'When in doubt, return the action step rather than []' must be removed"
        )

    def test_old_act_phrase_absent_case_insensitive(self):
        """Guard against the phrase being re-introduced in any capitalisation."""
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        assert "return the action step rather than" not in msg.lower()

    def test_new_stop_on_doubt_instruction_present(self):
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        assert "when in doubt" in msg.lower(), "A 'when in doubt' instruction must still exist"
        # The new instruction must direct toward stopping, not acting
        lower = msg.lower()
        # Find position of "when in doubt" and check that "return []" appears nearby
        pos = lower.find("when in doubt")
        snippet = lower[pos:pos + 120]
        assert "return []" in snippet or "speculative" in snippet, (
            "New 'when in doubt' instruction must direct toward stopping, not acting.\n"
            f"Got: {snippet!r}"
        )

    def test_stop_on_doubt_applies_without_history(self):
        """Bias test holds even when called with no completed steps."""
        agent, client = _make_agent()
        agent.plan_next_step(_subtask(), completed=[])
        msg = _user_msg(client)
        assert "return the action step rather than" not in msg.lower()

    def test_stop_on_doubt_applies_with_history(self):
        """Bias test holds when called with a non-empty completed list."""
        agent, client = _make_agent()
        agent.plan_next_step(
            _subtask(),
            completed=["click Desktop", "right_click Desktop"],
        )
        msg = _user_msg(client)
        assert "return the action step rather than" not in msg.lower()


class TestLoopPrevention:
    """The LOOP PREVENTION rule must be in every user message sent to the LLM."""

    def test_loop_prevention_rule_present_no_history(self):
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        assert "loop prevention" in msg.lower(), (
            "LOOP PREVENTION rule must always be present in the user message"
        )

    def test_loop_prevention_rule_present_with_history(self):
        agent, client = _make_agent()
        agent.plan_next_step(
            _subtask(),
            completed=["click New", "click New"],
        )
        msg = _user_msg(client)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_mentions_preceding_step(self):
        """The rule must reference the 'immediately preceding' step concept."""
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        lower = msg.lower()
        assert "preceding" in lower or "immediately" in lower, (
            "LOOP PREVENTION rule must mention the immediately preceding step"
        )

    def test_loop_prevention_advises_next_action_or_stop(self):
        """The rule must tell the model to either advance or return []."""
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(client)
        lower = msg.lower()
        pos = lower.find("loop prevention")
        snippet = lower[pos:pos + 250]
        assert "return []" in snippet or "next logical" in snippet or "in sequence" in snippet, (
            "LOOP PREVENTION rule must direct the model to advance or stop.\n"
            f"Got: {snippet!r}"
        )

    def test_loop_prevention_present_with_screen_context(self):
        """Rule must be present regardless of whether screen_context is provided."""
        agent, client = _make_agent()
        agent.plan_next_step(
            _subtask(),
            screen_context='"New Folder" "Rename" "Copy"',
        )
        msg = _user_msg(client)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_present_with_failure_hints(self):
        agent, client = _make_agent()
        agent.plan_next_step(
            _subtask(),
            failure_hints=["clicking 'New' dismissed the submenu"],
        )
        msg = _user_msg(client)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_present_with_task_context(self):
        agent, client = _make_agent()
        agent.plan_next_step(
            _subtask(),
            task_context=["Opened the context menu"],
        )
        msg = _user_msg(client)
        assert "loop prevention" in msg.lower()


class TestExistingCriteriaPreserved:
    """Fix 4.1 must not accidentally remove any of the existing completion rules."""

    def _msg(self):
        agent, client = _make_agent()
        agent.plan_next_step(_subtask())
        return _user_msg(client)

    def test_open_terminal_criterion_present(self):
        assert "open terminal" in self._msg().lower()

    def test_run_command_criterion_present(self):
        assert "run:" in self._msg().lower() or "key_press enter" in self._msg().lower()

    def test_open_browser_criterion_present(self):
        assert "open browser" in self._msg().lower()

    def test_click_menu_item_criterion_present(self):
        assert "submenu" in self._msg().lower()

    def test_type_and_enter_criterion_present(self):
        assert "type x" in self._msg().lower() or "key_press enter" in self._msg().lower()

    def test_caution_app_name_in_log_present(self):
        assert "caution" in self._msg().lower()


# ═══════════════════════════════════════════════════════════════════════════
# 2. right_click override
#
# PlanningAgent forces action_type="right_click" when the subtask description
# explicitly says "right click" (even if the LLM outputs "click"). Covers:
#   - LLM returns "click" but subtask says "right click on X" → overridden
#   - Hyphenated form "right-click on X" → overridden
#   - Subtask description that merely mentions "right click" inside a longer
#     sentence but does NOT start with "right click" and has no
#     "right click on" → NOT overridden
#   - LLM already outputs "right_click" → unchanged
#   - Non-click action types (key_press, type) → NOT changed
#   - Normal "click" subtask with no "right click" phrase → unchanged
# ═══════════════════════════════════════════════════════════════════════════

def _planner(action_type: str, target: str = "Desktop", key: str = None, value: str = None) -> PlanningAgent:
    """Build a PlanningAgent whose LLM always returns one step with the given action_type."""
    client = MagicMock()
    resp = MagicMock()
    t = f'"{target}"' if target else "null"
    k = f'"{key}"' if key else "null"
    v = f'"{value}"' if value else "null"
    resp.content = (
        f'[{{"id":1,"action_type":"{action_type}","target":{t},'
        f'"value":{v},"key":{k},"description":"test step","verification":"done"}}]'
    )
    client.query_llm = MagicMock(return_value=resp)
    return PlanningAgent(client)


def _sub(description: str) -> SubTask:
    return SubTask(id=1, description=description, depends_on=[])


class TestRightClickOverride:

    def test_right_click_on_desktop_overrides_click(self):
        """LLM returns 'click' but subtask says 'right click on the desktop' → right_click."""
        step = _planner("click").plan_next_step(
            _sub("right click on the desktop to open the context menu")
        )
        assert step is not None
        assert step.action_type == "right_click", (
            f"Expected right_click, got {step.action_type}"
        )

    def test_right_click_on_file_overrides_click(self):
        """'right click on' phrase in the middle of the description is caught."""
        step = _planner("click", target="myfile.txt").plan_next_step(
            _sub("right click on myfile.txt to rename it")
        )
        assert step is not None
        assert step.action_type == "right_click"

    def test_right_click_startswith_overrides_click(self):
        """Subtask starting with 'right click' (no 'on') is overridden."""
        step = _planner("click").plan_next_step(_sub("right click the taskbar icon"))
        assert step is not None
        assert step.action_type == "right_click"

    def test_right_click_hyphenated_on_overrides_click(self):
        """'right-click on X' with hyphen is overridden."""
        step = _planner("click").plan_next_step(_sub("right-click on the selected file"))
        assert step is not None
        assert step.action_type == "right_click"

    def test_right_click_hyphenated_startswith_overrides_click(self):
        """Subtask starting with 'right-click' (hyphen, no 'on') is overridden."""
        step = _planner("click").plan_next_step(_sub("right-click the desktop"))
        assert step is not None
        assert step.action_type == "right_click"

    def test_normal_click_subtask_not_overridden(self):
        """Regular 'click' subtask with no 'right click' phrase stays 'click'."""
        step = _planner("click", target="Start").plan_next_step(
            _sub("click on the Start menu button")
        )
        assert step is not None
        assert step.action_type == "click"

    def test_llm_already_returns_right_click_unchanged(self):
        """If the LLM already outputs 'right_click', the override leaves it as-is."""
        step = _planner("right_click").plan_next_step(_sub("right click on the desktop"))
        assert step is not None
        assert step.action_type == "right_click"

    def test_key_press_step_not_overridden_even_with_right_click_desc(self):
        """A key_press step is NOT changed to right_click even if subtask says 'right click'."""
        step = _planner("key_press", target=None, key="escape").plan_next_step(
            _sub("right click on the desktop to open the context menu")
        )
        assert step is not None
        assert step.action_type == "key_press"

    def test_type_step_not_overridden_even_with_right_click_desc(self):
        """A type step is NOT changed to right_click."""
        step = _planner("type", target=None, value="hello").plan_next_step(
            _sub("right click on the desktop to open the context menu")
        )
        assert step is not None
        assert step.action_type == "type"

    def test_mention_of_right_click_not_at_start_and_no_on_does_not_override(self):
        """'right click' mentioned mid-sentence without 'right click on' and description
        does not start with 'right click' → override does NOT fire.
        The check requires startswith OR 'right click on' to avoid over-eager matching.
        """
        step = _planner("click", target="Button").plan_next_step(
            _sub("after the right click context menu appears, click OK")
        )
        assert step is not None
        # This subtask does NOT start with "right click" and does NOT contain
        # "right click on" — override must not fire.
        assert step.action_type == "click"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Visual planning recovery path and the planner parse-error fix
#
#   1. _parse_visual_action — UI-TARS native action strings → ActionStep, with
#      0-1000 coordinates scaled to screen pixels and carried in step.value ("x,y").
#   2. plan_next_step — a parse error is retried once, then raises
#      PlanningParseError (never silently returns None / "goal achieved").
#   3. Orchestrator — escalates to plan_next_step_visual after
#      visual_replan_after consecutive failures; explicit-coordinate click steps
#      execute directly without grounding.
# ═══════════════════════════════════════════════════════════════════════════

_W, _H = 1000, 1000   # identity scaling — 0-1000 coords map 1:1 to pixels


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
        by the orchestrator as 'goal achieved'.
        """
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
    from agents.grounding import GroundingResult

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
