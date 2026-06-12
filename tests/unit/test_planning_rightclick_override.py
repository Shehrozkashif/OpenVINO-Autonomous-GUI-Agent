# tests/unit/test_planning_rightclick_override.py
"""
Unit tests for right_click override — PlanningAgent forces action_type="right_click"
when the subtask description explicitly says "right click" (even if the LLM outputs "click").

Covers:
  - LLM returns "click" but subtask says "right click on X" → overridden to "right_click"
  - Hyphenated form "right-click on X" → overridden
  - Subtask description that merely mentions "right click" inside a longer sentence
    but does NOT start with "right click" and has no "right click on" → NOT overridden
  - LLM already outputs "right_click" → unchanged
  - Non-click action types (key_press, type) → NOT changed
  - Normal "click" subtask with no "right click" phrase → unchanged
"""
import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock

from core.protocols.a2a import SubTask
from agents.planning.planning_agent import PlanningAgent


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── tests ─────────────────────────────────────────────────────────────────────

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
        """
        'right click' mentioned mid-sentence without 'right click on' and description
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
