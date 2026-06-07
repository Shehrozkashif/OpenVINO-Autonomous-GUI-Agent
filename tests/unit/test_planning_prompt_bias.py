# tests/unit/test_planning_prompt_bias.py
"""
Unit tests for planning prompt safety — "when in doubt, stop" bias and LOOP PREVENTION rule.

Two properties are verified:

1. PROMPT BIAS    — the old "When in doubt, return the action step" instruction
                    is gone; the new instruction biases the planner toward
                    stopping (returning []) on uncertainty.

2. LOOP PREVENTION — the new LOOP PREVENTION rule is present in the message
                     sent to the LLM on every call to plan_next_step(), telling
                     the model not to repeat the immediately preceding step.

Tests inspect the actual messages[] passed to ovms.query_llm so they catch
regressions in both the prompt text and the call path.
"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List

from agents.planning.planning_agent import PlanningAgent
from core.protocols.a2a import SubTask


# ── helpers ───────────────────────────────────────────────────────────────────

def _subtask(desc="click a button"):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_agent(llm_response: str = "[]"):
    """PlanningAgent with a mocked LLM that records call args and returns llm_response."""
    ovms = MagicMock()
    ovms.query_llm = MagicMock(return_value=MagicMock(content=llm_response))
    return PlanningAgent(ovms_client=ovms), ovms


def _user_msg(ovms) -> str:
    """Extract the user-role message content from the last query_llm call."""
    call_args = ovms.query_llm.call_args
    messages: list = call_args[0][0]   # first positional arg
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    raise AssertionError("No user-role message found in query_llm call")


# ── 1. Prompt-bias tests ──────────────────────────────────────────────────────

class TestPromptBias:
    """The old 'act on doubt' instruction must be gone; the new 'stop on doubt' must be present."""

    def test_old_act_on_doubt_instruction_removed(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
        assert "when in doubt, return the action step" not in msg.lower(), (
            "Old 'When in doubt, return the action step rather than []' must be removed"
        )

    def test_old_act_phrase_absent_case_insensitive(self):
        """Guard against the phrase being re-introduced in any capitalisation."""
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
        assert "return the action step rather than" not in msg.lower()

    def test_new_stop_on_doubt_instruction_present(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
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
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask(), completed=[])
        msg = _user_msg(ovms)
        assert "return the action step rather than" not in msg.lower()

    def test_stop_on_doubt_applies_with_history(self):
        """Bias test holds when called with a non-empty completed list."""
        agent, ovms = _make_agent()
        agent.plan_next_step(
            _subtask(),
            completed=["click Desktop", "right_click Desktop"],
        )
        msg = _user_msg(ovms)
        assert "return the action step rather than" not in msg.lower()


# ── 2. Loop-prevention rule tests ─────────────────────────────────────────────

class TestLoopPrevention:
    """The LOOP PREVENTION rule must be in every user message sent to the LLM."""

    def test_loop_prevention_rule_present_no_history(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
        assert "loop prevention" in msg.lower(), (
            "LOOP PREVENTION rule must always be present in the user message"
        )

    def test_loop_prevention_rule_present_with_history(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(
            _subtask(),
            completed=["click New", "click New"],
        )
        msg = _user_msg(ovms)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_mentions_preceding_step(self):
        """The rule must reference the 'immediately preceding' step concept."""
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
        lower = msg.lower()
        assert "preceding" in lower or "immediately" in lower, (
            "LOOP PREVENTION rule must mention the immediately preceding step"
        )

    def test_loop_prevention_advises_next_action_or_stop(self):
        """The rule must tell the model to either advance or return []."""
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        msg = _user_msg(ovms)
        lower = msg.lower()
        pos = lower.find("loop prevention")
        snippet = lower[pos:pos + 250]
        assert "return []" in snippet or "next logical" in snippet or "in sequence" in snippet, (
            "LOOP PREVENTION rule must direct the model to advance or stop.\n"
            f"Got: {snippet!r}"
        )

    def test_loop_prevention_present_with_screen_context(self):
        """Rule must be present regardless of whether screen_context is provided."""
        agent, ovms = _make_agent()
        agent.plan_next_step(
            _subtask(),
            screen_context='"New Folder" "Rename" "Copy"',
        )
        msg = _user_msg(ovms)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_present_with_failure_hints(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(
            _subtask(),
            failure_hints=["clicking 'New' dismissed the submenu"],
        )
        msg = _user_msg(ovms)
        assert "loop prevention" in msg.lower()

    def test_loop_prevention_present_with_task_context(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(
            _subtask(),
            task_context=["Opened the context menu"],
        )
        msg = _user_msg(ovms)
        assert "loop prevention" in msg.lower()


# ── 3. Interaction — existing completion criteria must still be present ────────

class TestExistingCriteriaPreserved:
    """Fix 4.1 must not accidentally remove any of the existing completion rules."""

    def _msg(self):
        agent, ovms = _make_agent()
        agent.plan_next_step(_subtask())
        return _user_msg(ovms)

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
