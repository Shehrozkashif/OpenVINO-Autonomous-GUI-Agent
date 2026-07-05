# tests/unit/test_uia_actions.py
"""Tests for the UIA structured-control actions (set_value / select / invoke).

Live failure these guard against: scheduling forms (Zoom, Teams, calendar
dialogs) are full of dropdowns, date pickers, and labelled fields where a
pixel click is a guess. The structured actions manipulate the real control
through the accessibility tree and self-verify by read-back.

The UIA layer itself only runs on Windows; here we verify the dispatch
plumbing — planner schema, action-agent routing, firewall coverage, and
orchestrator reflection/idempotency policy.
"""
import json
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

import pytest

from agents.action import ActionExecutionAgent
from agents.planning import PlanningAgent
from core.orchestrator import (
    DEDUP_LIMIT_BY_ACTION_TYPE,
    OrchestratorConfig,
    TaskOrchestrator,
)
from core.protocols import ActionStep, SubTask


def _step(action_type, target=None, value=None, key=None):
    return ActionStep(
        id=1, subtask_id=1, action_type=action_type,
        target=target, value=value, key=key,
        description=f"test {action_type}", verification="",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Action agent dispatch
# ═══════════════════════════════════════════════════════════════════════════

class TestActionAgentDispatch:

    def _agent(self):
        return ActionExecutionAgent(controller=MagicMock())

    def test_set_value_routes_to_uia(self):
        with patch("core.windows_uia.set_element_value", return_value=True) as sv:
            ok = self._agent().execute(_step("set_value", target="Topic", value="Weekly Sync"))
        assert ok is True
        sv.assert_called_once_with("Topic", "Weekly Sync")

    def test_select_routes_to_uia(self):
        with patch("core.windows_uia.select_option", return_value=True) as so:
            ok = self._agent().execute(_step("select", target="Start time", value="3:00 PM"))
        assert ok is True
        so.assert_called_once_with("Start time", "3:00 PM")

    def test_invoke_routes_to_uia(self):
        with patch("core.windows_uia.invoke_element", return_value=True) as inv:
            ok = self._agent().execute(_step("invoke", target="Save"))
        assert ok is True
        inv.assert_called_once_with("Save")

    def test_uia_miss_returns_false_for_planner_fallback(self):
        with patch("core.windows_uia.select_option", return_value=False):
            ok = self._agent().execute(_step("select", target="Nope", value="X"))
        assert ok is False

    def test_missing_fields_fail_without_calling_uia(self):
        with patch("core.windows_uia.set_element_value") as sv:
            assert self._agent().execute(_step("set_value", target="Topic")) is False
            assert self._agent().execute(_step("set_value", value="text")) is False
            assert self._agent().execute(_step("select", target="Time")) is False
            assert self._agent().execute(_step("invoke")) is False
        sv.assert_not_called()

    def test_set_value_substitutes_credentials(self):
        with patch("core.windows_uia.set_element_value", return_value=True) as sv, \
             patch("utils.credentials.has_tokens", return_value=True), \
             patch("utils.credentials.substitute", return_value="realsecret"):
            ok = self._agent().execute(
                _step("set_value", target="Password", value="{{cred:zoom:password}}")
            )
        assert ok is True
        sv.assert_called_once_with("Password", "realsecret")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Planner schema and validation
# ═══════════════════════════════════════════════════════════════════════════

def _planner_with(steps_json: list[dict]) -> PlanningAgent:
    client = MagicMock()
    client.query_llm = MagicMock(return_value=MagicMock(content=json.dumps(steps_json)))
    return PlanningAgent(client)


def _base(action_type, **over):
    d = {"id": 1, "action_type": action_type, "target": None, "value": None,
         "key": None, "description": "d", "verification": "v"}
    d.update(over)
    return d


class TestPlannerParsesStructuredActions:

    def test_set_value_and_select_parse(self):
        planner = _planner_with([
            _base("set_value", target="Topic", value="Weekly Sync"),
            _base("select", id=2, target="Start time", value="3:00 PM"),
            _base("invoke", id=3, target="Save"),
        ])
        steps = planner.plan_steps(SubTask(id=1, description="fill the form", depends_on=[]))
        assert [s.action_type for s in steps] == ["set_value", "select", "invoke"]

    def test_set_value_without_target_is_rejected(self):
        planner = PlanningAgent(MagicMock())
        with pytest.raises(ValueError):
            planner._parse_steps(json.dumps([_base("set_value", value="x")]), 1)

    def test_select_without_value_is_rejected(self):
        planner = PlanningAgent(MagicMock())
        with pytest.raises(ValueError):
            planner._parse_steps(json.dumps([_base("select", target="Time")]), 1)

    def test_invoke_without_target_is_rejected(self):
        planner = PlanningAgent(MagicMock())
        with pytest.raises(ValueError):
            planner._parse_steps(json.dumps([_base("invoke")]), 1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Orchestrator policy for structured actions
# ═══════════════════════════════════════════════════════════════════════════

def _orch():
    orch = TaskOrchestrator(
        router=MagicMock(), planner=MagicMock(), grounder=MagicMock(),
        actor=MagicMock(), reflector=MagicMock(), capturer=MagicMock(),
        task_memory=MagicMock(),
        config=OrchestratorConfig(max_steps_per_subtask=3),
        on_step_log=lambda _: None, ocr=MagicMock(),
    )
    orch._wait_for_settle = MagicMock()
    return orch


class TestOrchestratorPolicy:

    def test_set_value_goes_through_firewall(self):
        orch = _orch()
        step = _step("set_value", target="cmd box", value="rm -rf / --no-preserve-root")
        assert orch._execute_step(step) is False
        orch.actor.execute.assert_not_called()

    def test_benign_set_value_reaches_actor(self):
        orch = _orch()
        orch.actor.execute = MagicMock(return_value=True)
        step = _step("set_value", target="Topic", value="Weekly Sync")
        assert orch._execute_step(step) is True
        orch.actor.execute.assert_called_once()

    def test_set_value_skips_llm_reflection(self):
        # set_value verifies itself by UIA read-back — reflection would only
        # add a ~4 s LLM call per field.
        orch = _orch()
        orch.actor.execute = MagicMock(return_value=True)
        step = _step("set_value", target="Topic", value="Weekly Sync")
        orch.planner.plan_steps = MagicMock(side_effect=[[step], None])
        ok = orch._execute_subtask(SubTask(id=1, description="fill the form", depends_on=[]))
        assert ok is True
        orch.reflector.verify.assert_not_called()

    def test_invoke_is_non_idempotent_and_dedup_limited(self):
        # Pressing a real button twice double-submits — same class as Enter.
        assert DEDUP_LIMIT_BY_ACTION_TYPE["invoke"] == 1
        assert DEDUP_LIMIT_BY_ACTION_TYPE["set_value"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 4. windows_uia helpers that run on any OS
# ═══════════════════════════════════════════════════════════════════════════

class TestUiaHelpers:

    def test_norm_text_collapses_whitespace_and_case(self):
        from core.windows_uia import _norm_text
        assert _norm_text("  Weekly   Sync ") == "weekly sync"
        assert _norm_text(None) == ""

    def test_actions_are_safe_noops_without_uiautomation(self):
        # On machines without the uiautomation package (e.g. Linux CI) the
        # actions must return False, never raise.
        import core.windows_uia as wu
        with patch.object(wu, "_available", False), patch.object(wu, "_load", return_value=False):
            assert wu.set_element_value("Topic", "x") is False
            assert wu.select_option("Time", "3:00 PM") is False
            assert wu.invoke_element("Save") is False


class TestAppAnchor:
    """Task app anchor: ground truth for which window the task works in.

    Live failure this guards against: Outlook launched fine but a leftover
    Edge error page owned the foreground, so every subsequent action landed
    on Edge ('New Appointment' unfindable, 'New Tab' invoked instead).
    """

    def test_launch_subtask_sets_anchor_from_foreground(self):
        orch = _orch()
        orch._foreground_app = lambda: (1234, 42, "olk.exe")
        orch._maybe_set_app_anchor("open Outlook")
        assert orch._app_anchor == (1234, 42, "olk.exe")

    def test_launch_verbs_all_anchor(self):
        for verb in ("open", "launch", "start", "Open", "Launch"):
            orch = _orch()
            orch._foreground_app = lambda: (1, 2, "app.exe")
            orch._maybe_set_app_anchor(f"{verb} SomeApp")
            assert orch._app_anchor == (1, 2, "app.exe"), verb

    def test_non_launch_subtask_does_not_anchor(self):
        orch = _orch()
        orch._app_anchor = None
        orch._foreground_app = lambda: (1234, 42, "olk.exe")
        orch._maybe_set_app_anchor("with Outlook already open, click New Appointment")
        assert orch._app_anchor is None

    def test_already_running_phrase_does_not_anchor(self):
        orch = _orch()
        orch._app_anchor = None
        orch._foreground_app = lambda: (1234, 42, "olk.exe")
        orch._maybe_set_app_anchor("open Outlook which is already running")
        assert orch._app_anchor is None

    def test_own_window_never_anchors(self):
        orch = _orch()
        orch._app_anchor = None
        orch._own_hwnd = 777
        orch._foreground_app = lambda: (777, 42, "python.exe")
        orch._maybe_set_app_anchor("open Notepad")
        assert orch._app_anchor is None

    def test_refocus_without_anchor_is_noop(self):
        orch = _orch()
        orch._ensure_anchor_foreground()   # must not raise off-Windows
