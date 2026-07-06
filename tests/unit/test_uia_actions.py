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


# ═══════════════════════════════════════════════════════════════════════════
# invoke_element pattern chain
#
# Live failures (AI-PC log 2026-07-05): WebView2-hosted Outlook buttons raised
# "An event was unable to invoke any of the subscribers" from InvokePattern,
# and the old fallback called GetSelectionItemPattern() — an attribute
# ButtonControl doesn't have — aborting the whole invoke. The chain must try
# each pattern independently and reach LegacyIAccessible.DoDefaultAction.
# ═══════════════════════════════════════════════════════════════════════════

from types import SimpleNamespace

import core.windows_uia as wu


class _FakePatternId:
    InvokePattern = 10000
    SelectionItemPattern = 10010
    TogglePattern = 10015
    LegacyIAccessiblePattern = 10018


def _fake_ctrl(control_type, patterns):
    """patterns: {pattern_id: pattern_object_or_None}"""
    ctrl = MagicMock()
    ctrl.ControlTypeName = control_type
    ctrl.GetPattern.side_effect = lambda pid: patterns.get(pid)
    return ctrl


def _run_invoke(ctrl):
    fake_uia = SimpleNamespace(PatternId=_FakePatternId)
    with patch.object(wu, "_load", return_value=True), \
         patch.object(wu, "_uia", fake_uia), \
         patch.object(wu, "_find_control", return_value=ctrl):
        return wu.invoke_element("Calendar", timeout_s=2.0)


class TestInvokePatternChain:

    def test_invoke_com_error_falls_through_to_do_default_action(self):
        invoke = MagicMock()
        invoke.Invoke.side_effect = OSError(
            "(-2147220991, 'An event was unable to invoke any of the subscribers')")
        legacy = MagicMock()
        ctrl = _fake_ctrl("ButtonControl", {
            _FakePatternId.InvokePattern: invoke,
            _FakePatternId.SelectionItemPattern: None,
            _FakePatternId.TogglePattern: None,
            _FakePatternId.LegacyIAccessiblePattern: legacy,
        })
        assert _run_invoke(ctrl) is True
        legacy.DoDefaultAction.assert_called_once()

    def test_unsupported_patterns_do_not_abort_chain(self):
        """GetPattern returning None (pattern unsupported) must be skipped."""
        legacy = MagicMock()
        ctrl = _fake_ctrl("ButtonControl", {
            _FakePatternId.LegacyIAccessiblePattern: legacy,
        })
        assert _run_invoke(ctrl) is True
        legacy.DoDefaultAction.assert_called_once()

    def test_invoke_success_stops_chain(self):
        invoke = MagicMock()
        legacy = MagicMock()
        ctrl = _fake_ctrl("ButtonControl", {
            _FakePatternId.InvokePattern: invoke,
            _FakePatternId.LegacyIAccessiblePattern: legacy,
        })
        assert _run_invoke(ctrl) is True
        invoke.Invoke.assert_called_once()
        legacy.DoDefaultAction.assert_not_called()

    def test_checkbox_prefers_toggle(self):
        toggle = MagicMock()
        invoke = MagicMock()
        ctrl = _fake_ctrl("CheckBoxControl", {
            _FakePatternId.TogglePattern: toggle,
            _FakePatternId.InvokePattern: invoke,
        })
        assert _run_invoke(ctrl) is True
        toggle.Toggle.assert_called_once()
        invoke.Invoke.assert_not_called()

    def test_all_patterns_fail_returns_false_for_click_fallback(self):
        ctrl = _fake_ctrl("ButtonControl", {})
        assert _run_invoke(ctrl) is False


# ═══════════════════════════════════════════════════════════════════════════
# set_value keyboard fallback (focus + type + read-back)
#
# Demo blocker (AI-PC 16:19 run): the Outlook event form's date/time fields
# expose no writable ValuePattern — set_element_value fails, and clicking the
# compound row is inert. Keyboard input still works: focus via the tree,
# ctrl+a, type, verify against the focused control's value.
# ═══════════════════════════════════════════════════════════════════════════

class TestSetValueKeyboardFallback:

    def _agent(self):
        agent = ActionExecutionAgent(controller=MagicMock())
        agent.controller.hotkey.return_value = True
        agent.controller.type_text.return_value = True
        return agent

    def _run(self, agent, focused_value="Tue 7/7/2026"):
        with patch("core.windows_uia.set_element_value", return_value=False), \
             patch("core.windows_uia.focus_element", return_value=True) as fe, \
             patch("core.windows_uia.focused_element_info",
                   return_value={"value": focused_value}), \
             patch("agents.action.time.sleep"):
            ok = agent.execute(_step("set_value", target="Start date",
                                     value="7/7/2026"))
        return ok, fe

    def test_focus_type_verified_by_reformatted_readback(self):
        """App reformats '7/7/2026' to 'Tue 7/7/2026' — still a verify."""
        agent = self._agent()
        ok, fe = self._run(agent)
        assert ok is True
        fe.assert_called_once_with("Start date")
        agent.controller.hotkey.assert_called_once_with("ctrl", "a")
        agent.controller.type_text.assert_called_once()

    def test_readback_mismatch_returns_false(self):
        agent = self._agent()
        ok, _ = self._run(agent, focused_value="1/1/2020")
        assert ok is False

    def test_empty_readback_returns_false(self):
        """Unreadable field ≠ verified — the step must not claim success."""
        agent = self._agent()
        ok, _ = self._run(agent, focused_value="")
        assert ok is False

    def test_focus_failure_skips_typing(self):
        agent = self._agent()
        with patch("core.windows_uia.set_element_value", return_value=False), \
             patch("core.windows_uia.focus_element", return_value=False):
            ok = agent.execute(_step("set_value", target="Start date",
                                     value="7/7/2026"))
        assert ok is False
        agent.controller.type_text.assert_not_called()

    def test_value_pattern_success_skips_fallback(self):
        agent = self._agent()
        with patch("core.windows_uia.set_element_value", return_value=True), \
             patch("core.windows_uia.focus_element") as fe:
            ok = agent.execute(_step("set_value", target="Add title",
                                     value="project progress"))
        assert ok is True
        fe.assert_not_called()


class TestTypeFocusesNamedTarget:
    """Regression (live 13:05 run): a type step's text landed in the Title
    field because the preceding click never moved focus. A type step that
    names its field must focus it through the tree before typing.
    """

    def _agent(self):
        from unittest.mock import MagicMock

        from agents.action import ActionExecutionAgent
        controller = MagicMock()
        controller.type_text = MagicMock(return_value=True)
        return ActionExecutionAgent(controller)

    def _step(self, target):
        from core.protocols import ActionStep
        return ActionStep(id=1, subtask_id=1, action_type="type",
                          target=target, value="a@b.com", key=None,
                          description="type email", verification="")

    def test_type_with_target_focuses_via_tree(self):
        from unittest import mock
        agent = self._agent()
        with mock.patch("core.windows_uia.focus_element",
                        return_value=True) as f:
            assert agent.execute(self._step("Add required attendees")) is True
        f.assert_called_once_with("Add required attendees")

    def test_type_without_target_skips_focus(self):
        from unittest import mock
        agent = self._agent()
        with mock.patch("core.windows_uia.focus_element") as f:
            assert agent.execute(self._step(None)) is True
        f.assert_not_called()

    def test_focus_failure_still_types(self):
        from unittest import mock
        agent = self._agent()
        with mock.patch("core.windows_uia.focus_element", return_value=False):
            assert agent.execute(self._step("Add required attendees")) is True
        agent.controller.type_text.assert_called_once()


class TestSelectValuePatternFallback:
    """Regression (live 13:36 run): Teams' 'Start time' combo materializes NO
    list items in the tree ('visible items: system') — select failed 15× over
    14 minutes while a direct ValuePattern write verifies instantly. When no
    option matches, select_option must fall back to writing the value.
    """

    class _FakeVP:
        IsReadOnly = False

        def __init__(self):
            self.Value = "2:00 PM"

        def SetValue(self, v):
            self.Value = v

    class _FakeCombo:
        ControlTypeName = "ComboBoxControl"
        Name = "Start time"

        def __init__(self, vp):
            self._vp = vp

        def GetValuePattern(self):
            return self._vp

        def GetExpandCollapsePattern(self):
            raise RuntimeError("no pattern")

        def GetChildren(self):
            return []

    def _run_select(self, vp):
        import core.windows_uia as wu
        combo = self._FakeCombo(vp)
        with patch.object(wu, "_load", return_value=True), \
             patch.object(wu, "_thread_com_init", return_value=None), \
             patch.object(wu, "_uia", MagicMock()), \
             patch.object(wu, "_find_control", return_value=combo):
            return wu.select_option("Start time", "3:00 PM")

    def test_no_items_falls_back_to_value_write(self):
        vp = self._FakeVP()
        assert self._run_select(vp) is True
        assert vp.Value == "3:00 PM"

    def test_readonly_value_still_fails_honestly(self):
        vp = self._FakeVP()
        vp.IsReadOnly = True
        assert self._run_select(vp) is False
        assert vp.Value == "2:00 PM"   # never written
