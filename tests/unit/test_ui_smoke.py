# tests/unit/test_ui_smoke.py
"""
End-to-end UI smoke test (offscreen).

Builds the full command-center window and pushes REAL orchestrator log lines
through the same signal path used at runtime:

    orchestrator.log(str) → WorkerSignals.log_update → AgentEventBus.feed()
        → MissionPage timeline / IntelligencePanel / status chip

verifying the UI is wired to the backend without needing OVMS running.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from ui.events import AgentEventBus, AgentState


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_live_screen_capture(monkeypatch):
    """UI tests must never grab the real screen. Live Xlib captures firing on
    the 1 s preview timer while Qt pumps events have caused native segfaults
    in full-suite runs (X11 + offscreen Qt in one process), and the preview
    path is already exercised deterministically via WorkerSignals."""
    from ui.main_window import DesktopGUIAgent
    monkeypatch.setattr(DesktopGUIAgent, "_refresh_screen", lambda self: None)


# Verbatim log lines as core/orchestrator.py emits them
ORCHESTRATOR_LOG = [
    "[TASK START] 'open notepad and type hello'",
    "[MEMORY] Similar past task found (sim=0.91)",
    "[ROUTER] 2 sub-task(s)",
    "\n[SUBTASK 1] Open Notepad",
    "  Step 1: [key_press] Open the search launcher",
    "  Verified (conf=0.95)",
    "  Step 2: [type] Type 'notepad' in the search box",
    "  Uncertain result — retrying (uncertain outcome (conf=0.50, threshold=0.95))",
    "  Retry 1/3…",
    "  Verified (conf=0.88)",
    "  [CHECK] 'notepad' process confirmed running",
    "[SUBTASK 1] Complete",
    "\n[SUBTASK 2] Type hello in Notepad",
    "  Step 1: [click] Click the Notepad text area",
    "  Verification failed: text area not focused (conf=0.97)",
    "  Step failed — re-evaluating next action",
    "  [VISUAL-REPLAN] Text planning stuck — asking VLM with screenshot",
    "  Step 2: [type] Type hello",
    "  [FIREWALL] MEDIUM risk detected: shell keyword",
    "  Verified (conf=0.92)",
    "  [EXTRACT] 'page_title' = 'Untitled - Notepad'",
    "[SUBTASK 2] Complete",
    "\n[TASK DONE] All subtasks completed (42.3s)",
]


def test_event_bus_parses_orchestrator_stream(app):
    bus = AgentEventBus()
    events = []
    bus.task_started.connect(lambda i: events.append(("start", i)))
    bus.plan_ready.connect(lambda n: events.append(("plan", n)))
    bus.subtask_started.connect(lambda s, d: events.append(("sub", s)))
    bus.subtask_finished.connect(lambda s, ok: events.append(("sub_done", s, ok)))
    bus.step_started.connect(lambda n, a, d: events.append(("step", a)))
    bus.step_verified.connect(lambda c: events.append(("ok", c)))
    bus.step_failed.connect(lambda r, c: events.append(("fail", r)))
    bus.retrying.connect(lambda a, t: events.append(("retry", a, t)))
    bus.guard_event.connect(lambda k, m: events.append(("guard", k)))
    bus.extracted.connect(lambda k, v: events.append(("extract", k, v)))
    bus.memory_hint.connect(lambda s: events.append(("memory", s)))
    bus.task_done.connect(lambda s, e: events.append(("done", e)))

    for line in ORCHESTRATOR_LOG:
        bus.feed(line)

    assert ("start", "open notepad and type hello") in events
    assert ("plan", 2) in events
    assert ("sub", 1) in events and ("sub", 2) in events
    assert ("sub_done", 1, True) in events and ("sub_done", 2, True) in events
    assert ("step", "key_press") in events and ("step", "type") in events
    assert ("ok", 0.95) in events and ("ok", 0.92) in events
    assert ("retry", 1, 3) in events
    assert any(e[0] == "fail" and "not focused" in e[1] for e in events)
    assert any(e[0] == "guard" and e[1] == "FIREWALL" for e in events)
    assert any(e[0] == "guard" and e[1] == "VISION" for e in events)
    assert ("extract", "page_title", "Untitled - Notepad") in events
    assert ("memory", 0.91) in events
    assert ("done", 42.3) in events
    assert bus.steps_total == 4
    assert bus.last_confidence == 0.92


# Verbatim loguru lines from a real run (agents/, tools/ — via LoguruBridge)
PIPELINE_LOG = [
    "[KILL-SWITCH] Armed — triple-Esc or top-left corner to stop",
    "[PLANNING] Next: [click] Click Microsoft Edge pinned button to open it",
    "[GROUNDING] 'Microsoft Edge pinned' → (1073,1050) conf=1.00 "
    "method=uia attempt=1 latency=1301ms",
    "[GROUNDING] 'Save button' → (200,300) conf=0.93 method=cache/ocr",
    "[GROUNDING] 'Submit' → (640,480) conf=0.88 method=rephrase/uia (as 'OK')",
    "[ACTION] click(left) @ (1073,1050)",
    "[REFLECTION] Uncertain click verdict (conf=0.50) — escalating to VLM "
    "screenshot check",
]


def test_event_bus_parses_pipeline_loguru_lines(app):
    bus = AgentEventBus()
    details, elements, guards, states = [], [], [], []
    bus.detail.connect(details.append)
    bus.element_located.connect(
        lambda t, x, y, c, m: elements.append((t, x, y, c, m)))
    bus.guard_event.connect(lambda k, m: guards.append(k))
    bus.state_changed.connect(lambda s: states.append(s))

    bus.feed("[TASK START] 'open brave browser'")
    bus.feed("  Step 1: [click] Click Microsoft Edge pinned button")
    for line in PIPELINE_LOG:
        bus.feed(line)

    assert AgentState.GROUNDING in states     # click step → locating first
    assert AgentState.ACTING in states        # element located → acting
    assert AgentState.VERIFYING in states     # [ACTION] fired → verifying
    assert ("Microsoft Edge pinned", 1073, 1050, 1.0, "uia") in elements
    # cache-hit and rephrase locates use compound method names
    assert ("Save button", 200, 300, 0.93, "cache/ocr") in elements
    assert ("Submit", 640, 480, 0.88, "rephrase/uia") in elements
    assert "KILL-SWITCH" in guards
    assert any("vision model" in d for d in details)   # VLM wait explained
    assert any("Located" in d for d in details)


def test_mission_hud_masks_itself_from_captures(app):
    from ui.hud import MissionHUD

    class FakeCapturer:
        persistent_exclude_regions = []

    bus = AgentEventBus()
    hud = MissionHUD(bus)
    cap = FakeCapturer()
    hud.attach_capturer(cap)
    try:
        hud.show_mission()
        app.processEvents()
        assert len(cap.persistent_exclude_regions) == 1, \
            "HUD must mask its own pixels out of agent captures"
        x1, y1, x2, y2 = cap.persistent_exclude_regions[0]
        assert x2 > x1 and y2 > y1

        # live state flows into the HUD
        bus.feed("[TASK START] 'demo'")
        bus.feed("  Step 1: [click] Click something")
        app.processEvents()
        assert hud.state_label.text() == "Locating element"

        hud.hide_mission()
        assert cap.persistent_exclude_regions == [], \
            "mask must be cleared when the HUD hides"
    finally:
        hud.close()


def test_grounding_overlay_marks_preview(app):
    """[GROUNDING] events must place a reticle on the Mission Control preview
    at the located element's normalized screen position."""
    from ui.main_window import DesktopGUIAgent

    win = DesktopGUIAgent(orchestrator=None)
    try:
        win.signals.log_update.emit("[TASK START] 'click save'")
        win.signals.log_update.emit("  Step 1: [click] Click the Save button")
        app.processEvents()
        preview = win.page_mission.preview
        assert preview._targets == []  # task start clears stale reticles

        sw, sh = win.page_mission._screen_wh()
        win.signals.log_update.emit(
            f"[GROUNDING] 'Save button' → ({sw // 2},{sh // 4}) "
            f"conf=0.93 method=cache/ocr")
        app.processEvents()

        assert len(preview._targets) == 1
        label, fx, fy, conf, _born = preview._targets[0]
        assert label == "Save button"
        assert abs(fx - (sw // 2) / sw) < 1e-6
        assert abs(fy - (sh // 4) / sh) < 1e-6
        assert conf == 0.93

        # markers are capped and a new mission wipes them
        for i in range(5):
            preview.mark_target(f"el{i}", 0.5, 0.5, 0.9)
        assert len(preview._targets) == 3
        win.signals.log_update.emit("[TASK START] 'next mission'")
        app.processEvents()
        assert preview._targets == []
    finally:
        win._screen_timer.stop()
        win.tray.hide()
        win.close()


def test_window_end_to_end_wiring(app):
    from ui.main_window import DesktopGUIAgent

    win = DesktopGUIAgent(orchestrator=None)
    try:
        # main.py contract
        assert hasattr(win, "instruction_input")
        win.instruction_input.setPlainText("demo task")
        assert win.instruction_input.toPlainText() == "demo task"
        assert callable(win._run_task)
        assert "Desktop GUI Agent" in win.windowTitle()  # masking contract

        # Drive the REAL runtime path: WorkerSignals → bus → widgets
        for line in ORCHESTRATOR_LOG:
            win.signals.log_update.emit(line)
        app.processEvents()

        # Timeline received both subtasks and their steps
        tl = win.page_mission.timeline
        assert set(tl._headers.keys()) == {1, 2}
        assert tl._current_step is not None

        # Intelligence panel reflects the mission
        assert "hello" in win.panel.objective.text().lower()
        assert win.panel.conf_value.text() == "92%"
        assert "TASK DONE" in win.panel.console.toPlainText()

        # Mission page stats updated
        assert win.page_mission.s_steps[1].text() == "4"

        # Completion path: worker result dict → UI reset
        win.signals.task_complete.emit({
            "task_id": "t", "success": True, "subtasks_completed": [1, 2],
            "elapsed_s": 42.3, "summary": "All subtasks completed",
            "extracted_data": {"page_title": "Untitled - Notepad"},
        })
        app.processEvents()
        assert win.bus.state == AgentState.COMPLETE
        assert win.dock.run_btn.isEnabled()

        # Pages construct and refresh without an orchestrator
        for idx in range(win.stack.count()):
            win._goto_page(idx)
            app.processEvents()
    finally:
        win._screen_timer.stop()
        win.tray.hide()
        win.close()


def test_run_task_requires_orchestrator(app, monkeypatch):
    """_run_task must fail safe (dialog, no crash) when OVMS is down."""
    from PyQt6.QtWidgets import QMessageBox

    from ui.main_window import DesktopGUIAgent

    win = DesktopGUIAgent(orchestrator=None)
    try:
        shown = []
        monkeypatch.setattr(QMessageBox, "critical",
                            staticmethod(lambda *a, **k: shown.append(a)))
        win.instruction_input.setPlainText("open notepad")
        win._run_task()
        assert shown, "expected an 'agent offline' dialog"
        assert win._running is False
    finally:
        win._screen_timer.stop()
        win.tray.hide()
        win.close()
