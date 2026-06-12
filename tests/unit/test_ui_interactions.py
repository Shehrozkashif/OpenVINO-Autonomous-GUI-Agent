# tests/unit/test_ui_interactions.py
"""
Click-through audit: every interactive control in the command center is
triggered programmatically and its effect asserted, so a broken
signal/slot connection fails CI instead of shipping as a dead button.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QMessageBox


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_live_screen_capture(monkeypatch):
    """UI tests must never grab the real screen (see test_ui_smoke.py)."""
    from ui.main_window import DesktopGUIAgent
    monkeypatch.setattr(DesktopGUIAgent, "_refresh_screen", lambda self: None)


@pytest.fixture()
def win(app):
    from ui.main_window import DesktopGUIAgent
    w = DesktopGUIAgent(orchestrator=None)
    yield w
    w._screen_timer.stop()
    w.tray.hide()
    w.close()


def test_nav_rail_buttons_switch_pages(win, app):
    for idx, item in enumerate(win.nav.items):
        item.click()
        app.processEvents()
        assert win.stack.currentIndex() == idx, f"nav item {idx} dead"
    assert win.page_title.text() == "Settings"


def test_suggestion_chips_fill_composer(win, app):
    win._goto_page(0)
    win.page_home.refresh()
    app.processEvents()
    # find a chip in the suggestion row and click it
    chips = []
    for i in range(win.page_home.sug_row.count()):
        w = win.page_home.sug_row.itemAt(i).widget()
        if w is not None:
            chips.append(w)
    assert chips, "no suggestion chips rendered"
    chips[0].click()
    app.processEvents()
    composer_text = win.page_home.composer.input.toPlainText()
    assert composer_text, "chip click did not fill the composer"
    # composer text mirrors into the persistent dock input
    assert win.instruction_input.toPlainText() == composer_text


def test_composer_run_button_triggers_run(win, app, monkeypatch):
    ran = []
    monkeypatch.setattr(win, "_run_task", lambda: ran.append(True))
    win.page_home.composer.set_text("open calculator")
    win.page_home.composer.run_btn.click()
    app.processEvents()
    assert ran, "composer Run Task button dead"
    assert win.instruction_input.toPlainText() == "open calculator"


def test_composer_enter_key_submits(win, app, monkeypatch):
    ran = []
    monkeypatch.setattr(win, "_run_task", lambda: ran.append(True))
    win.page_home.composer.set_text("open notepad")
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return,
                   Qt.KeyboardModifier.NoModifier)
    win.page_home.composer.input.keyPressEvent(ev)
    app.processEvents()
    assert ran, "Enter in composer did not submit"


def test_dock_run_empty_input_warns(win, app, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a)))
    win.instruction_input.setPlainText("")
    win.dock.run_btn.click()
    app.processEvents()
    assert warned, "dock Run with empty input should warn"


def test_dock_run_without_orchestrator_shows_offline(win, app, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **k: shown.append(a)))
    win.instruction_input.setPlainText("open notepad")
    win.dock.run_btn.click()
    app.processEvents()
    assert shown, "dock Run button dead"
    assert win._running is False


def test_panel_and_console_toggles(win, app):
    hidden_before = win.panel.isHidden()
    win.panel_toggle.click()
    app.processEvents()
    assert win.panel.isHidden() != hidden_before, "panel toggle dead"
    win.panel_toggle.click()
    app.processEvents()
    assert win.panel.isHidden() == hidden_before

    win.panel.console_toggle.click()  # open raw log
    app.processEvents()
    assert win.panel.console_toggle.isChecked()


def test_sessions_refresh_and_settings_buttons(win, app, monkeypatch):
    win._goto_page(2)
    app.processEvents()
    win.page_sessions.refresh()  # must not raise

    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a)))
    win._goto_page(6)
    app.processEvents()
    # credential save with empty fields → warning path wired
    win.page_settings._save_credential()
    assert warned, "Settings Save button dead"


def test_empty_state_quick_action_runs_through_composer(win, app):
    """Empty-state chips on Home must land in the composer."""
    win._goto_page(0)
    app.processEvents()
    win.page_home._pick("Open Calculator")
    assert win.page_home.composer.input.toPlainText() == "Open Calculator"
    assert win.instruction_input.toPlainText() == "Open Calculator"
