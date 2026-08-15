# ui/main_window.py
"""Agent command center — the main application shell.

Layout:  NavRail | header + page stack + CommandDock | IntelligencePanel

Public contract (used by main.py):
    DesktopGUIAgent(orchestrator=...)   window.instruction_input.setPlainText()
    window._run_task()                  window.show()

NOTE: the window title MUST contain "Desktop GUI Agent" — the orchestrator
masks this window out of screen captures by matching that title substring.
"""
import io
import threading
import time
from collections import deque

from PyQt6.QtCore import QObject, QSettings, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QIcon,
    QLinearGradient,
    QPainter,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ui.events import BUSY_STATES, AgentEventBus, AgentState, LoguruBridge
from ui.hud import MissionHUD
from ui.icons import icon_pixmap
from ui.pages import (
    HomePage,
    MissionPage,
    ScreenHistoryPage,
    SessionsPage,
    SettingsPage,
    TaskHistoryPage,
    WorkflowsPage,
)
from ui.panels import IntelligencePanel
from ui.theme import STATE_STYLE, C, S, build_stylesheet
from ui.widgets import CommandDock, NavRail, StatusChip


class WorkerSignals(QObject):
    log_update = pyqtSignal(str)
    screenshot_update = pyqtSignal(bytes)
    task_complete = pyqtSignal(dict)
    error = pyqtSignal(str)
    # Blocking questions from the worker thread (missing-parameter elicitation,
    # destructive-command confirmation). ctx = {"event": threading.Event,
    # "answer": [None]} — the UI slot fills answer and sets the event.
    ask_user = pyqtSignal(str, object)
    confirm_action = pyqtSignal(str, str, object)
    # Missing-detail elicitation finished (pre-mission, window still visible);
    # carries the instruction enriched with the user's answers.
    elicit_done = pyqtSignal(str)
    # Pointer action fired by the controller (worker thread) — the UI thread
    # pulses a capture-excluded ring at that spot so clicks are visible.
    pointer_action = pyqtSignal(int, int, str)


class Shell(QWidget):
    """Background canvas: deep vertical gradient + two soft color glows."""

    def paintEvent(self, e):
        p = QPainter(self)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(C.BG0))
        grad.setColorAt(1.0, QColor(C.BG1))
        p.fillRect(self.rect(), QBrush(grad))

        glow1 = QRadialGradient(self.width() * 0.18, 0, self.width() * 0.5)
        glow1.setColorAt(0, QColor(34, 211, 238, 16))
        glow1.setColorAt(1, QColor(34, 211, 238, 0))
        p.fillRect(self.rect(), QBrush(glow1))

        glow2 = QRadialGradient(self.width() * 0.95, self.height(),
                                self.width() * 0.6)
        glow2.setColorAt(0, QColor(124, 108, 246, 18))
        glow2.setColorAt(1, QColor(124, 108, 246, 0))
        p.fillRect(self.rect(), QBrush(glow2))


_PAGES = [
    ("home",   "Home"),
    ("bolt",   "Mission Control"),
    ("layers", "Agent Sessions"),
    ("flow",   "Workflows"),
    ("db",     "Task History"),
    ("screen", "Screen History"),
    ("gear",   "Settings"),
]


class DesktopGUIAgent(QMainWindow):
    # Preferred opening size on a roomy display. Not a promise — the window is
    # shrunk to whatever the screen actually offers, see _apply_startup_geometry.
    _WANT_W, _WANT_H = 1480, 920
    # Floor. The nav rail (196 expanded) and the activity panel (312 fixed)
    # together take 508, so below ~1040 the middle column stops being usable.
    # The height floor is kept under 720 so a 1366x768 laptop still fits after
    # its taskbar.
    _MIN_W, _MIN_H = 1040, 640
    # resize() sizes the CLIENT area; the border and title bar sit outside it.
    # Clamping to exactly the screen width therefore still hangs the frame off
    # the edge, which is visible on a 1024x768 display. Reserve room for it.
    _FRAME_W, _FRAME_H = 16, 48

    def _apply_startup_geometry(self):
        """Open at a size that FITS, centred on the current screen.

        This used to be setGeometry(80, 60, 1480, 920) — a fixed rectangle
        needing 980 px of height once the 60 px offset is counted. A 1080p
        laptop has about 1032 px left after the taskbar, and less at any
        display scaling above 100%, so the bottom of the window — the
        instruction box and the Run button — sat under the taskbar or off the
        screen entirely. The first thing a new user had to do was drag the
        window bigger to find the controls.

        availableGeometry() is the taskbar-aware rectangle of the screen, so
        sizing against it is what keeps every control reachable on a small
        laptop and a 4K monitor alike.
        """
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:      # no display at all — nothing to fit to
            self.setMinimumSize(self._MIN_W, self._MIN_H)
            self.resize(self._WANT_W, self._WANT_H)
            return

        avail = screen.availableGeometry()
        # The floor is capped by the screen. A minimum wider than the display
        # cannot be honoured by shrinking — Qt just hands back a window bigger
        # than the screen, which is the very bug this method exists to fix.
        # Dropping our own floor here does not let the layout collapse: Qt still
        # enforces the minimum its child widgets need.
        # Same frame allowance as fit_to_screen: a minimum of exactly the screen
        # width would override the fitted size and put the border back off-edge.
        self.setMinimumSize(min(self._MIN_W, avail.width() - self._FRAME_W),
                            min(self._MIN_H, avail.height() - self._FRAME_H))

        w, h = self.fit_to_screen(avail.width(), avail.height())
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w) // 2,
                  avail.y() + (avail.height() - h) // 2)

    @classmethod
    def fit_to_screen(cls, avail_w: int, avail_h: int) -> tuple[int, int]:
        """Opening size for a screen whose usable area is avail_w x avail_h.

        Split out from _apply_startup_geometry so the rule can be checked
        against real display sizes without a display — see test_ui.py. The one
        invariant that matters: the result never exceeds what it was given.
        """
        # A margin keeps it reading as a window rather than a kiosk, but must
        # never shrink it past the floor.
        w = max(cls._MIN_W, min(cls._WANT_W, avail_w - 80))
        h = max(cls._MIN_H, min(cls._WANT_H, avail_h - 80))
        # On a display smaller than the floor, fitting wins over the floor.
        # The frame allowance only bites here — on any normal screen the 80 px
        # margin above is already the binding constraint.
        return min(w, avail_w - cls._FRAME_W), min(h, avail_h - cls._FRAME_H)

    def __init__(self, orchestrator=None):
        super().__init__()
        self.orchestrator = orchestrator
        self.settings = QSettings("OpenVINO-GSoC", "DesktopGUIAgent")
        self.signals = WorkerSignals()
        self.bus = AgentEventBus()
        self._running = False
        self._history = None
        self._click_pulse = None   # lazy ClickPulse overlay (see _on_pointer_action)
        # (timestamp, QPixmap, action_text) frames recorded during missions
        self.frame_store = deque(maxlen=48)
        self._frame_counter = 0

        self.setWindowTitle("Desktop GUI Agent")
        self._apply_startup_geometry()
        self.setWindowIcon(QIcon(icon_pixmap("sparkle", QColor(C.ACCENT), 32)))
        self.setStyleSheet(build_stylesheet())

        self._setup_tray()
        self._init_ui()
        self._connect_signals()
        self._start_screen_timer()

        # Always-on-top mission HUD — visible while this window is minimized
        self.hud = MissionHUD(self.bus)
        self.hud.stop_btn.clicked.connect(self._stop_task)
        if self.orchestrator is not None:
            self.hud.attach_capturer(self.orchestrator.capturer)

        # Deep pipeline events (planner/grounder/reflector loguru logs) →
        # the same event bus, so the UI shows planning/locating/verifying live.
        self.log_bridge = LoguruBridge(self)
        self.log_bridge.line.connect(self.bus.feed)
        self.log_bridge.install()

    # ── Task-history access (shared by pages) ─────────────────────────────────

    def _get_history(self):
        if self.orchestrator is not None:
            return self.orchestrator.history
        if self._history is None:
            try:
                from core.history import TaskHistory
                self._history = TaskHistory()
            except Exception:
                return None
        return self._history

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self):
        shell = Shell()
        root = QHBoxLayout(shell)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left: navigation rail
        self.nav = NavRail(_PAGES)
        self.nav.navigate.connect(self._goto_page)
        root.addWidget(self.nav)

        # Center column
        center = QWidget()
        col = QVBoxLayout(center)
        col.setContentsMargins(S.LG, S.MD, S.MD, S.LG)
        col.setSpacing(S.MD)

        header = QHBoxLayout()
        self.page_title = QLabel("Home")
        self.page_title.setProperty("role", "h2")
        header.addWidget(self.page_title)
        header.addStretch()
        self.status_chip = StatusChip("Ready", C.TEXT_DIM)
        header.addWidget(self.status_chip)
        self.panel_toggle = QPushButton()
        self.panel_toggle.setProperty("kind", "ghost")
        self.panel_toggle.setIcon(
            QIcon(icon_pixmap("panel", QColor(C.TEXT_DIM), 16)))
        self.panel_toggle.setFixedSize(34, 30)
        self.panel_toggle.setToolTip("Toggle intelligence panel")
        self.panel_toggle.clicked.connect(self._toggle_panel)
        header.addWidget(self.panel_toggle)
        col.addLayout(header)

        # Pages
        self.stack = QStackedWidget()
        self.page_home = HomePage(self._get_history, self.bus)
        self.page_mission = MissionPage(self.bus)
        self.page_sessions = SessionsPage(self._get_history)
        self.page_workflows = WorkflowsPage(self._get_history)
        self.page_history = TaskHistoryPage(self._get_history)
        self.page_screens = ScreenHistoryPage(self.frame_store)
        self.page_settings = SettingsPage()
        for page in (self.page_home, self.page_mission, self.page_sessions,
                     self.page_workflows, self.page_history, self.page_screens,
                     self.page_settings):
            self.stack.addWidget(page)
        col.addWidget(self.stack, stretch=1)

        # Command dock (persistent operator bar)
        self.dock = CommandDock()
        self.dock.run_requested.connect(self._run_task)
        self.dock.stop_requested.connect(self._stop_task)
        col.addWidget(self.dock)
        # main.py contract: instruction_input has setPlainText/toPlainText
        self.instruction_input = self.dock.input

        root.addWidget(center, stretch=1)

        # Right: intelligence panel
        self.panel = IntelligencePanel(self.bus)
        root.addWidget(self.panel)

        self.setCentralWidget(shell)
        self.nav.set_active(0)

        # Wire page → command dock interactions
        self.page_home.suggestion_chosen.connect(self._fill_input)
        self.page_home.run_requested.connect(self._fill_and_run)
        self.page_sessions.rerun.connect(self._fill_and_run)
        self.page_workflows.run_workflow.connect(self._fill_and_run)

        self.bus.state_changed.connect(self._on_state)

    def _goto_page(self, idx: int):
        # No fade here: pages contain GlassCards with drop-shadow effects, and
        # nesting those inside a page-level QGraphicsOpacityEffect makes Qt
        # emit a re-entrant-QPainter warning for every shadowed child on every
        # frame of the fade. Leaf widgets (feed/timeline items) still fade.
        self.stack.setCurrentIndex(idx)
        self.page_title.setText(_PAGES[idx][1])
        self.nav.set_active(idx)

    def _toggle_panel(self):
        # isHidden() (explicit hide state) — isVisible() is False whenever the
        # window is minimized, which made the toggle a no-op mid-mission.
        self.panel.setVisible(self.panel.isHidden())

    def _fill_input(self, text: str):
        self.instruction_input.setPlainText(text)
        self.instruction_input.setFocus()

    def _fill_and_run(self, text: str):
        self._fill_input(text)
        self._run_task()

    def _on_state(self, state: AgentState):
        color, label = STATE_STYLE[state.value]
        self.status_chip.set_state(label, color)
        self.dock.orb.set_state(color, state in BUSY_STATES)

    # ── Tray ──────────────────────────────────────────────────────────────────

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(QIcon(icon_pixmap("sparkle", QColor(C.ACCENT), 32)))
        menu = QMenu()
        menu.addAction("Open", self.show)
        menu.addSeparator()
        menu.addAction("Quit", QApplication.instance().quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    # ── Signals / screen timer ────────────────────────────────────────────────

    def _connect_signals(self):
        self.signals.log_update.connect(self.bus.feed)
        self.signals.screenshot_update.connect(self._show_screenshot)
        self.signals.task_complete.connect(self._on_done)
        self.signals.error.connect(self._on_error)
        self.signals.ask_user.connect(self._on_ask_user)
        self.signals.confirm_action.connect(self._on_confirm_action)
        self.signals.elicit_done.connect(self._launch_mission)
        self.signals.pointer_action.connect(self._on_pointer_action)
        # Let the controller report every click so the user can see where the
        # agent's mouse lands (emit is thread-safe from the worker thread).
        if self.orchestrator is not None:
            try:
                self.orchestrator.actor.controller.on_pointer = (
                    lambda x, y, kind: self.signals.pointer_action.emit(x, y, kind)
                )
            except AttributeError:
                pass  # headless/mock orchestrator without a real controller

    def _on_pointer_action(self, x: int, y: int, kind: str):
        """UI-thread slot: pulse the click indicator at the action's location."""
        if self._click_pulse is None:
            from ui.click_pulse import ClickPulse
            self._click_pulse = ClickPulse()
        self._click_pulse.flash(x, y, kind)

    # ── Blocking questions from the worker thread ─────────────────────────────

    def _restore_for_dialog(self) -> bool:
        """Bring the window back if minimized so a modal dialog can be seen
        and hold focus (a dialog parented to a minimized window centers
        off-screen at -32000 and loses keystrokes). Returns True if the
        window must be re-minimized afterwards.
        """
        if self.isMinimized():
            self.showNormal()
            self.raise_()
            self.activateWindow()
            return True
        return False

    def _on_ask_user(self, question: str, ctx: dict):
        """UI-thread slot: ask the user for a missing detail (e.g. meeting time)."""
        from PyQt6.QtWidgets import QInputDialog
        re_minimize = self._restore_for_dialog()
        try:
            text, ok = QInputDialog.getText(self, "The agent needs a detail", question)
            ctx["answer"][0] = text.strip() if ok and text.strip() else None
        finally:
            ctx["event"].set()
            if re_minimize:
                self.showMinimized()

    def _on_confirm_action(self, summary: str, command: str, ctx: dict):
        """UI-thread slot: confirm a potentially destructive command."""
        re_minimize = self._restore_for_dialog()
        reply = QMessageBox.question(
            self, "Confirm potentially destructive action",
            f"{summary}\n\nCommand:\n{command}\n\nAllow the agent to run this?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        try:
            ctx["answer"][0] = reply == QMessageBox.StandardButton.Yes
        finally:
            ctx["event"].set()
            if re_minimize:
                self.showMinimized()

    def _ask_blocking(self, question: str) -> str | None:
        """Called from the worker thread. Blocks (max 180 s) until the user
        answers in the UI thread; returns None on timeout or dismissal.
        """
        ctx = {"event": threading.Event(), "answer": [None]}
        self.signals.ask_user.emit(question, ctx)
        ctx["event"].wait(timeout=180)
        return ctx["answer"][0]

    def _confirm_blocking(self, summary: str, command: str) -> bool:
        """Called from the worker thread. Blocks (max 120 s) for a yes/no;
        returns False (deny) on timeout — never default-allow.
        """
        ctx = {"event": threading.Event(), "answer": [False]}
        self.signals.confirm_action.emit(summary, command, ctx)
        ctx["event"].wait(timeout=120)
        return bool(ctx["answer"][0])

    def _start_screen_timer(self):
        self._capturing = False
        self._screen_timer = QTimer(self)
        self._screen_timer.timeout.connect(self._refresh_screen)
        self._screen_timer.start(1000)  # 1 FPS live view

    def _refresh_screen(self):
        # Grab + PNG-encode the whole desktop OFF the UI thread. At 1440p/4K a
        # full GDI capture + LANCZOS resize + PNG encode is 150-400 ms; running
        # it on the Qt event loop once a second stalled clicks and keystrokes
        # (buttons felt dead, typing stuttered). Only the cheap QPixmap decode
        # stays on the UI thread (_show_screenshot). Skip the cycle if a prior
        # grab is still in flight so slow frames never pile up.
        if self._capturing:
            return
        self._capturing = True
        threading.Thread(target=self._grab_screen_frame, daemon=True).start()

    def _grab_screen_frame(self):
        from desktop.capture import ScreenCapture
        try:
            img = ScreenCapture().capture_resized(960, 540)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self.signals.screenshot_update.emit(buf.getvalue())
        except Exception:
            pass
        finally:
            self._capturing = False

    def _show_screenshot(self, img_bytes: bytes):
        px = QPixmap()
        px.loadFromData(img_bytes)
        self.page_mission.preview.set_frame(px)
        if self._running:
            # record every 3rd frame (~one per 3s) for Screen History
            self._frame_counter += 1
            if self._frame_counter % 3 == 1:
                self.frame_store.append(
                    (time.time(), px, self.bus.current_step))

    # ── Task lifecycle ────────────────────────────────────────────────────────

    def _emit_log(self, msg: str):
        """Route an orchestrator decision to loguru, which reaches both places.

        The GUI wired orchestrator.log straight to the HUD signal, so every
        routing decision ([SUBTASK]/[GOAL-CHECK]/[CLICK-CHECK]/…) was invisible
        in the terminal — the log the dev loop actually pastes. Emitting through
        loguru puts it in the terminal (and file sink) AND still reaches the HUD
        via the always-installed LoguruBridge (events.py) → bus.feed. Going
        through loguru only — not also signals.log_update — avoids feeding
        bus.feed twice (LoguruBridge + log_update), which would double the line.
        """
        try:
            from loguru import logger
            logger.opt(depth=1).info(msg)
        except Exception:
            # Loguru unavailable for some reason — fall back to the HUD signal
            # so a decision is never silently dropped.
            self.signals.log_update.emit(msg)

    def _run_task(self):
        if self._running:
            return
        instruction = self.instruction_input.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "Empty mission",
                                "Tell the agent what to do first.")
            return
        if self.orchestrator is None:
            QMessageBox.critical(
                self, "Agent offline",
                "Orchestrator not initialized - is OpenVINO Model Server running?\n\n"
                "Start it with:  python start.py")
            return

        self._running = True
        self.dock.set_running(True)
        self.page_home.composer.set_running(True)
        self.bus.reset()
        self.panel.clear_mission()
        self._goto_page(1)  # Mission Control

        # Ask for missing details BEFORE minimizing. A modal question parented
        # to a minimized window centers off-screen at (-32000,-32000) and
        # cannot hold keyboard focus — users lost their answer mid-typing and
        # the mission proceeded with the raw instruction. The ~3 s LLM check +
        # dialogs run on a prep thread while the window is still visible;
        # _launch_mission then minimizes and starts the worker with the
        # enriched instruction.
        self.orchestrator.log = self._emit_log
        self.orchestrator.on_ask = self._ask_blocking

        def _prep():
            try:
                enriched = self.orchestrator._elicit_missing_parameters(instruction)
            except Exception:
                enriched = instruction
            self.signals.elicit_done.emit(enriched)

        threading.Thread(target=_prep, daemon=True).start()

    def _launch_mission(self, instruction: str):
        """UI-thread slot: details are in — minimize and start the mission."""
        self.showMinimized()
        self.hud.show_mission()
        # Delay the worker 500 ms so the window manager fully hides this
        # window before clicks/keypresses begin (it would steal focus).
        QTimer.singleShot(
            500,
            lambda: threading.Thread(
                target=self._worker, args=(instruction,), daemon=True
            ).start(),
        )

    def _worker(self, instruction: str):
        try:
            self.orchestrator.log = self._emit_log
            # Elicitation already ran pre-minimize (_run_task) — disable it in
            # execute() so the mission never pops a dialog under a minimized
            # window. Destructive-command confirmation stays wired.
            self.orchestrator.on_ask = None
            self.orchestrator.on_confirm = self._confirm_blocking
            result = self.orchestrator.execute(instruction)
            self.signals.task_complete.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

    def _stop_task(self):
        if self.orchestrator:
            self.orchestrator.stop()
            # Instant feedback: the worker may still be finishing an in-flight
            # model call for a few seconds, so the mission does not end on this
            # exact click. Tell the user their Stop registered instead of
            # leaving a dead-looking button.
            self.hud.state_label.setText("Stopping…")
            self.hud.detail.setText("Finishing the current step, then halting…")
            self.status_chip.set_state("Stopping…", C.WARNING)

    def _on_done(self, result: dict):
        self._running = False
        self.dock.set_running(False)
        self.page_home.composer.set_running(False)
        self.hud.hide_mission()
        self.showNormal()
        self.raise_()
        self.activateWindow()
        stopped = not result["success"] and \
            self.bus.state == AgentState.STOPPED
        self.bus.finish(result["success"], stopped=stopped)
        summary = result.get("summary") or (
            "Task complete" if result["success"] else "Task failed")
        self.tray.showMessage(
            "Agent", summary,
            QSystemTrayIcon.MessageIcon.Information, 3000)
        extracted = result.get("extracted_data", {})
        for key, val in extracted.items():
            if val:
                self.bus.extracted.emit(key, val)
        self.page_home.refresh()

    def _on_error(self, msg: str):
        self._running = False
        self.dock.set_running(False)
        self.page_home.composer.set_running(False)
        self.hud.hide_mission()
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.bus.finish(False)
        QMessageBox.critical(self, "Error", msg)
