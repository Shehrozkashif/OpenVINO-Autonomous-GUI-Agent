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
from PyQt6.QtGui import QBrush, QColor, QIcon, QLinearGradient, QPainter, QPixmap, QRadialGradient
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
    MemoryPage,
    MissionPage,
    ScreenHistoryPage,
    SessionsPage,
    SettingsPage,
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
    ("db",     "Memory"),
    ("screen", "Screen History"),
    ("gear",   "Settings"),
]


class DesktopGUIAgent(QMainWindow):
    def __init__(self, orchestrator=None):
        super().__init__()
        self.orchestrator = orchestrator
        self.settings = QSettings("OpenVINO-GSoC", "DesktopGUIAgent")
        self.signals = WorkerSignals()
        self.bus = AgentEventBus()
        self._running = False
        self._memory = None
        # (timestamp, QPixmap, action_text) frames recorded during missions
        self.frame_store = deque(maxlen=48)
        self._frame_counter = 0

        self.setWindowTitle("Desktop GUI Agent — GSoC 2026")
        self.setGeometry(80, 60, 1480, 920)
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

    # ── Memory access (shared by pages) ───────────────────────────────────────

    def _get_memory(self):
        if self.orchestrator is not None:
            return self.orchestrator.memory
        if self._memory is None:
            try:
                from memory.task_memory import TaskMemory
                self._memory = TaskMemory()
            except Exception:
                return None
        return self._memory

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
        self.page_home = HomePage(self._get_memory, self.bus)
        self.page_mission = MissionPage(self.bus)
        self.page_sessions = SessionsPage(self._get_memory)
        self.page_workflows = WorkflowsPage(self._get_memory)
        self.page_memory = MemoryPage(self._get_memory)
        self.page_screens = ScreenHistoryPage(self.frame_store)
        self.page_settings = SettingsPage()
        for page in (self.page_home, self.page_mission, self.page_sessions,
                     self.page_workflows, self.page_memory, self.page_screens,
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

    def _start_screen_timer(self):
        self._screen_timer = QTimer(self)
        self._screen_timer.timeout.connect(self._refresh_screen)
        self._screen_timer.start(1000)  # 1 FPS live view

    def _refresh_screen(self):
        from core.capture.screenshot import ScreenCapture
        try:
            img = ScreenCapture().capture_resized(960, 540)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self.signals.screenshot_update.emit(buf.getvalue())
        except Exception:
            pass

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
                "Orchestrator not initialized — is OpenVINO Model Server running?\n\n"
                "Start it with:  python start.py")
            return

        self._running = True
        self.dock.set_running(True)
        self.page_home.composer.set_running(True)
        self.bus.reset()
        self.panel.clear_mission()
        self._goto_page(1)  # Mission Control
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
            self.orchestrator.log = \
                lambda msg: self.signals.log_update.emit(msg)
            result = self.orchestrator.execute(instruction)
            self.signals.task_complete.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

    def _stop_task(self):
        if self.orchestrator:
            self.orchestrator.stop()

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
