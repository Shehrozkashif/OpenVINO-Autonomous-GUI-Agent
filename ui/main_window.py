# ui/main_window.py
"""
Main PyQt6 application window.
Tabs: Agent Control | History | Settings
Agent runs in background thread; UI stays responsive via Qt signals.
"""
import io
import sys
import threading

from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFormLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSystemTrayIcon,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    QMenu
)


class WorkerSignals(QObject):
    log_update = pyqtSignal(str)
    screenshot_update = pyqtSignal(bytes)
    task_complete = pyqtSignal(dict)
    error = pyqtSignal(str)


class DesktopGUIAgent(QMainWindow):
    def __init__(self, orchestrator=None):
        super().__init__()
        self.orchestrator = orchestrator
        self.settings = QSettings("OpenVINO-GSoC", "DesktopGUIAgent")
        self.signals = WorkerSignals()
        self._setup_tray()
        self._init_ui()
        self._connect_signals()
        self._start_screen_timer()

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        px = QPixmap(32, 32)
        px.fill(Qt.GlobalColor.blue)
        self.tray.setIcon(QIcon(px))
        menu = QMenu()
        menu.addAction("Open", self.show)
        menu.addSeparator()
        menu.addAction("Quit", QApplication.instance().quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _init_ui(self):
        self.setWindowTitle("Desktop GUI Agent — GSoC 2026")
        self.setGeometry(100, 100, 1400, 900)
        tabs = QTabWidget()
        tabs.addTab(self._agent_tab(), "Agent Control")
        tabs.addTab(self._history_tab(), "History")
        tabs.addTab(self._settings_tab(), "Settings")
        self.setCentralWidget(tabs)

    def _agent_tab(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)

        # Left control panel
        ctrl = QWidget()
        ctrl.setMaximumWidth(350)
        cl = QVBoxLayout(ctrl)
        cl.addWidget(QLabel("Your Instruction:"))
        self.instruction_input = QTextEdit()
        self.instruction_input.setPlaceholderText(
            "What should I do?\n\nExamples:\n"
            "• Open VS Code and enable autosave\n"
            "• Move latest file from Downloads to Desktop\n"
            "• Open Chrome and search for OpenVINO"
        )
        self.instruction_input.setMinimumHeight(200)
        cl.addWidget(self.instruction_input)

        self.run_btn = QPushButton("Run Task")
        self.run_btn.setMinimumHeight(45)
        self.run_btn.clicked.connect(self._run_task)
        cl.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_task)
        cl.addWidget(self.stop_btn)
        cl.addStretch()

        # Right display panel
        disp = QWidget()
        dl = QVBoxLayout(disp)
        dl.addWidget(QLabel("Live Screen:"))
        self.screen_label = QLabel("Screen view appears here during task execution")
        self.screen_label.setFixedSize(900, 550)
        self.screen_label.setStyleSheet("border: 2px solid #333; background: black; color: white;")
        self.screen_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dl.addWidget(self.screen_label)
        dl.addWidget(QLabel("Agent Log:"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(200)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 12px;")
        dl.addWidget(self.log_view)

        layout.addWidget(ctrl)
        layout.addWidget(disp, stretch=1)
        return w

    def _history_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Past task executions:"))
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        l.addWidget(self.history_view)
        return w

    def _settings_tab(self) -> QWidget:
        w = QWidget()
        l = QFormLayout(w)
        l.addRow("Backend:", QLabel("Ollama  (qwen3:14b LLM  +  qwen2.5vl-gui VLM)"))
        l.addRow("LLM port:", QLabel("localhost:11434"))
        l.addRow("VLM port:", QLabel("localhost:8000  (vLLM/UI-TARS, if running)  →  fallback to Ollama"))
        return w

    def _connect_signals(self):
        self.signals.log_update.connect(self._append_log)
        self.signals.screenshot_update.connect(self._show_screenshot)
        self.signals.task_complete.connect(self._on_done)
        self.signals.error.connect(self._on_error)

    def _start_screen_timer(self):
        self._screen_timer = QTimer()
        self._screen_timer.timeout.connect(self._refresh_screen)
        self._screen_timer.start(1000)  # 1 FPS live view

    def _refresh_screen(self):
        from core.capture.screenshot import ScreenCapture
        try:
            img = ScreenCapture().capture_resized(900, 550)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self.signals.screenshot_update.emit(buf.getvalue())
        except Exception:
            pass

    def _show_screenshot(self, img_bytes: bytes):
        px = QPixmap()
        px.loadFromData(img_bytes)
        self.screen_label.setPixmap(px.scaled(
            900, 550,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _run_task(self):
        instruction = self.instruction_input.toPlainText().strip()
        if not instruction:
            QMessageBox.warning(self, "Empty", "Please type an instruction.")
            return
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_view.clear()
        self.showMinimized()
        # Delay thread start by 500 ms so the window manager fully hides
        # the agent window before keypresses/clicks begin.  Using QTimer
        # keeps the main thread responsive (no blocking sleep).
        QTimer.singleShot(
            500,
            lambda: threading.Thread(
                target=self._worker, args=(instruction,), daemon=True
            ).start(),
        )

    def _worker(self, instruction: str):
        try:
            if self.orchestrator is None:
                self.signals.error.emit("Orchestrator not initialized — is Ollama running? (ollama serve)")
                return
            self.orchestrator.log = lambda msg: self.signals.log_update.emit(msg)
            result = self.orchestrator.execute(instruction)
            self.signals.task_complete.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))

    def _stop_task(self):
        if self.orchestrator:
            self.orchestrator.stop()
        self.stop_btn.setEnabled(False)

    def _append_log(self, text: str):
        self.log_view.append(text)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _on_done(self, result: dict):
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        status = "Task complete" if result["success"] else "Task failed"
        self.tray.showMessage("Agent", result.get("summary", status), QSystemTrayIcon.MessageIcon.Information, 3000)
        self.history_view.append(
            f"[{result['task_id']}] {'OK' if result['success'] else 'FAIL'} "
            f"({result['elapsed_s']:.1f}s) {result.get('summary', '')}"
        )

    def _on_error(self, msg: str):
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", msg)

