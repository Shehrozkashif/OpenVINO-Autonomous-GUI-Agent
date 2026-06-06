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
    QApplication, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
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
        self._tabs = QTabWidget()
        self._tabs.addTab(self._agent_tab(), "Agent Control")
        self._tabs.addTab(self._history_tab(), "History")
        self._tabs.addTab(self._settings_tab(), "Settings")
        self._tabs.currentChanged.connect(self._on_tab_change)
        self.setCentralWidget(self._tabs)

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
        import datetime
        w = QWidget()
        l = QVBoxLayout(w)

        header = QHBoxLayout()
        header.addWidget(QLabel("Past task executions (from memory):"))
        header.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_history)
        header.addWidget(refresh_btn)
        l.addLayout(header)

        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setStyleSheet("font-family: monospace; font-size: 12px;")
        l.addWidget(self.history_view)

        # Load history on first show
        QTimer.singleShot(500, self._load_history)
        return w

    def _load_history(self):
        """Reload task history from SQLite into the history view."""
        import datetime
        try:
            from memory.task.task_memory import TaskMemory
            memory = TaskMemory()
            tasks = memory.get_recent_tasks(limit=50)
        except Exception as e:
            self.history_view.setPlainText(f"Could not load history: {e}")
            return

        if not tasks:
            self.history_view.setPlainText("No task history yet. Run a task to see it here.")
            return

        lines = []
        for t in tasks:
            ts = datetime.datetime.fromtimestamp(t["last_used"]).strftime("%Y-%m-%d %H:%M")
            dur = f"{t['avg_duration_s']:.1f}s" if t["avg_duration_s"] else "?"
            count = t["success_count"]
            lines.append(
                f"[{ts}]  runs={count}  avg={dur}\n"
                f"  {t['instruction']}\n"
            )
        self.history_view.setPlainText("\n".join(lines))

    def _on_tab_change(self, index: int):
        if self._tabs.tabText(index) == "History":
            self._load_history()

    def _settings_tab(self) -> QWidget:
        from config import LLM_MODEL, VLM_OLLAMA, VLM_VLLM, LLM_BASE_URL, VLM_BASE_URL
        w = QWidget()
        root = QVBoxLayout(w)

        # ── Model info ────────────────────────────────────────────
        model_form = QFormLayout()
        model_form.addRow("LLM model:", QLabel(f"{LLM_MODEL}  (routing, planning, reflection)"))
        model_form.addRow("VLM primary:", QLabel(f"{VLM_VLLM}  via vLLM at {VLM_BASE_URL}"))
        model_form.addRow("VLM fallback:", QLabel(f"{VLM_OLLAMA}  via Ollama at {LLM_BASE_URL}"))
        model_form.addRow("", QLabel("Edit config.py to change models or ports."))
        root.addLayout(model_form)

        root.addWidget(QLabel(""))   # spacer

        # ── Credential manager ────────────────────────────────────
        root.addWidget(QLabel("Saved Credentials  (used with {{cred:site:field}} in tasks):"))

        cred_row = QHBoxLayout()
        self._cred_site  = QLineEdit(); self._cred_site.setPlaceholderText("site  (e.g. github.com)")
        self._cred_user  = QLineEdit(); self._cred_user.setPlaceholderText("username")
        self._cred_pass  = QLineEdit(); self._cred_pass.setPlaceholderText("password")
        self._cred_pass.setEchoMode(QLineEdit.EchoMode.Password)
        save_btn  = QPushButton("Save")
        save_btn.clicked.connect(self._save_credential)
        del_btn   = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_credential)
        for widget in (self._cred_site, self._cred_user, self._cred_pass, save_btn, del_btn):
            cred_row.addWidget(widget)
        root.addLayout(cred_row)

        self._cred_list = QTextEdit()
        self._cred_list.setReadOnly(True)
        self._cred_list.setMaximumHeight(120)
        self._cred_list.setStyleSheet("font-family: monospace; font-size: 11px;")
        root.addWidget(self._cred_list)
        self._refresh_cred_list()

        root.addStretch()
        return w

    def _save_credential(self):
        site = self._cred_site.text().strip()
        user = self._cred_user.text().strip()
        pwd  = self._cred_pass.text()
        if not site or not user:
            QMessageBox.warning(self, "Missing", "Site and username are required.")
            return
        from utils.credentials import set_credential
        set_credential(site, user, pwd)
        self._cred_site.clear(); self._cred_user.clear(); self._cred_pass.clear()
        self._refresh_cred_list()

    def _delete_credential(self):
        site = self._cred_site.text().strip()
        if not site:
            QMessageBox.warning(self, "Missing", "Enter the site name to delete.")
            return
        from utils.credentials import delete
        delete(site)
        self._cred_site.clear()
        self._refresh_cred_list()

    def _refresh_cred_list(self):
        try:
            from utils.credentials import list_sites
            sites = list_sites()
            if sites:
                self._cred_list.setPlainText("\n".join(f"  {s}" for s in sites))
            else:
                self._cred_list.setPlainText("  (no credentials stored)")
        except Exception as e:
            self._cred_list.setPlainText(f"  Error: {e}")

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
        # Show any extracted data prominently in the log
        extracted = result.get("extracted_data", {})
        if extracted:
            self._append_log("\n── Extracted Data ──")
            for key, val in extracted.items():
                self._append_log(f"  {key}: {val}")
        self._load_history()

    def _on_error(self, msg: str):
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "Error", msg)

