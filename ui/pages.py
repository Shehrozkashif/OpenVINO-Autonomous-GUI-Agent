# ui/pages.py
"""The seven workspace pages of the command center.

Every page refreshes its data on showEvent (cheap SQLite reads) and never
blocks the UI thread — anything slow (health checks) runs in a worker thread.
"""
import datetime
import threading
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.events import AgentEventBus, AgentState
from ui.icons import icon_pixmap
from ui.theme import STATE_STYLE, C, S
from ui.widgets import (
    CommandInput,
    EmptyState,
    GlassCard,
    MetricTile,
    PulseOrb,
    ScreenPreview,
    SectionHeader,
    Timeline,
    relative_time,
)

_SUGGESTIONS = [
    "Open Notepad and write a haiku about automation",
    "Open Chrome and search for OpenVINO",
    "Open Calculator",
    "Move the latest file from Downloads to Desktop",
]


def _scrollable(inner: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.Shape.NoFrame)
    sc.setWidget(inner)
    return sc


# ── Home ──────────────────────────────────────────────────────────────────────

class MissionComposer(GlassCard):
    """The primary prompt surface: a large natural-language instruction box
    with a Run button — front and center, exactly like a command line for
    the agent.
    """

    run_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("tone", "accent")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        lay.setSpacing(S.SM)

        cap = QLabel("GIVE THE AGENT A MISSION")
        cap.setProperty("role", "micro")
        lay.addWidget(cap)

        self.input = CommandInput()
        self.input.setPlaceholderText(
            "Describe a task in plain language…\n"
            "e.g.  Open Notepad and write a haiku about automation")
        self.input.setStyleSheet(
            "background: transparent; border: none; font-size: 15px;")
        self.input.setMinimumHeight(64)
        self.input.submitted.connect(self._submit)
        lay.addWidget(self.input)

        row = QHBoxLayout()
        hint = QLabel("Enter to run · Shift+Enter for a new line")
        hint.setProperty("role", "faint")
        row.addWidget(hint)
        row.addStretch()
        self.run_btn = QPushButton("  Run Task")
        self.run_btn.setProperty("kind", "primary")
        self.run_btn.setIcon(_qicon("play", "#06121A"))
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setMinimumWidth(140)
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.clicked.connect(self._submit)
        row.addWidget(self.run_btn)
        lay.addLayout(row)

    def _submit(self):
        text = self.input.toPlainText().strip()
        if text:
            self.run_requested.emit(text)

    def set_text(self, text: str):
        self.input.setPlainText(text)
        self.input.setFocus()
        cursor = self.input.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.input.setTextCursor(cursor)

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.input.setReadOnly(running)


class HomePage(QWidget):
    suggestion_chosen = pyqtSignal(str)
    run_requested = pyqtSignal(str)

    def __init__(self, get_memory, bus: AgentEventBus, parent=None):
        super().__init__(parent)
        self.get_memory = get_memory
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(S.XL, S.XL, S.XL, S.XL)
        root.setSpacing(S.LG)

        # ── Hero ───────────────────────────────────────────────
        hero = QHBoxLayout()
        hero.setSpacing(S.XL)
        self.orb = PulseOrb(104)
        hero.addWidget(self.orb)
        hcol = QVBoxLayout()
        hcol.setSpacing(4)
        hcol.addStretch()
        self.headline = QLabel("Your agent is ready.")
        self.headline.setProperty("role", "display")
        hcol.addWidget(self.headline)
        try:
            from config import LLM_MODEL, VLM_MODEL
            sub = f"{LLM_MODEL}  ·  {VLM_MODEL}  ·  OpenVINO Model Server"
        except Exception:
            sub = "local inference"
        subtitle = QLabel(sub)
        subtitle.setProperty("role", "dim")
        hcol.addWidget(subtitle)
        hcol.addStretch()
        hero.addLayout(hcol, stretch=1)
        root.addLayout(hero)

        bus.state_changed.connect(self._on_state)

        # ── Mission composer (primary prompt box) ──────────────
        self.composer = MissionComposer()
        self.composer.run_requested.connect(self.run_requested)
        root.addWidget(self.composer)

        # ── Suggestions ────────────────────────────────────────
        sug_cap = QLabel("SUGGESTED MISSIONS")
        sug_cap.setProperty("role", "micro")
        root.addWidget(sug_cap)
        self.sug_row = QHBoxLayout()
        self.sug_row.setSpacing(8)
        self.sug_row.addStretch()
        root.addLayout(self.sug_row)

        # ── Metrics ────────────────────────────────────────────
        grid = QHBoxLayout()
        grid.setSpacing(S.MD)
        self.m_automations = MetricTile("Automations learned", "flow", C.ACCENT2)
        self.m_runs = MetricTile("Successful runs", "check", C.SUCCESS)
        self.m_avg = MetricTile("Avg task duration", "clock", C.ACCENT)
        self.m_patterns = MetricTile("Failure patterns avoided", "shield", C.WARNING)
        for m in (self.m_automations, self.m_runs, self.m_avg, self.m_patterns):
            grid.addWidget(m, stretch=1)
        root.addLayout(grid)

        # ── Recent automations ─────────────────────────────────
        rec_cap = QLabel("RECENT AUTOMATIONS")
        rec_cap.setProperty("role", "micro")
        root.addWidget(rec_cap)
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(8)
        root.addLayout(self.recent_box)
        root.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))

    def _on_state(self, state: AgentState):
        color, label = STATE_STYLE[state.value]
        busy = state not in (AgentState.IDLE, AgentState.COMPLETE,
                             AgentState.FAILED, AgentState.STOPPED)
        self.orb.set_state(color, busy)
        headlines = {
            AgentState.IDLE: "Your agent is ready.",
            AgentState.COMPLETE: "Mission accomplished.",
            AgentState.FAILED: "Mission needs attention.",
            AgentState.STOPPED: "Mission stopped.",
        }
        self.headline.setText(headlines.get(state, "Agent operating…"))

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def refresh(self):
        mem = self.get_memory()
        tasks, patterns = [], 0
        if mem is not None:
            try:
                tasks = mem.get_recent_tasks(limit=30)
                patterns = mem.conn.execute(
                    "SELECT COUNT(*) FROM failure_patterns").fetchone()[0]
            except Exception:
                pass

        self.m_automations.set_value(str(len(tasks)))
        self.m_runs.set_value(str(sum(t["success_count"] for t in tasks)))
        durs = [t["avg_duration_s"] for t in tasks if t["avg_duration_s"]]
        self.m_avg.set_value(f"{sum(durs) / len(durs):.0f}s" if durs else "—")
        self.m_patterns.set_value(str(patterns))

        # suggestion chips: 2 from memory + canned examples
        while self.sug_row.count() > 1:
            it = self.sug_row.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        chips = [t["instruction"] for t in tasks[:2]]
        chips += [s for s in _SUGGESTIONS if s not in chips]
        for text in chips[:4]:
            b = QPushButton(text if len(text) <= 52 else text[:50] + "…")
            b.setProperty("kind", "chip")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setToolTip(text)
            b.clicked.connect(lambda _, t=text: self._pick(t))
            self.sug_row.insertWidget(self.sug_row.count() - 1, b)

        # recent list
        while self.recent_box.count():
            it = self.recent_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        if not tasks:
            empty = EmptyState(
                "sparkle", "No automations yet",
                "Run your first mission below — successful runs are remembered "
                "and become one-click workflows.",
                actions=("Open Calculator", "Open Notepad and write a haiku"))
            empty.action_clicked.connect(self._pick)
            self.recent_box.addWidget(empty)
            return
        for t in tasks[:5]:
            card = GlassCard(hoverable=True, shadow=False)
            cl = QHBoxLayout(card)
            cl.setContentsMargins(S.LG, S.MD, S.LG, S.MD)
            ic = QLabel()
            ic.setPixmap(icon_pixmap("bolt", QColor(C.ACCENT), 16))
            cl.addWidget(ic)
            name = QLabel(t["instruction"])
            name.setStyleSheet("font-size: 13px;")
            cl.addWidget(name, stretch=1)
            meta = QLabel(
                f"{t['success_count']}× · "
                f"{(t['avg_duration_s'] or 0):.0f}s · "
                f"{relative_time(t['last_used'])}")
            meta.setProperty("role", "faint")
            cl.addWidget(meta)
            card.clicked.connect(
                lambda t=t: self._pick(t["instruction"]))
            self.recent_box.addWidget(card)

    def _pick(self, text: str):
        """Suggestion chosen — drop it into the composer, ready to run."""
        self.composer.set_text(text)
        self.suggestion_chosen.emit(text)


# ── Mission Control (Tasks) ───────────────────────────────────────────────────

class MissionPage(QWidget):
    """Live execution view: screen preview + subtask/step timeline + metrics."""

    def __init__(self, bus: AgentEventBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        root = QVBoxLayout(self)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.MD)
        root.setSpacing(S.MD)

        top = QHBoxLayout()
        top.addWidget(SectionHeader(
            "Mission Control",
            "What the agent sees and does, in real time."))
        top.addStretch()
        self.elapsed = QLabel("")
        self.elapsed.setProperty("role", "mono")
        top.addWidget(self.elapsed)
        root.addLayout(top)

        self.preview = ScreenPreview()
        root.addWidget(self.preview, stretch=5)

        strip = QHBoxLayout()
        strip.setSpacing(S.MD)
        self.s_steps = _stat("STEPS", "0")
        self.s_fail = _stat("RECOVERIES", "0")
        self.s_conf = _stat("LAST CONFIDENCE", "—")
        self.s_subtask = _stat("OBJECTIVE", "—", stretch=True)
        strip.addWidget(self.s_subtask[0], stretch=1)
        strip.addWidget(self.s_steps[0])
        strip.addWidget(self.s_fail[0])
        strip.addWidget(self.s_conf[0])
        root.addLayout(strip)

        tl_cap = QLabel("EXECUTION TIMELINE")
        tl_cap.setProperty("role", "micro")
        root.addWidget(tl_cap)
        self.timeline = Timeline()
        self.timeline.setMinimumHeight(170)
        root.addWidget(self.timeline, stretch=4)

        self._t0 = None
        self._screen_dims = None
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(500)
        self._tick_timer.timeout.connect(self._tick)

        self._wire(bus)

    def _wire(self, bus: AgentEventBus):
        bus.task_started.connect(self._on_task_started)
        bus.subtask_started.connect(self._on_subtask)
        bus.subtask_finished.connect(self.timeline.finish_subtask)
        bus.step_started.connect(self._on_step)
        bus.step_verified.connect(self._on_verified)
        bus.step_failed.connect(self._on_failed)
        bus.retrying.connect(self.timeline.retry_step)
        bus.guard_event.connect(self.timeline.add_guard)
        bus.task_done.connect(self._on_done)
        bus.state_changed.connect(self._on_state)
        # live narration on the preview ("Located 'X' (80% · vlm)…")
        bus.detail.connect(self.preview.set_action_text)
        # grounding-overlay reticle at the located element
        bus.element_located.connect(self._on_located)

    def _on_task_started(self, instruction: str):
        self.timeline.clear()
        self._t0 = time.time()
        self._tick_timer.start()
        self.preview.set_active(True)
        self.preview.clear_targets()
        self.preview.set_action_text(f"Mission: {instruction}")
        for w, default in ((self.s_steps, "0"), (self.s_fail, "0"),
                           (self.s_conf, "—")):
            w[1].setText(default)

    def _on_subtask(self, sid: int, desc: str):
        self.timeline.add_subtask(sid, desc)
        self.s_subtask[1].setText(desc if len(desc) < 60 else desc[:58] + "…")

    def _on_step(self, n: int, action: str, desc: str):
        self.timeline.add_step(action, desc)
        self.preview.set_action_text(desc)
        self.s_steps[1].setText(str(self.bus.steps_total))

    def _on_verified(self, conf: float):
        self.timeline.verify_step(conf)
        self.s_conf[1].setText(f"{conf:.0%}")

    def _on_failed(self, reason: str, conf: float):
        self.timeline.fail_step(reason)
        self.s_fail[1].setText(str(self.bus.steps_failed + self.bus.retries))

    def _on_located(self, target: str, x: int, y: int, conf: float,
                    method: str):
        sw, sh = self._screen_wh()
        self.preview.mark_target(target, x / sw, y / sh, conf)

    def _screen_wh(self):
        """Logical screen size — the grounder's coordinate space."""
        if self._screen_dims is None:
            try:
                from core.capture.screenshot import _screen_size
                w, h = _screen_size()
            except Exception:
                geo = self.screen().geometry()
                w, h = geo.width(), geo.height()
            self._screen_dims = (max(1, w), max(1, h))
        return self._screen_dims

    def _on_done(self, summary: str, elapsed: float):
        self.preview.set_action_text(summary)

    def _on_state(self, state: AgentState):
        if state in (AgentState.COMPLETE, AgentState.FAILED,
                     AgentState.STOPPED, AgentState.IDLE):
            self._tick_timer.stop()
            self.preview.set_active(False)

    def _tick(self):
        if self._t0:
            self.elapsed.setText(f"⏱ {time.time() - self._t0:5.1f}s")


def _stat(caption: str, value: str, stretch: bool = False):
    card = GlassCard(shadow=False)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(S.MD, 6, S.MD, 6)
    lay.setSpacing(0)
    cap = QLabel(caption)
    cap.setProperty("role", "micro")
    lay.addWidget(cap)
    val = QLabel(value)
    val.setStyleSheet("font-size: 14px; font-weight: 700;")
    if stretch:
        val.setSizePolicy(QSizePolicy.Policy.Expanding,
                          QSizePolicy.Policy.Preferred)
    lay.addWidget(val)
    return card, val


# ── Sessions (execution history) ──────────────────────────────────────────────

class SessionsPage(QWidget):
    rerun = pyqtSignal(str)

    def __init__(self, get_memory, parent=None):
        super().__init__(parent)
        self.get_memory = get_memory
        root = QVBoxLayout(self)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.LG)
        root.setSpacing(S.MD)
        top = QHBoxLayout()
        top.addWidget(SectionHeader(
            "Agent Sessions", "Every mission the agent has completed."))
        top.addStretch()
        refresh = QPushButton(" Refresh")
        refresh.setProperty("kind", "ghost")
        refresh.setIcon(_qicon("refresh", C.TEXT_DIM))
        refresh.clicked.connect(self.refresh)
        top.addWidget(refresh)
        root.addLayout(top)

        self.list_host = QWidget()
        self.list_lay = QVBoxLayout(self.list_host)
        self.list_lay.setContentsMargins(0, 0, 0, 0)
        self.list_lay.setSpacing(8)
        self.list_lay.addStretch()
        root.addWidget(_scrollable(self.list_host), stretch=1)

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def refresh(self):
        while self.list_lay.count() > 1:
            it = self.list_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        mem = self.get_memory()
        tasks = []
        if mem is not None:
            try:
                tasks = mem.get_recent_tasks(limit=50)
            except Exception:
                pass
        if not tasks:
            self.list_lay.insertWidget(0, EmptyState(
                "layers", "No sessions yet",
                "Completed missions appear here with timing and reliability "
                "stats."))
            return
        for t in tasks:
            card = GlassCard(hoverable=True, shadow=False)
            lay = QHBoxLayout(card)
            lay.setContentsMargins(S.LG, S.MD, S.LG, S.MD)
            lay.setSpacing(S.MD)
            badge = QLabel(f"{t['success_count']}×")
            badge.setFixedWidth(40)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                f"background: rgba(52,211,153,26); color: {C.SUCCESS};"
                f"border-radius: 10px; padding: 4px; font-weight: 700;"
                f"font-size: 12px;")
            lay.addWidget(badge)
            col = QVBoxLayout()
            col.setSpacing(2)
            name = QLabel(t["instruction"])
            name.setWordWrap(True)
            name.setStyleSheet("font-size: 13px; font-weight: 600;")
            col.addWidget(name)
            ts = datetime.datetime.fromtimestamp(
                t["last_used"]).strftime("%Y-%m-%d %H:%M")
            meta = QLabel(
                f"{len(t['steps'])} subtask(s) · "
                f"avg {(t['avg_duration_s'] or 0):.1f}s · last run {ts}")
            meta.setProperty("role", "faint")
            col.addWidget(meta)
            lay.addLayout(col, stretch=1)
            run = QPushButton("Run again")
            run.setProperty("kind", "ghost")
            run.clicked.connect(lambda _, i=t["instruction"]: self.rerun.emit(i))
            lay.addWidget(run)
            self.list_lay.insertWidget(self.list_lay.count() - 1, card)


# ── Workflows ─────────────────────────────────────────────────────────────────

class WorkflowsPage(QWidget):
    run_workflow = pyqtSignal(str)

    def __init__(self, get_memory, parent=None):
        super().__init__(parent)
        self.get_memory = get_memory
        root = QVBoxLayout(self)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.LG)
        root.setSpacing(S.MD)
        root.addWidget(SectionHeader(
            "Workflow Library",
            "Proven automations the agent has learned. One click to replay."))
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(S.MD)
        self.grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(_scrollable(self.grid_host), stretch=1)

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def refresh(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        mem = self.get_memory()
        tasks = []
        if mem is not None:
            try:
                tasks = mem.get_recent_tasks(limit=30)
            except Exception:
                pass
        if not tasks:
            self.grid.addWidget(EmptyState(
                "flow", "The library is empty",
                "Every successful mission is stored as a reusable workflow —\n"
                "the agent reuses its plan to run faster and more reliably."),
                0, 0)
            return
        for i, t in enumerate(tasks):
            card = GlassCard(hoverable=True, shadow=False)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(S.LG, S.LG, S.LG, S.MD)
            lay.setSpacing(8)
            head = QHBoxLayout()
            ic = QLabel()
            ic.setPixmap(icon_pixmap("flow", QColor(C.ACCENT2), 18))
            head.addWidget(ic)
            head.addStretch()
            runs = QLabel(f"{t['success_count']} run(s)")
            runs.setProperty("role", "faint")
            head.addWidget(runs)
            lay.addLayout(head)
            name = QLabel(t["instruction"])
            name.setWordWrap(True)
            name.setStyleSheet("font-size: 13px; font-weight: 600;")
            name.setMinimumHeight(36)
            lay.addWidget(name)
            steps = QLabel(
                f"{len(t['steps'])} step(s) · ~{(t['avg_duration_s'] or 0):.0f}s")
            steps.setProperty("role", "faint")
            lay.addWidget(steps)
            run = QPushButton(" Run workflow")
            run.setProperty("kind", "primary")
            run.setIcon(_qicon("play", "#06121A"))
            run.clicked.connect(
                lambda _, instr=t["instruction"]: self.run_workflow.emit(instr))
            lay.addWidget(run)
            self.grid.addWidget(card, i // 2, i % 2)
        self.grid.setRowStretch(self.grid.rowCount(), 1)


# ── Memory ────────────────────────────────────────────────────────────────────

class MemoryPage(QWidget):
    def __init__(self, get_memory, parent=None):
        super().__init__(parent)
        self.get_memory = get_memory
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.LG)
        root.setSpacing(S.MD)
        root.addWidget(SectionHeader(
            "Agent Memory",
            "What the agent has learned — successes it can reuse and failure "
            "patterns it now avoids."))

        cap1 = QLabel("LEARNED TASKS  (semantic memory)")
        cap1.setProperty("role", "micro")
        root.addWidget(cap1)
        self.learned_box = QVBoxLayout()
        self.learned_box.setSpacing(6)
        root.addLayout(self.learned_box)

        cap2 = QLabel("FAILURE PATTERNS  (episodic memory)")
        cap2.setProperty("role", "micro")
        root.addWidget(cap2)
        self.fail_box = QVBoxLayout()
        self.fail_box.setSpacing(6)
        root.addLayout(self.fail_box)
        root.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def refresh(self):
        for box in (self.learned_box, self.fail_box):
            while box.count():
                it = box.takeAt(0)
                if it.widget():
                    it.widget().deleteLater()
        mem = self.get_memory()
        if mem is None:
            self.learned_box.addWidget(EmptyState(
                "db", "Memory unavailable",
                "The agent's memory database could not be opened."))
            return
        try:
            tasks = mem.get_recent_tasks(limit=12)
        except Exception:
            tasks = []
        if not tasks:
            self.learned_box.addWidget(QLabel("Nothing learned yet."))
        for t in tasks:
            row = GlassCard(shadow=False)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(S.LG, S.SM, S.LG, S.SM)
            lay.addWidget(_dot(C.ACCENT2))
            name = QLabel(t["instruction"])
            name.setStyleSheet("font-size: 12px;")
            lay.addWidget(name, stretch=1)
            meta = QLabel(f"{t['success_count']}× reinforced")
            meta.setProperty("role", "faint")
            lay.addWidget(meta)
            self.learned_box.addWidget(row)

        try:
            rows = mem.conn.execute(
                "SELECT target, action_type, error, fail_count, last_seen "
                "FROM failure_patterns ORDER BY last_seen DESC LIMIT 20"
            ).fetchall()
        except Exception:
            rows = []
        if not rows:
            self.fail_box.addWidget(QLabel(
                "No failure patterns recorded — the agent hasn't needed to "
                "learn any workarounds yet."))
        for target, action, error, count, last_seen in rows:
            row = GlassCard(shadow=False)
            lay = QHBoxLayout(row)
            lay.setContentsMargins(S.LG, S.SM, S.LG, S.SM)
            lay.addWidget(_dot(C.WARNING))
            col = QVBoxLayout()
            col.setSpacing(0)
            head = QLabel(f"{action or 'action'} → “{target}”")
            head.setStyleSheet("font-size: 12px; font-weight: 600;")
            col.addWidget(head)
            if error:
                err = QLabel(error)
                err.setProperty("role", "faint")
                err.setWordWrap(True)
                col.addWidget(err)
            lay.addLayout(col, stretch=1)
            meta = QLabel(f"{count}× · {relative_time(last_seen or 0)}")
            meta.setProperty("role", "faint")
            lay.addWidget(meta)
            self.fail_box.addWidget(row)


def _dot(color: str) -> QLabel:
    lbl = QLabel()
    lbl.setPixmap(icon_pixmap("dot", QColor(color), 12))
    return lbl


# ── Screen History ────────────────────────────────────────────────────────────

class ScreenHistoryPage(QWidget):
    """Thumbnails of what the agent saw during recent missions."""

    def __init__(self, frame_store, parent=None):
        """frame_store: deque of (timestamp, QPixmap, action_text)."""
        super().__init__(parent)
        self.frame_store = frame_store
        root = QVBoxLayout(self)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.LG)
        root.setSpacing(S.MD)
        root.addWidget(SectionHeader(
            "Screen History",
            "Frames captured while the agent was operating — click to inspect."))
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(S.MD)
        self.grid.setContentsMargins(0, 0, 0, 0)
        root.addWidget(_scrollable(self.grid_host), stretch=1)

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def refresh(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        frames = list(self.frame_store)
        if not frames:
            self.grid.addWidget(EmptyState(
                "screen", "No frames yet",
                "While a mission runs, the agent's view of your screen is "
                "recorded here for review."), 0, 0)
            return
        for i, (ts, pixmap, action) in enumerate(reversed(frames)):
            card = GlassCard(hoverable=True, shadow=False)
            lay = QVBoxLayout(card)
            lay.setContentsMargins(8, 8, 8, 8)
            lay.setSpacing(6)
            thumb = QLabel()
            thumb.setPixmap(pixmap.scaledToWidth(
                300, Qt.TransformationMode.SmoothTransformation))
            thumb.setStyleSheet("border-radius: 8px;")
            lay.addWidget(thumb)
            t = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            cap = QLabel(f"{t} — {action}" if action else t)
            cap.setProperty("role", "faint")
            cap.setWordWrap(True)
            lay.addWidget(cap)
            card.clicked.connect(
                lambda px=pixmap, tt=t: self._show_full(px, tt))
            self.grid.addWidget(card, i // 3, i % 3)
        self.grid.setRowStretch(self.grid.rowCount(), 1)

    def _show_full(self, pixmap: QPixmap, title: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Frame · {title}")
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setPixmap(pixmap.scaled(
            1100, 640, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        lay.addWidget(lbl)
        dlg.exec()


# ── Settings ──────────────────────────────────────────────────────────────────

class SettingsPage(QWidget):
    # emitted from the health-check worker thread; queued back to the UI thread
    _health_ready = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._health_ready.connect(self._health_done)
        inner = QWidget()
        root = QVBoxLayout(inner)
        root.setContentsMargins(S.XL, S.LG, S.XL, S.LG)
        root.setSpacing(S.LG)
        root.addWidget(SectionHeader(
            "Settings", "Models, safety and credentials."))

        # ── Models card ───────────────────────────────────────
        models = GlassCard(shadow=False)
        ml = QVBoxLayout(models)
        ml.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        ml.setSpacing(8)
        mc = QLabel("INTELLIGENCE STACK")
        mc.setProperty("role", "micro")
        ml.addWidget(mc)
        form = QFormLayout()
        form.setHorizontalSpacing(S.XL)
        try:
            from config import LLM_MODEL, OVMS_BASE_URL, TARGET_DEVICE, VLM_MODEL
            rows = [
                ("Language model", f"{LLM_MODEL} — routing · planning · reflection"),
                ("Vision model", f"{VLM_MODEL} — grounding · visual verification"),
                ("Served by", f"OpenVINO Model Server ({OVMS_BASE_URL}) · device {TARGET_DEVICE}"),
            ]
        except Exception:
            rows = [("Configuration", "config.py could not be read")]
        for k, v in rows:
            key = QLabel(k)
            key.setProperty("role", "dim")
            val = QLabel(v)
            val.setProperty("role", "mono")
            form.addRow(key, val)
        ml.addLayout(form)
        hint = QLabel("Edit config.py to change models or ports.")
        hint.setProperty("role", "faint")
        ml.addWidget(hint)

        self.health_btn = QPushButton(" Check backend health")
        self.health_btn.setProperty("kind", "ghost")
        self.health_btn.setIcon(_qicon("cpu", C.TEXT_DIM))
        self.health_btn.clicked.connect(self._check_health)
        self.health_result = QLabel("")
        self.health_result.setProperty("role", "mono")
        hrow = QHBoxLayout()
        hrow.addWidget(self.health_btn)
        hrow.addWidget(self.health_result, stretch=1)
        ml.addLayout(hrow)
        root.addWidget(models)

        # ── Safety card ───────────────────────────────────────
        safety = GlassCard(shadow=False)
        sl = QVBoxLayout(safety)
        sl.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        sl.setSpacing(8)
        sc = QLabel("SAFETY & CONTROL")
        sc.setProperty("role", "micro")
        sl.addWidget(sc)
        for icon, color, text in (
            ("shield", C.SUCCESS,
             "Action firewall — destructive commands are classified before "
             "typing; HIGH-risk commands are blocked, MEDIUM-risk are logged."),
            ("stop", C.DANGER,
             "Kill switch — armed for the duration of every mission; the Stop "
             "button halts execution immediately."),
            ("eye", C.ACCENT,
             "Full transparency — every action is verified on-screen and "
             "shown in Mission Control with its confidence score."),
            ("key", C.WARNING,
             "Credentials are stored in the OS keychain and redacted from all "
             "logs when typed."),
        ):
            row = QHBoxLayout()
            ic = QLabel()
            ic.setPixmap(icon_pixmap(icon, QColor(color), 16))
            ic.setFixedWidth(24)
            row.addWidget(ic, alignment=Qt.AlignmentFlag.AlignTop)
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setProperty("role", "dim")
            row.addWidget(lbl, stretch=1)
            sl.addLayout(row)
        root.addWidget(safety)

        # ── Credentials card ──────────────────────────────────
        creds = GlassCard(shadow=False)
        cl = QVBoxLayout(creds)
        cl.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        cl.setSpacing(8)
        cc = QLabel("SAVED CREDENTIALS")
        cc.setProperty("role", "micro")
        cl.addWidget(cc)
        chint = QLabel(
            "Used with {{cred:site:field}} placeholders inside missions.")
        chint.setProperty("role", "faint")
        cl.addWidget(chint)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._cred_site = QLineEdit()
        self._cred_site.setPlaceholderText("site (e.g. github.com)")
        self._cred_user = QLineEdit()
        self._cred_user.setPlaceholderText("username")
        self._cred_pass = QLineEdit()
        self._cred_pass.setPlaceholderText("password")
        self._cred_pass.setEchoMode(QLineEdit.EchoMode.Password)
        save = QPushButton("Save")
        save.setProperty("kind", "primary")
        save.clicked.connect(self._save_credential)
        delete = QPushButton("Delete")
        delete.setProperty("kind", "danger")
        delete.clicked.connect(self._delete_credential)
        for w in (self._cred_site, self._cred_user, self._cred_pass,
                  save, delete):
            row.addWidget(w)
        cl.addLayout(row)
        self._cred_list = QLabel("")
        self._cred_list.setProperty("role", "mono")
        cl.addWidget(self._cred_list)
        root.addWidget(creds)
        root.addStretch()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scrollable(inner))
        self._refresh_cred_list()

    def _check_health(self):
        self.health_btn.setEnabled(False)
        self.health_result.setText("checking…")

        def worker():
            try:
                from core.pipeline.ovms_client import OVMSClient
                health = OVMSClient().check_health()
                text = "   ".join(f"{k}: {v}" for k, v in health.items())
            except Exception as e:
                text = f"unreachable: {e}"
            self._health_ready.emit(text)

        threading.Thread(target=worker, daemon=True).start()

    def _health_done(self, text: str):
        self.health_btn.setEnabled(True)
        self.health_result.setText(text)

    def _save_credential(self):
        site = self._cred_site.text().strip()
        user = self._cred_user.text().strip()
        pwd = self._cred_pass.text()
        if not site or not user:
            QMessageBox.warning(self, "Missing", "Site and username are required.")
            return
        from utils.credentials import set_credential
        set_credential(site, user, pwd)
        self._cred_site.clear()
        self._cred_user.clear()
        self._cred_pass.clear()
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
            self._cred_list.setText(
                "  ·  ".join(sites) if sites else "(no credentials stored)")
        except Exception as e:
            self._cred_list.setText(f"Error: {e}")


def _qicon(name, color):
    from PyQt6.QtGui import QIcon
    return QIcon(icon_pixmap(name, QColor(color), 16))
