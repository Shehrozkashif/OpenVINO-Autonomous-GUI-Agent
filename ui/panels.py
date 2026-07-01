# ui/panels.py
"""IntelligencePanel — the right-hand "agent mind" column.

Shows what the agent is thinking, doing and how confident it is. Everything
here is driven by AgentEventBus signals; nothing polls.
"""
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.events import BUSY_STATES, AgentEventBus, AgentState
from ui.icons import icon_pixmap
from ui.theme import STATE_STYLE, C, S
from ui.widgets import ConfidenceBar, GlassCard, PulseOrb, fade_in


class _FeedItem(QLabel):
    COLORS = {
        "plan":     C.ACCENT2,
        "act":      C.ACCENT,
        "ok":       C.SUCCESS,
        "fail":     C.DANGER,
        "guard":    C.WARNING,
        "memory":   C.INFO,
        "extract":  C.WARNING,
    }

    def __init__(self, kind: str, text: str, parent=None):
        super().__init__(text, parent)
        color = self.COLORS.get(kind, C.TEXT_DIM)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"color: {C.TEXT_DIM}; font-size: 12px; padding: 4px 8px;"
            f"border-left: 2px solid {color}; background: rgba(255,255,255,5);"
            f"border-radius: 4px; margin: 0px;"
        )


class IntelligencePanel(QWidget):
    WIDTH = 312

    def __init__(self, bus: AgentEventBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self.setFixedWidth(self.WIDTH)

        root = QVBoxLayout(self)
        root.setContentsMargins(S.MD, S.LG, S.LG, S.LG)
        root.setSpacing(S.MD)

        # ── State header ──────────────────────────────────────
        header = GlassCard(shadow=False)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(S.MD, S.MD, S.MD, S.MD)
        self.orb = PulseOrb(46)
        hl.addWidget(self.orb)
        col = QVBoxLayout()
        col.setSpacing(0)
        cap = QLabel("AGENT STATE")
        cap.setProperty("role", "micro")
        col.addWidget(cap)
        self.state_label = QLabel("Ready")
        self.state_label.setProperty("role", "h2")
        col.addWidget(self.state_label)
        hl.addLayout(col, stretch=1)
        root.addWidget(header)

        # ── Current objective ─────────────────────────────────
        obj_card = GlassCard(shadow=False)
        ol = QVBoxLayout(obj_card)
        ol.setContentsMargins(S.MD, S.MD, S.MD, S.MD)
        ol.setSpacing(4)
        ocap = QLabel("CURRENT OBJECTIVE")
        ocap.setProperty("role", "micro")
        ol.addWidget(ocap)
        self.objective = QLabel("Waiting for a mission.")
        self.objective.setWordWrap(True)
        self.objective.setStyleSheet("font-size: 13px;")
        ol.addWidget(self.objective)
        root.addWidget(obj_card)

        # ── Confidence ────────────────────────────────────────
        conf_card = GlassCard(shadow=False)
        cl = QVBoxLayout(conf_card)
        cl.setContentsMargins(S.MD, S.MD, S.MD, S.MD)
        cl.setSpacing(6)
        crow = QHBoxLayout()
        ccap = QLabel("VERIFICATION CONFIDENCE")
        ccap.setProperty("role", "micro")
        crow.addWidget(ccap)
        crow.addStretch()
        self.conf_value = QLabel("-")
        self.conf_value.setStyleSheet(
            f"color: {C.TEXT_DIM}; font-size: 12px; font-weight: 700;")
        crow.addWidget(self.conf_value)
        cl.addLayout(crow)
        self.conf_bar = ConfidenceBar()
        cl.addWidget(self.conf_bar)
        root.addWidget(conf_card)

        # ── Detected elements (live grounding results) ────────
        el_card = GlassCard(shadow=False)
        el = QVBoxLayout(el_card)
        el.setContentsMargins(S.MD, S.MD, S.MD, S.MD)
        el.setSpacing(4)
        ecap = QLabel("DETECTED ELEMENTS")
        ecap.setProperty("role", "micro")
        el.addWidget(ecap)
        self.elements_box = QVBoxLayout()
        self.elements_box.setSpacing(2)
        el.addLayout(self.elements_box)
        self._no_elements = QLabel("None located yet.")
        self._no_elements.setProperty("role", "faint")
        self.elements_box.addWidget(self._no_elements)
        root.addWidget(el_card)

        # ── Activity feed ─────────────────────────────────────
        feed_cap = QLabel("REASONING & ACTIVITY")
        feed_cap.setProperty("role", "micro")
        root.addWidget(feed_cap)

        self.feed_scroll = QScrollArea()
        self.feed_scroll.setWidgetResizable(True)
        self.feed_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feed_host = QWidget()
        self.feed_lay = QVBoxLayout(feed_host)
        self.feed_lay.setContentsMargins(0, 0, 4, 0)
        self.feed_lay.setSpacing(6)
        self.feed_lay.addStretch()
        self.feed_scroll.setWidget(feed_host)
        root.addWidget(self.feed_scroll, stretch=1)

        # ── Raw console (collapsible) ─────────────────────────
        self.console_toggle = QPushButton("  Raw agent log")
        self.console_toggle.setProperty("kind", "ghost")
        self.console_toggle.setIcon(
            _icon("chevron-right", C.TEXT_DIM))
        self.console_toggle.setCheckable(True)
        self.console_toggle.toggled.connect(self._toggle_console)
        root.addWidget(self.console_toggle)

        self.console = QPlainTextEdit()
        self.console.setProperty("role", "console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setMinimumHeight(0)
        self.console.setMaximumHeight(0)
        root.addWidget(self.console)

        self._wire(bus)

    # ── Event wiring ──────────────────────────────────────────

    def _wire(self, bus: AgentEventBus):
        bus.state_changed.connect(self.on_state)
        bus.task_started.connect(lambda instr: self._feed("plan", f"Mission: {instr}"))
        bus.plan_ready.connect(lambda n: self._feed("plan", f"Plan ready - {n} subtask(s)"))
        bus.subtask_started.connect(self._on_subtask)
        bus.step_started.connect(
            lambda n, a, d: self._feed("act", f"{a.replace('_', ' ')} -> {d}"))
        bus.step_verified.connect(self._on_verified)
        bus.step_failed.connect(self._on_failed)
        bus.retrying.connect(
            lambda a, t: self._feed("guard", f"Retrying ({a}/{t})"))
        bus.guard_event.connect(
            lambda k, m: self._feed("guard", f"{k}: {m}"))
        bus.extracted.connect(
            lambda k, v: self._feed("extract", f"Extracted {k} = '{v}'"))
        bus.memory_hint.connect(
            lambda sim: self._feed(
                "memory", f"Recognized a similar past task ({sim:.0%} match)"))
        bus.task_done.connect(
            lambda summary, s: self._feed("ok", f"{summary} ({s:.1f}s)"))
        bus.element_located.connect(self._on_element)
        bus.raw_line.connect(self._append_console)

    def on_state(self, state: AgentState):
        color, label = STATE_STYLE[state.value]
        self.orb.set_state(color, state in BUSY_STATES)
        self.state_label.setText(label)
        self.state_label.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: 600;")

    def _on_subtask(self, sid: int, desc: str):
        self.objective.setText(desc)
        self._feed("plan", f"Subtask {sid}: {desc}")

    def _on_verified(self, conf: float):
        self.conf_bar.set_value(conf)
        self.conf_value.setText(f"{conf:.0%}")
        self.conf_value.setStyleSheet(
            f"color: {ConfidenceBar.color_for(conf).name()};"
            f"font-size: 12px; font-weight: 700;")
        self._feed("ok", f"Step verified | confidence {conf:.0%}")

    def _on_failed(self, reason: str, conf: float):
        self.conf_bar.set_value(conf)
        self.conf_value.setText(f"{conf:.0%}")
        self._feed("fail", f"Step failed: {reason}")

    def _on_element(self, target: str, x: int, y: int, conf: float,
                    method: str):
        self._no_elements.hide()
        row = QLabel(f"'{target}'  |  {conf:.0%} {method}  |  ({x},{y})")
        row.setStyleSheet(
            f"color: {C.TEXT_DIM}; font-size: 11px;"
            f"font-family: Consolas, monospace;")
        row.setToolTip(f"{target} located at ({x},{y}) "
                       f"via {method}, confidence {conf:.0%}")
        self.elements_box.addWidget(row)
        while self.elements_box.count() > 5:  # keep placeholder + last 4
            it = self.elements_box.takeAt(1)
            if it.widget():
                it.widget().deleteLater()

    def _feed(self, kind: str, text: str):
        item = _FeedItem(kind, text)
        self.feed_lay.insertWidget(self.feed_lay.count() - 1, item)
        fade_in(item, 200)
        # cap memory: keep last 80 entries
        while self.feed_lay.count() > 81:
            it = self.feed_lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        sb = self.feed_scroll.verticalScrollBar()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(30, lambda: sb.setValue(sb.maximum()))

    def clear_mission(self):
        self.objective.setText("Waiting for a mission.")
        self.conf_bar.set_value(0)
        self.conf_value.setText("-")
        while self.elements_box.count() > 1:
            it = self.elements_box.takeAt(1)
            if it.widget():
                it.widget().deleteLater()
        self._no_elements.show()

    def _append_console(self, line: str):
        self.console.appendPlainText(line)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _toggle_console(self, open_: bool):
        anim = QPropertyAnimation(self.console, b"maximumHeight", self)
        anim.setDuration(220)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(self.console.maximumHeight())
        anim.setEndValue(180 if open_ else 0)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self.console_toggle.setIcon(
            _icon("chevron-left" if open_ else "chevron-right", C.TEXT_DIM))


def _icon(name, color):
    from PyQt6.QtGui import QIcon
    return QIcon(icon_pixmap(name, QColor(color), 14))
