# ui/widgets.py
"""
Reusable design-system components.

Performance rules (the GPU belongs to the VLM, not the UI):
  * No QGraphicsBlurEffect (software-rendered, expensive).
  * Continuous timers only run while a widget is visible AND animating.
  * Glass = translucent fill + 1px stroke; depth = one QGraphicsDropShadowEffect
    per floating card, never nested.
"""
import math
import time

from PyQt6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer,
    QVariantAnimation, pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush, QColor, QConicalGradient, QFont, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap, QRadialGradient,
)
from PyQt6.QtWidgets import (
    QFrame, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QHBoxLayout,
    QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout,
    QWidget,
)

from ui.icons import draw_icon, icon_pixmap
from ui.theme import C, S, T, qcolor


# ── PulseOrb — the agent's visual presence ───────────────────────────────────

class PulseOrb(QWidget):
    """Breathing orb with a rotating arc while the agent is busy.

    States are expressed purely through color + animation speed, so every
    surface (hero, dock, intelligence panel) reads the same body language.
    """

    def __init__(self, diameter: int = 96, parent=None):
        super().__init__(parent)
        self._d = diameter
        self.setFixedSize(diameter, diameter)
        self._color = QColor(C.ACCENT)
        self._busy = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def set_state(self, color_hex: str, busy: bool):
        self._color = QColor(color_hex)
        self._busy = busy
        self.update()

    def _tick(self):
        self._phase += 0.05 if not self._busy else 0.16
        self.update()

    def showEvent(self, e):
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        breath = 1.0 + 0.045 * math.sin(self._phase)
        r = (self._d / 2 - 6) * breath
        col = self._color

        # Outer aura
        aura = QRadialGradient(cx, cy, r * 1.45)
        aura.setColorAt(0.55, QColor(col.red(), col.green(), col.blue(), 0))
        aura.setColorAt(0.78, QColor(col.red(), col.green(), col.blue(), 36))
        aura.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.setBrush(QBrush(aura))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), r * 1.45, r * 1.45)

        # Core sphere — lit from upper-left for depth
        core = QRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 1.6)
        core.setColorAt(0.0, QColor(255, 255, 255, 215))
        core.setColorAt(0.25, col.lighter(125))
        core.setColorAt(0.75, col.darker(165))
        core.setColorAt(1.0, QColor(10, 12, 18))
        p.setBrush(QBrush(core))
        p.drawEllipse(QPointF(cx, cy), r * 0.62, r * 0.62)

        # Orbit ring — rotating arc when busy, faint static ring when idle
        ring_r = r * 0.92
        pen = QPen(QColor(col.red(), col.green(), col.blue(), 70), 1.6)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)
        if self._busy:
            grad = QConicalGradient(QPointF(cx, cy), -math.degrees(self._phase))
            grad.setColorAt(0.0, QColor(col.red(), col.green(), col.blue(), 255))
            grad.setColorAt(0.30, QColor(col.red(), col.green(), col.blue(), 0))
            grad.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            pen = QPen(QBrush(grad), 2.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            p.drawEllipse(QPointF(cx, cy), ring_r, ring_r)


# ── Glass surfaces ────────────────────────────────────────────────────────────

class GlassCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, parent=None, hoverable: bool = False, shadow: bool = True):
        super().__init__(parent)
        self.setProperty("hoverable", "true" if hoverable else "false")
        if shadow:
            eff = QGraphicsDropShadowEffect(self)
            eff.setBlurRadius(28)
            eff.setOffset(0, 8)
            eff.setColor(QColor(0, 0, 0, 110))
            self.setGraphicsEffect(eff)
        if hoverable:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, e):
        if self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


def fade_in(widget: QWidget, duration: int = 260):
    eff = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    # Drop the effect when done — stacked opacity effects slow painting.
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


# ── Status chip ───────────────────────────────────────────────────────────────

class StatusChip(QWidget):
    """Pill with a glowing dot + label; color animates between states."""

    def __init__(self, text: str = "Ready", color: str = C.TEXT_DIM, parent=None):
        super().__init__(parent)
        self._text = text
        self._color = QColor(color)
        self._anim = None
        self.setFixedHeight(28)
        self.setMinimumWidth(110)

    def set_state(self, text: str, color: str):
        self._text = text
        target = QColor(color)
        if self._anim:
            self._anim.stop()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(350)
        self._anim.setStartValue(self._color)
        self._anim.setEndValue(target)
        self._anim.valueChanged.connect(self._on_color)
        self._anim.start()
        self.update()

    def _on_color(self, value):
        self._color = value
        self.update()

    def sizeHint(self):
        fm = self.fontMetrics()
        return QSize(fm.horizontalAdvance(self._text) + 48, 28)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self._color
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), 80), 1))
        p.setBrush(QColor(c.red(), c.green(), c.blue(), 22))
        p.drawRoundedRect(rect, 14, 14)
        # glowing dot
        glow = QRadialGradient(16, self.height() / 2, 8)
        glow.setColorAt(0, c)
        glow.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(16, self.height() / 2), 7, 7)
        p.setBrush(c)
        p.drawEllipse(QPointF(16, self.height() / 2), 3, 3)
        p.setPen(QColor(C.TEXT))
        f = p.font()
        f.setPointSizeF(8.5)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.drawText(QRectF(28, 0, self.width() - 34, self.height()),
                   Qt.AlignmentFlag.AlignVCenter, self._text)


# ── Navigation rail ───────────────────────────────────────────────────────────

class NavItem(QPushButton):
    def __init__(self, icon_name: str, label: str, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.label_text = label
        self.setCheckable(False)
        self.setFixedHeight(42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QSize(20, 20))
        self.set_active(False)
        self.setToolTip(label)

    def set_active(self, active: bool):
        self.setProperty("active", "true" if active else "false")
        color = QColor(C.ACCENT if active else C.TEXT_DIM)
        self.setIcon(_qicon(self.icon_name, color))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_expanded(self, expanded: bool):
        self.setText(("  " + self.label_text) if expanded else "")


def _qicon(name: str, color: QColor):
    from PyQt6.QtGui import QIcon
    return QIcon(icon_pixmap(name, color, 20))


class NavRail(QWidget):
    """Compact icon rail that expands on hover (64px ↔ 196px)."""

    navigate = pyqtSignal(int)
    COLLAPSED, EXPANDED = 64, 196

    def __init__(self, items, parent=None):
        """items: list of (icon_name, label)."""
        super().__init__(parent)
        self.setFixedWidth(self.COLLAPSED)
        self._expanded = False
        self._anim = QPropertyAnimation(self, b"minimumWidth", self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim2 = QPropertyAnimation(self, b"maximumWidth", self)
        self._anim2.setDuration(220)
        self._anim2.setEasingCurve(QEasingCurve.Type.OutCubic)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 14, 10, 14)
        lay.setSpacing(4)

        self._brand = QLabel()
        self._brand.setPixmap(icon_pixmap("sparkle", QColor(C.ACCENT), 24))
        self._brand.setFixedHeight(40)
        self._brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._brand)
        lay.addSpacing(8)

        self.items = []
        for i, (icon, label) in enumerate(items):
            item = NavItem(icon, label)
            item.clicked.connect(lambda _, idx=i: self._on_click(idx))
            lay.addWidget(item)
            self.items.append(item)
        lay.addStretch()

        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(160)
        self._hover_timer.timeout.connect(lambda: self._set_expanded(True))

    def _on_click(self, idx: int):
        self.set_active(idx)
        self.navigate.emit(idx)

    def set_active(self, idx: int):
        for i, item in enumerate(self.items):
            item.set_active(i == idx)

    def _set_expanded(self, expanded: bool):
        if expanded == self._expanded:
            return
        self._expanded = expanded
        target = self.EXPANDED if expanded else self.COLLAPSED
        for anim in (self._anim, self._anim2):
            anim.stop()
            anim.setStartValue(self.width())
            anim.setEndValue(target)
            anim.start()
        for item in self.items:
            item.set_expanded(expanded)

    def enterEvent(self, e):
        self._hover_timer.start()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover_timer.stop()
        self._set_expanded(False)
        super().leaveEvent(e)


# ── Metric tile ───────────────────────────────────────────────────────────────

class MetricTile(GlassCard):
    def __init__(self, caption: str, icon: str = "bolt", accent: str = C.ACCENT,
                 parent=None):
        super().__init__(parent, shadow=False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(S.LG, S.MD, S.LG, S.MD)
        lay.setSpacing(2)
        top = QHBoxLayout()
        cap = QLabel(caption.upper())
        cap.setProperty("role", "micro")
        ic = QLabel()
        ic.setPixmap(icon_pixmap(icon, QColor(accent), 15))
        top.addWidget(cap)
        top.addStretch()
        top.addWidget(ic)
        lay.addLayout(top)
        self.value_label = QLabel("—")
        self.value_label.setProperty("role", "metric")
        lay.addWidget(self.value_label)

    def set_value(self, text: str):
        self.value_label.setText(text)


# ── Confidence bar ────────────────────────────────────────────────────────────

class ConfidenceBar(QWidget):
    """Thin animated bar; color follows the value (red→amber→cyan→green)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._shown = 0.0
        self.setFixedHeight(8)
        self._anim = None

    def set_value(self, v: float):
        v = max(0.0, min(1.0, v))
        if self._anim:
            self._anim.stop()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(420)
        self._anim.setStartValue(self._shown)
        self._anim.setEndValue(v)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.start()
        self._value = v

    def _on_anim(self, v):
        self._shown = v
        self.update()

    @staticmethod
    def color_for(v: float) -> QColor:
        if v < 0.5:
            return QColor(C.DANGER)
        if v < 0.75:
            return QColor(C.WARNING)
        if v < 0.9:
            return QColor(C.ACCENT)
        return QColor(C.SUCCESS)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(0, self.height() / 2 - 2, self.width(), 4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 18))
        p.drawRoundedRect(track, 2, 2)
        if self._shown > 0.01:
            c = self.color_for(self._shown)
            fill = QRectF(0, self.height() / 2 - 2,
                          self.width() * self._shown, 4)
            grad = QLinearGradient(0, 0, fill.width(), 0)
            grad.setColorAt(0, c.darker(130))
            grad.setColorAt(1, c)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(fill, 2, 2)


# ── Skeleton shimmer ──────────────────────────────────────────────────────────

class Skeleton(QWidget):
    def __init__(self, height: int = 16, radius: int = 6, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self._radius = radius
        self._offset = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        self._offset = (self._offset + 0.03) % 1.5
        self.update()

    def showEvent(self, e):
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        grad = QLinearGradient((self._offset - 0.5) * w, 0,
                               (self._offset + 0.25) * w, 0)
        grad.setColorAt(0.0, QColor(255, 255, 255, 14))
        grad.setColorAt(0.5, QColor(255, 255, 255, 34))
        grad.setColorAt(1.0, QColor(255, 255, 255, 14))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(QRectF(0, 0, w, self.height()),
                          self._radius, self._radius)


# ── Timeline (agent execution feed) ───────────────────────────────────────────

class TimelineStep(QFrame):
    """One row: status glyph | action chip | description | confidence."""

    PENDING, RUNNING, OK, FAIL, INFO = range(5)

    def __init__(self, action_type: str, description: str, parent=None):
        super().__init__(parent)
        self.status = self.RUNNING
        self.action_type = action_type
        self._pulse_phase = 0.0
        self.setFixedHeight(40)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(34, 0, 10, 0)
        lay.setSpacing(10)

        hue = C.ACTION_HUES.get(action_type, C.TEXT_DIM)
        hc = QColor(hue)
        chip = QLabel(action_type.replace("_", " ").upper())
        chip.setStyleSheet(
            f"color: {hue}; background: rgba(255,255,255,10);"
            f"border: 1px solid rgba({hc.red()},{hc.green()},{hc.blue()},64);"
            f"border-radius: 9px;"
            f"padding: 2px 8px; font-size: 10px; font-weight: 700;"
            f"letter-spacing: 0.5px;"
        )
        lay.addWidget(chip)

        self.desc = QLabel(description)
        self.desc.setStyleSheet(f"color: {C.TEXT}; font-size: 13px;")
        self.desc.setSizePolicy(QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Preferred)
        lay.addWidget(self.desc, stretch=1)

        self.meta = QLabel("")
        self.meta.setStyleSheet(f"color: {C.TEXT_FAINT}; font-size: 11px;")
        lay.addWidget(self.meta)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._pulse_phase += 0.18
        self.update()

    def set_ok(self, conf: float):
        self.status = self.OK
        self._timer.stop()
        self.meta.setText(f"conf {conf:.2f}")
        self.meta.setStyleSheet(f"color: {C.SUCCESS}; font-size: 11px;")
        self.update()

    def set_fail(self, reason: str):
        self.status = self.FAIL
        self._timer.stop()
        self.meta.setText("failed")
        self.meta.setStyleSheet(f"color: {C.DANGER}; font-size: 11px;")
        if reason:
            self.setToolTip(reason)
        self.update()

    def set_retry(self, attempt: int, total: int):
        self.status = self.RUNNING
        if not self._timer.isActive():
            self._timer.start()
        self.meta.setText(f"retry {attempt}/{total}")
        self.meta.setStyleSheet(f"color: {C.WARNING}; font-size: 11px;")

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = 16.0, self.height() / 2
        # connector line
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.drawLine(QPointF(cx, 0), QPointF(cx, cy - 9))
        p.drawLine(QPointF(cx, cy + 9), QPointF(cx, self.height()))
        if self.status == self.RUNNING:
            c = QColor(C.ACCENT)
            r = 4.5 + 1.5 * math.sin(self._pulse_phase)
            glow = QRadialGradient(cx, cy, r * 2.2)
            glow.setColorAt(0, QColor(c.red(), c.green(), c.blue(), 90))
            glow.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            p.drawEllipse(QPointF(cx, cy), r * 2.2, r * 2.2)
            p.setBrush(c)
            p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
        elif self.status == self.OK:
            draw_icon(p, "check", QRectF(cx - 7, cy - 7, 14, 14), QColor(C.SUCCESS))
        elif self.status == self.FAIL:
            draw_icon(p, "x", QRectF(cx - 7, cy - 7, 14, 14), QColor(C.DANGER))
        else:
            p.setPen(QPen(QColor(C.TEXT_FAINT), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), 4, 4)


class TimelineGuard(QFrame):
    """Inline system event (firewall, vision escalation, goal check…)."""

    KIND_STYLE = {
        "FIREWALL": (C.DANGER, "shield"),
        "VISION":   (C.ACCENT2, "eye"),
        "SEARCH":   (C.INFO, "refresh"),
        "VERIFY":   (C.INFO, "eye"),
        "GUARD":    (C.WARNING, "shield"),
    }

    def __init__(self, kind: str, message: str, parent=None):
        super().__init__(parent)
        color, icon = self.KIND_STYLE.get(kind, (C.TEXT_DIM, "dot"))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(34, 2, 10, 2)
        lay.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(icon_pixmap(icon, QColor(color), 13))
        lay.addWidget(ic)
        txt = QLabel(f"{kind} · {message}")
        txt.setWordWrap(True)
        txt.setStyleSheet(f"color: {color}; font-size: 11px;")
        lay.addWidget(txt, stretch=1)


class SubtaskHeader(QFrame):
    def __init__(self, sid: int, description: str, parent=None):
        super().__init__(parent)
        self.sid = sid
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 10, 10, 4)
        lay.setSpacing(8)
        self.badge = QLabel(f"{sid}")
        self.badge.setFixedSize(20, 20)
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._style_badge(C.ACCENT2)
        lay.addWidget(self.badge)
        t = QLabel(description)
        t.setStyleSheet(f"color: {C.TEXT}; font-size: 13px; font-weight: 600;")
        t.setWordWrap(True)
        lay.addWidget(t, stretch=1)
        self.state = QLabel("in progress")
        self.state.setStyleSheet(f"color: {C.ACCENT2}; font-size: 11px;")
        lay.addWidget(self.state)

    def _style_badge(self, color):
        self.badge.setStyleSheet(
            f"background: {color}30; color: {color}; border-radius: 10px;"
            f"font-size: 11px; font-weight: 700;"
        )

    def set_done(self, success: bool):
        color = C.SUCCESS if success else C.DANGER
        self._style_badge(color)
        self.state.setText("complete" if success else "failed")
        self.state.setStyleSheet(f"color: {color}; font-size: 11px;")


class Timeline(QScrollArea):
    """Scrolling execution feed: subtask headers, steps, guard events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._host = QWidget()
        self._lay = QVBoxLayout(self._host)
        self._lay.setContentsMargins(4, 4, 4, 4)
        self._lay.setSpacing(0)
        self._lay.addStretch()
        self.setWidget(self._host)
        self._headers = {}
        self._current_step = None

    def clear(self):
        while self._lay.count() > 1:
            item = self._lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._headers = {}
        self._current_step = None

    def _add(self, w: QWidget):
        self._lay.insertWidget(self._lay.count() - 1, w)
        fade_in(w)
        QTimer.singleShot(30, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()))

    def add_subtask(self, sid: int, description: str):
        h = SubtaskHeader(sid, description)
        self._headers[sid] = h
        self._add(h)

    def finish_subtask(self, sid: int, success: bool):
        if sid in self._headers:
            self._headers[sid].set_done(success)

    def add_step(self, action_type: str, description: str):
        step = TimelineStep(action_type, description)
        self._current_step = step
        self._add(step)

    def verify_step(self, conf: float):
        if self._current_step:
            self._current_step.set_ok(conf)

    def fail_step(self, reason: str):
        if self._current_step:
            self._current_step.set_fail(reason)

    def retry_step(self, attempt: int, total: int):
        if self._current_step:
            self._current_step.set_retry(attempt, total)

    def add_guard(self, kind: str, message: str):
        self._add(TimelineGuard(kind, message))


# ── Live screen preview ───────────────────────────────────────────────────────

class ScreenPreview(QWidget):
    """Rounded live view of the desktop with a scan sweep while the agent acts.

    The sweep tells the user "the agent is looking at this" without any fake
    bounding boxes — we only visualize what we actually know.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._active = False
        self._sweep = 0.0
        self._action_text = ""
        self.setMinimumSize(480, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)

    def set_frame(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self.update()

    def set_active(self, active: bool):
        self._active = active
        if active and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()
            self.update()

    def set_action_text(self, text: str):
        self._action_text = text
        self.update()

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def showEvent(self, e):
        if self._active:
            self._timer.start()
        super().showEvent(e)

    def _tick(self):
        self._sweep = (self._sweep + 0.012) % 1.4
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(0, 0, self.width(), self.height())
        path = QPainterPath()
        path.addRoundedRect(rect, S.RADIUS_LG, S.RADIUS_LG)
        p.setClipPath(path)

        p.fillRect(rect, QColor(4, 6, 10))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            x = (self.width() - scaled.width()) / 2
            y = (self.height() - scaled.height()) / 2
            p.drawPixmap(int(x), int(y), scaled)
        else:
            p.setPen(QColor(C.TEXT_FAINT))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                       "Awaiting first frame…")

        if self._active:
            # vertical scan sweep
            sx = (self._sweep - 0.2) * self.width()
            grad = QLinearGradient(sx - 90, 0, sx + 12, 0)
            ac = QColor(C.ACCENT)
            grad.setColorAt(0.0, QColor(ac.red(), ac.green(), ac.blue(), 0))
            grad.setColorAt(0.9, QColor(ac.red(), ac.green(), ac.blue(), 34))
            grad.setColorAt(1.0, QColor(ac.red(), ac.green(), ac.blue(), 90))
            p.fillRect(QRectF(sx - 90, 0, 102, self.height()), QBrush(grad))

        # border + badges drawn over content
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5),
                          S.RADIUS_LG, S.RADIUS_LG)

        # LIVE badge
        badge = QRectF(12, 12, 64, 22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(badge, 11, 11)
        dot_color = QColor(C.DANGER if self._active else C.TEXT_FAINT)
        p.setBrush(dot_color)
        p.drawEllipse(QPointF(24, 23), 3.5, 3.5)
        p.setPen(QColor(C.TEXT))
        f = p.font()
        f.setPointSizeF(8)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        p.drawText(badge.adjusted(20, 0, 0, 0),
                   Qt.AlignmentFlag.AlignVCenter, "LIVE")

        # current action chip (bottom)
        if self._action_text:
            fm = p.fontMetrics()
            text = self._action_text
            tw = min(fm.horizontalAdvance(text) + 36, self.width() - 24)
            chip = QRectF((self.width() - tw) / 2, self.height() - 34, tw, 24)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(chip, 12, 12)
            p.setBrush(QColor(C.ACCENT))
            p.drawEllipse(QPointF(chip.left() + 13, chip.center().y()), 3, 3)
            p.setPen(QColor(C.TEXT))
            p.drawText(chip.adjusted(24, 0, -10, 0),
                       Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(text, Qt.TextElideMode.ElideRight,
                                     int(tw) - 36))


# ── Command dock ──────────────────────────────────────────────────────────────

class CommandInput(QPlainTextEdit):
    """Auto-growing input (1–4 lines). Enter submits, Shift+Enter = newline."""

    submitted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(
            "Tell the agent what to do…   e.g. “Open Notepad and write a haiku”")
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTabChangesFocus(True)
        self.document().documentLayout().documentSizeChanged.connect(
            self._adjust_height)
        self._adjust_height()

    def _adjust_height(self, *_):
        line_h = self.fontMetrics().lineSpacing()
        lines = max(1, min(4, self.document().blockCount()))
        self.setFixedHeight(int(line_h * lines + 18))

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class CommandDock(GlassCard):
    """Persistent operator bar: orb · input · stop · run."""

    run_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 10, 8)
        lay.setSpacing(10)

        self.orb = PulseOrb(34)
        lay.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignVCenter)

        self.input = CommandInput()
        self.input.submitted.connect(self.run_requested)
        lay.addWidget(self.input, stretch=1)

        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.setProperty("kind", "danger")
        self.stop_btn.setIcon(_qicon("stop", QColor(C.DANGER)))
        self.stop_btn.clicked.connect(self.stop_requested)
        self.stop_btn.setVisible(False)
        lay.addWidget(self.stop_btn)

        self.run_btn = QPushButton(" Run")
        self.run_btn.setProperty("kind", "primary")
        self.run_btn.setIcon(_qicon("play", QColor("#06121A")))
        self.run_btn.setMinimumHeight(38)
        self.run_btn.clicked.connect(self.run_requested)
        lay.addWidget(self.run_btn)

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.input.setReadOnly(running)


# ── Misc helpers ──────────────────────────────────────────────────────────────

class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        t = QLabel(title)
        t.setProperty("role", "h1")
        lay.addWidget(t)
        if subtitle:
            s = QLabel(subtitle)
            s.setProperty("role", "dim")
            lay.addWidget(s)


class EmptyState(QWidget):
    """Engaging zero-data state: glow icon, message, quick-action chips."""

    action_clicked = pyqtSignal(str)

    def __init__(self, icon: str, title: str, subtitle: str,
                 actions=(), parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(S.MD)
        ic = QLabel()
        ic.setPixmap(icon_pixmap(icon, QColor(C.ACCENT2), 52))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ic)
        t = QLabel(title)
        t.setProperty("role", "h2")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        s = QLabel(subtitle)
        s.setProperty("role", "dim")
        s.setAlignment(Qt.AlignmentFlag.AlignCenter)
        s.setWordWrap(True)
        lay.addWidget(s)
        if actions:
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addStretch()
            for label in actions:
                chip = QPushButton(label)
                chip.setProperty("kind", "chip")
                chip.setCursor(Qt.CursorShape.PointingHandCursor)
                chip.clicked.connect(
                    lambda _, txt=label: self.action_clicked.emit(txt))
                row.addWidget(chip)
            row.addStretch()
            lay.addLayout(row)


def relative_time(ts: float) -> str:
    delta = time.time() - ts
    if delta < 90:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)} h ago"
    return f"{int(delta // 86400)} d ago"
