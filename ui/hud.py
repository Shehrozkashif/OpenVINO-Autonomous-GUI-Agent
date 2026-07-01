# ui/hud.py
"""MissionHUD — compact always-on-top status pill shown while a mission runs.

The main window minimizes during execution (it must not cover the screen the
agent is operating), which previously left the user staring at nothing for
minutes. The HUD keeps the agent's state visible the whole time:

    ◉  Locating element          41.2s   ■
       Looking for “Brave Browser” on screen…

Safety: the HUD's own pixels are registered in the shared ScreenCapture's
`persistent_exclude_regions`, so the agent's OCR/planning/verification never
see the HUD text (it would otherwise contaminate them — e.g. reflection
reading the step description off the HUD and declaring success).
"""
import time

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.events import BUSY_STATES, AgentEventBus, AgentState
from ui.icons import icon_pixmap
from ui.theme import STATE_STYLE, C
from ui.widgets import PulseOrb

_MASK_MARGIN = 8  # px safety margin around the HUD in the capture mask


class MissionHUD(QWidget):
    WIDTH, HEIGHT = 400, 84

    def __init__(self, bus: AgentEventBus, parent=None):
        super().__init__(parent)
        self.bus = bus
        self._capturer = None
        self._t0 = None
        self._drag_offset = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 12, 12, 12)
        lay.setSpacing(12)

        self.orb = PulseOrb(40)
        lay.addWidget(self.orb)

        col = QVBoxLayout()
        col.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.state_label = QLabel("Working…")
        self.state_label.setStyleSheet(
            f"color: {C.TEXT}; font-size: 14px; font-weight: 700;")
        top.addWidget(self.state_label)
        top.addStretch()
        self.elapsed = QLabel("")
        self.elapsed.setStyleSheet(
            f"color: {C.TEXT_FAINT}; font-size: 11px;"
            f"font-family: Consolas, monospace;")
        top.addWidget(self.elapsed)
        col.addLayout(top)
        self.detail = QLabel("")
        self.detail.setStyleSheet(f"color: {C.TEXT_DIM}; font-size: 11px;")
        col.addWidget(self.detail)
        lay.addLayout(col, stretch=1)

        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(QIcon(icon_pixmap("stop", QColor(C.DANGER), 16)))
        self.stop_btn.setToolTip("Stop the mission")
        self.stop_btn.setFixedSize(32, 32)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setStyleSheet(
            "QPushButton { background: rgba(248,113,110,26);"
            "border: 1px solid rgba(248,113,110,70); border-radius: 16px; }"
            "QPushButton:hover { background: rgba(248,113,110,60); }")
        lay.addWidget(self.stop_btn)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

        bus.state_changed.connect(self._on_state)
        bus.detail.connect(self._on_detail)

    # ── Mission lifecycle ─────────────────────────────────────────────────────

    def attach_capturer(self, capturer):
        """Shared ScreenCapture used by the pipeline — for HUD self-masking."""
        self._capturer = capturer

    def show_mission(self):
        self._t0 = time.time()
        self.detail.setText("Starting…")
        self._place_bottom_right()
        self.show()
        self._timer.start()
        self._update_mask()

    def hide_mission(self):
        self._timer.stop()
        self._clear_mask()
        self.hide()

    def _place_bottom_right(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(geo.right() - self.width() - 16,
                  geo.bottom() - self.height() - 16)

    # ── Capture self-masking ──────────────────────────────────────────────────

    def _update_mask(self):
        if self._capturer is None or not self.isVisible():
            return
        dpr = self.devicePixelRatioF()
        g = self.frameGeometry()
        self._capturer.persistent_exclude_regions = [(
            int((g.left() - _MASK_MARGIN) * dpr),
            int((g.top() - _MASK_MARGIN) * dpr),
            int((g.right() + _MASK_MARGIN) * dpr),
            int((g.bottom() + _MASK_MARGIN) * dpr),
        )]

    def _clear_mask(self):
        if self._capturer is not None:
            self._capturer.persistent_exclude_regions = []

    def moveEvent(self, e):
        self._update_mask()
        super().moveEvent(e)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_state(self, state: AgentState):
        color, label = STATE_STYLE[state.value]
        self.orb.set_state(color, state in BUSY_STATES)
        self.state_label.setText(label)
        self.state_label.setStyleSheet(
            f"color: {color}; font-size: 14px; font-weight: 700;")

    def _on_detail(self, text: str):
        fm = self.detail.fontMetrics()
        self.detail.setText(fm.elidedText(
            text, Qt.TextElideMode.ElideRight, self.WIDTH - 130))
        self.detail.setToolTip(text)

    def _tick(self):
        if self._t0:
            self.elapsed.setText(f"{time.time() - self._t0:5.1f}s")

    # ── Painting / dragging ───────────────────────────────────────────────────

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setBrush(QColor(13, 16, 24, 242))
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawRoundedRect(rect, 18, 18)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = \
                e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._drag_offset is not None and \
                e.buttons() & Qt.MouseButton.LeftButton:
            self.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, e):
        self._drag_offset = None
