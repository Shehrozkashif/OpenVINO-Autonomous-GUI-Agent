# ui/click_pulse.py
"""On-screen click indicator — visible to the human, invisible to the agent.

A frameless, click-through, always-on-top ring that pulses wherever the agent
clicks, so the user can follow the mouse work in real time.

Excluded from screen capture via SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE):
the agent's own screenshots (phash delta checks, OCR, VLM verification) must
never see the indicator — otherwise the pulse itself would register as a
screen change and corrupt dead-click detection.
"""
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import QWidget

_SIZE = 120          # widget square, physical px
_DURATION_MS = 650   # pulse lifetime
_TICK_MS = 16        # ~60 fps

_COLORS = {
    "left": QColor(0, 170, 255),     # blue ring — normal click
    "right": QColor(255, 170, 0),    # orange — right click
    "double": QColor(0, 220, 130),   # green — double click
}


class ClickPulse(QWidget):
    """One reusable pulse window; flash(x, y) repositions and replays it."""

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(_SIZE, _SIZE)
        self._color = _COLORS["left"]
        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def flash(self, x: int, y: int, kind: str = "left"):
        """Show the pulse centered on PHYSICAL screen pixel (x, y)."""
        self._color = _COLORS.get(kind, _COLORS["left"])
        self._elapsed = 0
        # Qt positions windows in device-independent pixels; the agent's
        # coordinates are physical. Identical at 100 % scale, divided by the
        # devicePixelRatio otherwise.
        screen = self.screen() or QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen else 1.0
        self.move(int(x / dpr) - _SIZE // 2, int(y / dpr) - _SIZE // 2)
        self.show()
        self._exclude_from_capture()
        self._timer.start(_TICK_MS)

    def _tick(self):
        self._elapsed += _TICK_MS
        if self._elapsed >= _DURATION_MS:
            self._timer.stop()
            self.hide()
            return
        self.update()

    def paintEvent(self, _ev):
        frac = min(self._elapsed / _DURATION_MS, 1.0)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = _SIZE / 2
        # Expanding, fading ring
        ring = QColor(self._color)
        ring.setAlphaF(1.0 - frac)
        p.setPen(QPen(ring, 3.0))
        r = 8 + frac * (_SIZE / 2 - 10)
        p.drawEllipse(QRectF(c - r, c - r, 2 * r, 2 * r))
        # Center dot marking the exact click pixel
        dot = QColor(self._color)
        dot.setAlphaF(0.9 * (1.0 - frac))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(dot)
        p.drawEllipse(QRectF(c - 4, c - 4, 8, 8))
        p.end()

    def _exclude_from_capture(self):
        """Hide this window from BitBlt/ImageGrab (Windows 10 2004+).

        Re-applied after every show() because Qt may recreate the native
        window. Best-effort: on older Windows the pulse would appear in
        captures, so failure here is preferable to crashing.
        """
        try:
            import ctypes
            ctypes.windll.user32.SetWindowDisplayAffinity(
                int(self.winId()), 0x11  # WDA_EXCLUDEFROMCAPTURE
            )
        except Exception:
            pass
