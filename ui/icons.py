# ui/icons.py
"""Vector line-icon set drawn with QPainter — no asset files, crisp at any DPI,
recolorable per state. Icons are drawn in a normalized 24x24 box.

Usage:
    pixmap = icon_pixmap("home", QColor("#22D3EE"), 20)
    draw_icon(painter, "play", rect, color)
"""
import math

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF


def _pen(color: QColor, w: float = 1.7) -> QPen:
    pen = QPen(color, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def draw_icon(p: QPainter, name: str, rect: QRectF, color: QColor):
    """Draw icon `name` scaled into `rect`."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.translate(rect.x(), rect.y())
    s = min(rect.width(), rect.height()) / 24.0
    p.scale(s, s)
    p.setPen(_pen(color))
    p.setBrush(Qt.BrushStyle.NoBrush)
    fn = _ICONS.get(name, _icon_dot)
    fn(p, color)
    p.restore()


def icon_pixmap(name: str, color: QColor, size: int = 20, dpr: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    draw_icon(p, name, QRectF(0, 0, size, size), color)
    p.end()
    return pm


# ── Icon definitions (24x24 box) ─────────────────────────────────────────────

def _icon_home(p, c):
    path = QPainterPath(QPointF(4, 11))
    path.lineTo(12, 4.5)
    path.lineTo(20, 11)
    p.drawPath(path)
    body = QPainterPath(QPointF(6, 10.5))
    body.lineTo(6, 19.5)
    body.lineTo(18, 19.5)
    body.lineTo(18, 10.5)
    p.drawPath(body)
    p.drawLine(QPointF(10.5, 19.5), QPointF(10.5, 14.5))
    p.drawLine(QPointF(10.5, 14.5), QPointF(13.5, 14.5))
    p.drawLine(QPointF(13.5, 14.5), QPointF(13.5, 19.5))


def _icon_bolt(p, c):
    poly = QPolygonF([
        QPointF(13.5, 3.5), QPointF(6, 13.5), QPointF(11, 13.5),
        QPointF(10.5, 20.5), QPointF(18, 10.5), QPointF(13, 10.5),
    ])
    p.setBrush(QColor(c.red(), c.green(), c.blue(), 60))
    p.drawPolygon(poly)


def _icon_layers(p, c):
    p.drawPolygon(QPolygonF([
        QPointF(12, 4), QPointF(20, 8.5), QPointF(12, 13), QPointF(4, 8.5)]))
    path = QPainterPath(QPointF(4, 12.5))
    path.lineTo(12, 17)
    path.lineTo(20, 12.5)
    p.drawPath(path)
    path2 = QPainterPath(QPointF(4, 16))
    path2.lineTo(12, 20.5)
    path2.lineTo(20, 16)
    p.drawPath(path2)


def _icon_flow(p, c):
    p.drawEllipse(QPointF(6, 6), 2.6, 2.6)
    p.drawEllipse(QPointF(18, 12), 2.6, 2.6)
    p.drawEllipse(QPointF(6, 18), 2.6, 2.6)
    path = QPainterPath(QPointF(8.6, 6))
    path.cubicTo(13, 6, 12, 12, 15.4, 12)
    p.drawPath(path)
    path2 = QPainterPath(QPointF(8.6, 18))
    path2.cubicTo(13, 18, 12, 12, 15.4, 12)
    p.drawPath(path2)


def _icon_db(p, c):
    p.drawEllipse(QRectF(5, 3.5, 14, 5))
    path = QPainterPath(QPointF(5, 6))
    path.lineTo(5, 18)
    path.arcTo(QRectF(5, 15.5, 14, 5), 180, 180)
    path.lineTo(19, 6)
    p.drawPath(path)
    p.drawArc(QRectF(5, 9.5, 14, 5), 180 * 16, 180 * 16)


def _icon_screen(p, c):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 12), 2, 2)
    p.drawLine(QPointF(9.5, 19.5), QPointF(14.5, 19.5))
    p.drawLine(QPointF(12, 16.5), QPointF(12, 19.5))


def _icon_gear(p, c):
    p.drawEllipse(QPointF(12, 12), 4.0, 4.0)
    for i in range(8):
        a = math.radians(i * 45)
        x1, y1 = 12 + 6.0 * math.cos(a), 12 + 6.0 * math.sin(a)
        x2, y2 = 12 + 8.3 * math.cos(a), 12 + 8.3 * math.sin(a)
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


def _icon_play(p, c):
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([QPointF(8, 5.5), QPointF(19, 12), QPointF(8, 18.5)]))


def _icon_stop(p, c):
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2.5, 2.5)


def _icon_send(p, c):
    p.setBrush(QColor(c.red(), c.green(), c.blue(), 60))
    p.drawPolygon(QPolygonF([
        QPointF(4, 12), QPointF(20, 4.5), QPointF(15, 19.5),
        QPointF(11.5, 13.5),
    ]))
    p.drawLine(QPointF(11.5, 13.5), QPointF(20, 4.5))


def _icon_shield(p, c):
    path = QPainterPath(QPointF(12, 3.5))
    path.lineTo(19, 6.5)
    path.lineTo(19, 12)
    path.cubicTo(19, 16.5, 16, 19.2, 12, 20.5)
    path.cubicTo(8, 19.2, 5, 16.5, 5, 12)
    path.lineTo(5, 6.5)
    path.closeSubpath()
    p.drawPath(path)
    p.drawLine(QPointF(9, 12), QPointF(11.2, 14.2))
    p.drawLine(QPointF(11.2, 14.2), QPointF(15.2, 9.5))


def _icon_sparkle(p, c):
    p.setBrush(QColor(c.red(), c.green(), c.blue(), 70))
    path = QPainterPath(QPointF(12, 3))
    path.cubicTo(12.8, 9, 15, 11.2, 21, 12)
    path.cubicTo(15, 12.8, 12.8, 15, 12, 21)
    path.cubicTo(11.2, 15, 9, 12.8, 3, 12)
    path.cubicTo(9, 11.2, 11.2, 9, 12, 3)
    p.drawPath(path)


def _icon_clock(p, c):
    p.drawEllipse(QPointF(12, 12), 8, 8)
    p.drawLine(QPointF(12, 7.5), QPointF(12, 12))
    p.drawLine(QPointF(12, 12), QPointF(15.5, 14))


def _icon_refresh(p, c):
    p.drawArc(QRectF(5, 5, 14, 14), 30 * 16, 280 * 16)
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF([
        QPointF(19.6, 9.8), QPointF(15.6, 9.2), QPointF(18.8, 5.6)]))


def _icon_check(p, c):
    p.setPen(_pen(c, 2.2))
    p.drawLine(QPointF(5.5, 12.5), QPointF(10, 17))
    p.drawLine(QPointF(10, 17), QPointF(18.5, 7))


def _icon_x(p, c):
    p.setPen(_pen(c, 2.2))
    p.drawLine(QPointF(7, 7), QPointF(17, 17))
    p.drawLine(QPointF(17, 7), QPointF(7, 17))


def _icon_dot(p, c):
    p.setBrush(c)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(12, 12), 3.5, 3.5)


def _icon_chevron_left(p, c):
    p.drawLine(QPointF(14.5, 6), QPointF(8.5, 12))
    p.drawLine(QPointF(8.5, 12), QPointF(14.5, 18))


def _icon_chevron_right(p, c):
    p.drawLine(QPointF(9.5, 6), QPointF(15.5, 12))
    p.drawLine(QPointF(15.5, 12), QPointF(9.5, 18))


def _icon_panel(p, c):
    p.drawRoundedRect(QRectF(3.5, 5, 17, 14), 2.5, 2.5)
    p.drawLine(QPointF(15, 5.5), QPointF(15, 18.5))


def _icon_key(p, c):
    p.drawEllipse(QPointF(8.5, 14.5), 4, 4)
    p.drawLine(QPointF(11.5, 11.5), QPointF(19, 4.5))
    p.drawLine(QPointF(16.5, 7), QPointF(19, 9.5))


def _icon_cpu(p, c):
    p.drawRoundedRect(QRectF(6.5, 6.5, 11, 11), 2, 2)
    p.drawRect(QRectF(10, 10, 4, 4))
    for v in (9, 12, 15):
        p.drawLine(QPointF(v, 3.5), QPointF(v, 6.5))
        p.drawLine(QPointF(v, 17.5), QPointF(v, 20.5))
        p.drawLine(QPointF(3.5, v), QPointF(6.5, v))
        p.drawLine(QPointF(17.5, v), QPointF(20.5, v))


def _icon_eye(p, c):
    path = QPainterPath(QPointF(3.5, 12))
    path.cubicTo(7, 6.5, 17, 6.5, 20.5, 12)
    path.cubicTo(17, 17.5, 7, 17.5, 3.5, 12)
    p.drawPath(path)
    p.drawEllipse(QPointF(12, 12), 2.6, 2.6)


_ICONS = {
    "home": _icon_home,
    "bolt": _icon_bolt,
    "layers": _icon_layers,
    "flow": _icon_flow,
    "db": _icon_db,
    "screen": _icon_screen,
    "gear": _icon_gear,
    "play": _icon_play,
    "stop": _icon_stop,
    "send": _icon_send,
    "shield": _icon_shield,
    "sparkle": _icon_sparkle,
    "clock": _icon_clock,
    "refresh": _icon_refresh,
    "check": _icon_check,
    "x": _icon_x,
    "dot": _icon_dot,
    "chevron-left": _icon_chevron_left,
    "chevron-right": _icon_chevron_right,
    "panel": _icon_panel,
    "key": _icon_key,
    "cpu": _icon_cpu,
    "eye": _icon_eye,
}
