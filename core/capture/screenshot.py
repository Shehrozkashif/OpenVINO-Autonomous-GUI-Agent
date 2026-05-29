# core/capture/screenshot.py
"""
Cross-platform screen capture — auto-detects OS at startup.

  Linux  → Xlib root.get_image(): reads pixels directly from the X server
            without sending compositor events (PIL.ImageGrab causes GNOME Shell
            to dismiss the Activities overlay mid-task).

  Windows → PIL.ImageGrab: uses GDI BitBlt, safe on Windows.

  macOS   → PIL.ImageGrab: uses Quartz, safe on macOS.
"""
import base64
import io
import platform
from typing import Optional

import imagehash
from PIL import Image

_OS = platform.system()

# ── Backend selection ─────────────────────────────────────────────────────────

_XLIB_OK = False
_xd = None   # set only when Xlib initialises successfully

if _OS == "Linux":
    try:
        from Xlib import display as _Xdisplay, X as _Xconst
        import os as _os
        _xd = _Xdisplay.Display(_os.environ.get("DISPLAY", ":0"))
        _XLIB_OK = True
    except Exception as _xlib_err:
        import logging as _logging
        _logging.getLogger(__name__).debug(f"Xlib unavailable: {_xlib_err} — using PIL fallback")


def _xlib_grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    """Capture screen (or region) via Xlib — does not steal X11 focus."""
    root = _xd.screen().root
    geom = root.get_geometry()
    rw, rh = geom.width, geom.height
    if w == 0 or h == 0:
        w, h = rw, rh
    raw = root.get_image(x, y, w, h, _Xconst.ZPixmap, 0xFFFFFF)
    return Image.frombytes("RGB", (w, h), raw.data, "raw", "BGRX")


def _pil_grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    """Capture screen (or region) via PIL.ImageGrab — Windows/macOS."""
    from PIL import ImageGrab
    if w and h:
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    else:
        img = ImageGrab.grab()
    return img.convert("RGB")


def _grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    if _XLIB_OK:
        return _xlib_grab(x, y, w, h)
    return _pil_grab(x, y, w, h)


def _xlib_screen_size() -> tuple:
    s = _xd.screen()
    return s.width_in_pixels, s.height_in_pixels


def _screen_size() -> tuple:
    if _XLIB_OK:
        return _xlib_screen_size()
    img = _pil_grab()
    return img.width, img.height


# ── Public API ────────────────────────────────────────────────────────────────

class ScreenCapture:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._last_hash: Optional[imagehash.ImageHash] = None

    def capture(self) -> Image.Image:
        """Full-screen capture. Does not alter input focus on any platform."""
        return _grab()

    def capture_as_base64(self, quality: int = 85) -> str:
        img = self.capture()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def capture_resized(self, width: int, height: int) -> Image.Image:
        return self.capture().resize((width, height), Image.LANCZOS)

    def capture_region(self, x: int, y: int, width: int, height: int) -> Image.Image:
        return _grab(x, y, width, height)

    def has_changed(self, threshold: float = 0.05) -> bool:
        current = self.capture()
        current_hash = imagehash.phash(current)
        if self._last_hash is None:
            self._last_hash = current_hash
            return True
        changed = (self._last_hash - current_hash) / 64.0 > threshold
        if changed:
            self._last_hash = current_hash
        return changed

    def get_dpi_scale(self) -> float:
        try:
            w_logical, _ = _screen_size()
            w_physical = self.capture().width
            return w_physical / w_logical if w_logical else 1.0
        except Exception:
            return 1.0
