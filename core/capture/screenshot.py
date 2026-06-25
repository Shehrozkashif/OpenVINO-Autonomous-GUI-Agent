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
        import os as _os

        from Xlib import X as _Xconst
        from Xlib import display as _Xdisplay
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
    # On Windows use GetDeviceCaps to get physical pixel dimensions without
    # doing a full screen capture (which ImageGrab.grab() would require).
    if _OS == "Windows":
        try:
            import ctypes
            hdc = ctypes.windll.user32.GetDC(0)
            w = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES
            h = ctypes.windll.gdi32.GetDeviceCaps(hdc, 117)  # DESKTOPVERTRES
            ctypes.windll.user32.ReleaseDC(0, hdc)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    img = _pil_grab()
    return img.width, img.height


# ── Frame comparison ──────────────────────────────────────────────────────────

# Resolution and hash size used for before/after change detection. The previous
# 320×180 / 8-bit-DCT phash was far too coarse: a context menu or small dialog on
# a 1080p+ display changed zero hash bits, so legitimately-opened menus were
# scored as "no change → click failed" (a systematic false failure, H2). A larger
# thumbnail with a 16×16 hash (256-bit) makes small but real UI changes detectable
# while staying cheap. Both the orchestrator (pre-action) and the reflection agent
# (post-action) MUST use this helper so the two hashes are directly comparable.
_FRAME_HASH_SIZE = 16
_FRAME_THUMB = (960, 540)


def frame_phash(img: Image.Image) -> "imagehash.ImageHash":
    """Perceptual hash tuned for before/after UI change detection (see H2)."""
    thumb = img.copy()
    thumb.thumbnail(_FRAME_THUMB, Image.LANCZOS)
    return imagehash.phash(thumb, hash_size=_FRAME_HASH_SIZE)


# ── Public API ────────────────────────────────────────────────────────────────

class ScreenCapture:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._last_hash: Optional[imagehash.ImageHash] = None
        # Regions (x1, y1, x2, y2) to black out in every captured frame.
        # Used to mask the agent's own GUI window so its text doesn't pollute OCR.
        # NOTE: the orchestrator overwrites this list every refresh cycle.
        self.exclude_regions: list = []
        # Additional always-applied mask regions that survive the orchestrator's
        # exclude_regions refresh — used for the always-on-top mission HUD,
        # whose text would otherwise contaminate OCR/planning.
        self.persistent_exclude_regions: list = []

    def capture(self) -> Image.Image:
        """Full-screen capture. Does not alter input focus on any platform."""
        img = _grab()
        regions = self.exclude_regions + self.persistent_exclude_regions
        if regions:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for (x1, y1, x2, y2) in regions:
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        return img

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
