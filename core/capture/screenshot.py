# core/capture/screenshot.py
"""Windows screen capture — PIL.ImageGrab uses GDI BitBlt."""
import base64
import io

import imagehash
from PIL import Image


def _pil_grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    """Capture screen (or region) via PIL.ImageGrab (GDI BitBlt)."""
    from PIL import ImageGrab
    if w and h:
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    else:
        img = ImageGrab.grab()
    return img.convert("RGB")


def _grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    return _pil_grab(x, y, w, h)


def _screen_size() -> tuple:
    # Use GetDeviceCaps to get physical pixel dimensions without doing a full
    # screen capture (which ImageGrab.grab() would require).
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
        self._last_hash: imagehash.ImageHash | None = None
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
