# core/capture/screenshot.py
import base64
import io
from typing import Optional, Tuple

import imagehash
import mss
from PIL import Image


class ScreenCapture:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._last_hash: Optional[imagehash.ImageHash] = None

    def capture(self) -> Image.Image:
        """Capture full screen as PIL Image."""
        with mss.mss() as sct:
            raw = sct.grab(sct.monitors[self.monitor])
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def capture_as_base64(self, quality: int = 85) -> str:
        """Capture screen and return as base64 JPEG string (for sending to OVMS)."""
        img = self.capture()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def capture_resized(self, width: int, height: int) -> Image.Image:
        """Capture and resize. Used for the multi-resolution tiers."""
        return self.capture().resize((width, height), Image.LANCZOS)

    def capture_region(self, x: int, y: int, width: int, height: int) -> Image.Image:
        """Capture a specific screen region (for Reflection Agent zoom-in)."""
        with mss.mss() as sct:
            raw = sct.grab({"left": x, "top": y, "width": width, "height": height})
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def has_changed(self, threshold: float = 0.05) -> bool:
        """Return True if screen changed meaningfully since last call.

        Uses perceptual hash (pHash): 64-bit fingerprint of image content.
        threshold=0.05 → 5% of bits differ = meaningful change detected.
        Achieves the 60-80% storage reduction strategy from the proposal.
        """
        current = self.capture()
        current_hash = imagehash.phash(current)

        if self._last_hash is None:
            self._last_hash = current_hash
            return True

        # Hamming distance normalized 0.0 – 1.0
        distance = (self._last_hash - current_hash) / 64.0
        changed = distance > threshold
        if changed:
            self._last_hash = current_hash
        return changed

    def get_dpi_scale(self) -> float:
        """Detect DPI scaling factor.

        On HiDPI screens (e.g. 4K at 200% scaling), the OS reports virtual
        pixels to pyautogui but mss captures physical pixels.
        This causes clicks to land at wrong coordinates.
        """
        import pyautogui
        logical_w, _ = pyautogui.size()     # OS logical resolution
        with mss.mss() as sct:
            physical_w = sct.monitors[self.monitor]["width"]  # actual pixels
        return physical_w / logical_w        # typically 1.0 or 2.0
