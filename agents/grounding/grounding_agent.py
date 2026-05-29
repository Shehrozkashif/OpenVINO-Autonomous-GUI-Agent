# agents/grounding/grounding_agent.py
"""
UI Grounding Agent — converts natural language element descriptions to screen coordinates.

Uses a three-stage Hybrid Grounding Engine:
  Stage 1 — OCR Direct:    Fuzzy-match target text in live OCR output  (conf ~0.95)
  Stage 2 — VLM → OCR:    VLM identifies text label → OCR locates it   (conf ~0.85)
  Stage 3 — VLM Zone:     VLM picks screen zone (1-9) → zone center    (conf ~0.50)

"""
import base64
import io
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imagehash
from loguru import logger
from PIL import Image

from core.capture.screenshot import ScreenCapture, _screen_size
from core.grounding.hybrid_grounding import HybridGroundingEngine
from core.protocols.a2a import InferenceClient


@dataclass
class GroundingResult:
    x: int
    y: int
    confidence: float
    found: bool
    latency_ms: float
    target: str
    method: str = "unknown"


class ElementCache:
    """Cache UI element coordinates to avoid redundant VLM/OCR calls."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, target: str, screen_hash: str) -> Optional[Tuple[int, int, float, str]]:
        if target not in self._cache:
            return None
        x, y, conf, method, ts, cached_hash = self._cache[target]
        if screen_hash != cached_hash or (time.time() - ts) > self._ttl:
            del self._cache[target]
            return None
        return x, y, conf, method

    def put(self, target: str, x: int, y: int, conf: float, method: str, screen_hash: str):
        self._cache[target] = (x, y, conf, method, time.time(), screen_hash)

    def invalidate(self):
        self._cache.clear()


class UIGroundingAgent:
    """
    Locates UI elements from natural language descriptions using the
    three-stage Hybrid Grounding Engine (OCR + VLM).

    Constructor accepts any client that implements `query_vlm()` —
    both OVMSClient and DirectOpenVINOClient work transparently.
    """

    # Display resolution for VLM input (balance quality vs inference time)
    _DISPLAY_W = 1280
    _DISPLAY_H = 720

    def __init__(
        self,
        ovms_client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.5,
    ):
        self.client = ovms_client
        self.capturer = capturer
        self.cache = ElementCache()
        self.engine = HybridGroundingEngine(vlm_client=ovms_client)
        self.min_confidence = min_confidence
        self.screen_w, self.screen_h = _screen_size()
        logger.info(
            f"[GROUNDING] Initialized. Screen: {self.screen_w}×{self.screen_h}. "
            f"OCR available: {self.engine.ocr.is_available()}"
        )

    def ground(self, target: str, max_retries: int = 1) -> GroundingResult:
        """
        Locate a UI element by natural language description.

        Args:
            target: e.g. "the Save button", "browser address bar", "File menu"
            max_retries: number of additional attempts if first fails

        Returns:
            GroundingResult with (x, y) in screen coordinates.
        """
        start = time.time()

        # 1. Capture screen
        screenshot = self.capturer.capture()
        display = screenshot.copy()
        display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
        dw, dh = display.width, display.height

        # 2. Check cache
        h = str(imagehash.phash(display))
        cached = self.cache.get(target, h)
        if cached:
            x, y, conf, method = cached
            logger.info(f"[GROUNDING] Cache hit: '{target}' → ({x},{y}) via {method}")
            return GroundingResult(
                x=x, y=y, confidence=conf, found=True,
                latency_ms=(time.time() - start) * 1000,
                target=target, method=f"cache/{method}"
            )

        # 3. Encode display image for VLM
        img_b64 = self._encode(display)

        # 4. Run hybrid engine
        for attempt in range(max_retries + 1):
            try:
                result = self.engine.locate(
                    target=target,
                    image=display,
                    image_b64=img_b64,
                    screen_w=self.screen_w,
                    screen_h=self.screen_h,
                    display_w=dw,
                    display_h=dh,
                )
            except Exception as e:
                logger.warning(f"[GROUNDING] Attempt {attempt+1} error: {e}")
                result = None

            if result is not None:
                x, y, conf, method = result
                # Clamp to screen
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                self.cache.put(target, x, y, conf, method, h)
                logger.info(
                    f"[GROUNDING] '{target}' → ({x},{y}) conf={conf:.2f} "
                    f"method={method} attempt={attempt+1} "
                    f"latency={1000*(time.time()-start):.0f}ms"
                )
                return GroundingResult(
                    x=x, y=y, confidence=conf, found=conf >= self.min_confidence,
                    latency_ms=(time.time() - start) * 1000,
                    target=target, method=method
                )

        logger.warning(f"[GROUNDING] All stages failed for '{target}'")
        return GroundingResult(
            x=0, y=0, confidence=0.0, found=False,
            latency_ms=(time.time() - start) * 1000,
            target=target, method="failed"
        )

    def ground_multiple(self, targets: List[str]) -> List[GroundingResult]:
        """Ground multiple targets — captures screen once, reuses cache for same hash."""
        return [self.ground(t) for t in targets]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode(self, image: Image.Image, quality: int = 85) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
