# agents/grounding/grounding_agent.py
"""
UI Grounding Agent — locates UI elements by natural language description.

Three-stage pipeline (Windows) / Two-stage (Linux/macOS):
  Stage 0 — Windows UIA:  Accessibility tree bounding box lookup    (conf 1.0 / 0.85+)
  Stage 1 — OCR Direct:   Fuzzy-match target text in live OCR       (conf 0.95)
  Stage 2 — VLM Coords:   UI-TARS predicts normalized (x,y) directly (conf ~0.80)

Stage 0 is Windows-only. On Linux/macOS the pipeline starts at Stage 1.
If all stages fail, returns found=False so the orchestrator can retry.
"""
import base64
import difflib
import io
import json
import platform
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
from loguru import logger
from PIL import Image

from core.capture.screenshot import ScreenCapture, _screen_size
from core.grounding.windows_uia import find_element as _uia_find, is_available as _uia_ok
from core.protocols.a2a import InferenceClient

_IS_WINDOWS = platform.system() == "Windows"


# ── VLM prompt constants ──────────────────────────────────────────────────────

_UITARS_SYSTEM_PROMPT = (
    "You are a GUI grounding agent. Locate the requested UI element in the "
    "screenshot and output a click action with its bounding box on a 0-1000 scale."
)

# UI-TARS native output format uses 0-1000 scale action strings.
# Asking for JSON with 0-1 floats causes the model to sometimes output
# 0-1000 integers in JSON, making coordinate interpretation ambiguous.
# Using the native action format is the most reliable approach.
#
# Fallback parsers still handle:
#   JSON:   {"x": 0.5, "y": 0.3}       (0-1 normalised)
#   JSON:   {"x": 378, "y": 654}       (0-1000 scale — now treated correctly)
#   Point:  <point>x y</point>         (0-1000 scale)
#   BBox:   (x1,y1),(x2,y2)
_VLM_COORD_PROMPT = (
    'In this screenshot, locate the UI element described as: "{target}"\n\n'
    "Respond with the bounding box in this EXACT format:\n"
    "click(start_box='[[x1, y1, x2, y2]]')\n\n"
    "Rules:\n"
    "  - x1,y1 = top-left corner of the element\n"
    "  - x2,y2 = bottom-right corner of the element\n"
    "  - All values are on a 0-1000 scale where (0,0)=top-left and (1000,1000)=bottom-right\n"
    "  - Output the action line only, nothing else\n"
    "If the element is not visible in the screenshot: not_found()"
)

# Common words that describe element role, not its visible text label
_ROLE_WORDS = {
    "the", "a", "an", "button", "btn", "menu", "bar", "tab", "panel",
    "icon", "link", "field", "box", "input", "area", "window", "dialog",
    "click", "open", "close", "toggle", "select", "choose", "find", "locate",
    "at", "on", "in", "of", "for", "with", "to", "from", "top", "bottom",
    "left", "right", "center", "middle", "upper", "lower", "primary", "main",
    "current", "active", "focused", "corner", "side", "edge", "near",
}


# ── OCR layer ─────────────────────────────────────────────────────────────────

@dataclass
class OCRWord:
    text: str
    x: int      # left pixel in image coords
    y: int      # top pixel
    w: int      # width
    h: int      # height
    conf: float # 0.0 – 1.0
    is_in_foreground: bool = True    # set by capture_snapshot(); default True for plain OCR results
    element_type: str = "document_text"  # "foreground_interactive" when tagged by capture_snapshot()

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


class OCREngine:
    """
    Wraps RapidOCR (pure Python ONNX, no system deps) with fuzzy text search.
    Initialised lazily on first use. Results are cached by perceptual hash so
    repeated calls on an unchanged screen skip the ONNX inference entirely.
    """

    _CACHE_TTL = 2.5   # seconds before a cached result expires
    _CACHE_MAX = 30    # maximum number of entries to keep

    def __init__(self):
        self._ocr = None
        self._available: Optional[bool] = None
        self._cache: Dict[str, tuple] = {}   # phash_str → (words, timestamp)

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
                self._available = True
                logger.info("[OCR] RapidOCR initialised")
            except Exception as e:
                self._available = False
                logger.warning(f"[OCR] RapidOCR not available: {e}")
        return self._available

    def extract(self, image: Image.Image) -> List[OCRWord]:
        """
        Run OCR and return detected text boxes.
        Transparently caches by perceptual hash — unchanged screens reuse the
        previous result without running the ONNX model again (~150 ms saved).
        """
        if not self.is_available():
            return []

        # ── Cache lookup ──────────────────────────────────────────────────────
        phash_str: Optional[str] = None
        try:
            phash_str = str(imagehash.phash(image))
            cached = self._cache.get(phash_str)
            if cached is not None:
                words, ts = cached
                if time.time() - ts < self._CACHE_TTL:
                    logger.debug("[OCR] Cache hit — skipping inference")
                    return words
        except Exception:
            phash_str = None  # hash failed; run inference uncached

        # ── Run inference ─────────────────────────────────────────────────────
        img_np = np.array(image.convert("RGB"))
        try:
            results, _ = self._ocr(img_np)
        except Exception as e:
            logger.warning(f"[OCR] Inference error: {e}")
            return []
        if not results:
            return []

        words: List[OCRWord] = []
        for item in results:
            if len(item) < 3:
                continue
            box, text, conf = item[0], item[1], item[2]
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            if not str(text).strip():
                continue
            words.append(OCRWord(str(text).strip(), x, y, max(w, 1), max(h, 1), float(conf)))

        logger.debug(f"[OCR] Extracted {len(words)} text regions")

        # ── Cache store ───────────────────────────────────────────────────────
        if phash_str is not None:
            self._cache[phash_str] = (words, time.time())
            if len(self._cache) > self._CACHE_MAX:
                oldest = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest]

        return words

    def invalidate_cache(self):
        """Clear all cached OCR results (call after an action changes the screen)."""
        self._cache.clear()

    def find_text(
        self,
        words: List[OCRWord],
        query: str,
        threshold: float = 0.60,
        foreground_only: bool = False,
    ) -> Optional[OCRWord]:
        """
        Fuzzy-match query against all OCR words.
        Checks windows of 1-3 consecutive words to handle multi-word labels.
        When foreground_only=True, words with is_in_foreground=False are skipped.
        """
        if not words or not query.strip():
            return None
        q = query.strip().lower()
        best: Optional[Tuple[float, OCRWord]] = None

        for window in range(1, 4):
            for i in range(len(words) - window + 1):
                group = words[i : i + window]
                if foreground_only and any(not w.is_in_foreground for w in group):
                    continue
                if foreground_only and any(w.element_type != "foreground_interactive" for w in group):
                    continue
                combined = " ".join(w.text for w in group).lower()

                if q == combined:
                    score = 1.0
                elif q in combined and len(q) >= 3:
                    # Penalise matches where query is a tiny fragment of a long text.
                    # e.g. "folder" inside a 100-char Monitor event line → ~0.25, rejected.
                    length_penalty = min(1.0, (len(q) / max(len(combined), 1)) * 4)
                    score = 0.95 * length_penalty
                elif combined in q and len(combined) >= 4:
                    score = 0.90
                else:
                    len_ratio = min(len(q), len(combined)) / max(len(q), len(combined))
                    score = 0.0 if len_ratio < 0.4 else difflib.SequenceMatcher(None, q, combined).ratio()

                if score >= threshold:
                    gx  = min(w.x for w in group)
                    gy  = min(w.y for w in group)
                    gx2 = max(w.x + w.w for w in group)
                    gy2 = max(w.y + w.h for w in group)
                    merged = OCRWord(
                        text=" ".join(w.text for w in group),
                        x=gx, y=gy, w=gx2 - gx, h=gy2 - gy,
                        conf=min(w.conf for w in group),
                        is_in_foreground=all(w.is_in_foreground for w in group),
                        element_type=(
                            "foreground_interactive"
                            if all(w.element_type == "foreground_interactive" for w in group)
                            else "document_text"
                        ),
                    )
                    if best is None or score > best[0]:
                        best = (score, merged)

        if best:
            logger.debug(f"[OCR] Best match for '{query}': '{best[1].text}' score={best[0]:.2f}")
            return best[1]
        return None


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    x: int
    y: int
    confidence: float
    found: bool
    latency_ms: float
    target: str
    method: str = "unknown"
    element_type: str = "foreground_interactive"  # "document_text" for non-interactive OCR hits


class ElementCache:
    """Cache coordinates keyed by (target, perceptual screen hash) with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, target: str, screen_hash: str) -> Optional[Tuple[int, int, float, str, str]]:
        if target not in self._cache:
            return None
        x, y, conf, method, element_type, ts, cached_hash = self._cache[target]
        if screen_hash != cached_hash or (time.time() - ts) > self._ttl:
            del self._cache[target]
            return None
        return x, y, conf, method, element_type

    def put(
        self, target: str, x: int, y: int, conf: float, method: str,
        screen_hash: str, element_type: str = "foreground_interactive",
    ):
        self._cache[target] = (x, y, conf, method, element_type, time.time(), screen_hash)

    def invalidate(self):
        self._cache.clear()


class UIGroundingAgent:
    """
    Locates UI elements by natural language description.

    Stage 1 — OCR:  fast, free, pixel-perfect for text-labeled elements.
    Stage 2 — VLM:  UI-TARS-1.5-7B direct coordinate prediction for everything else.

    Accepts any client that implements InferenceClient (OllamaClient, OVMSClient, etc.).
    """

    _DISPLAY_W = 960   # must match capture_snapshot() thumbnail size so OCR cache entries are shared
    _DISPLAY_H = 540   # and element_type tags set by capture_snapshot() flow into grounding

    def __init__(
        self,
        ovms_client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.5,
        ocr: Optional["OCREngine"] = None,
    ):
        self.client = ovms_client
        self.capturer = capturer
        self.cache = ElementCache()
        self.ocr = ocr if ocr is not None else OCREngine()
        self.min_confidence = min_confidence
        self.screen_w, self.screen_h = _screen_size()
        logger.info(
            f"[GROUNDING] Ready. Screen: {self.screen_w}×{self.screen_h}. "
            f"OCR: {'on' if self.ocr.is_available() else 'off (pip install rapidocr-onnxruntime)'}"
        )

    # ── public API ────────────────────────────────────────────────────────────

    def ground(self, target: str, max_retries: int = 1) -> GroundingResult:
        """
        Locate a UI element by natural language description.
        Returns GroundingResult with screen (x, y). found=False if all stages fail.

        On failure, asks the LLM for 3 alternative phrasings and retries each
        before giving up — covers cases where the model used different label text
        than what OCR actually detected on screen.
        """
        start = time.time()

        screenshot = self.capturer.capture()
        display = screenshot.copy()
        display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
        dw, dh = display.width, display.height
        # Guard against zero-sized thumbnails (can happen on headless/virtual displays)
        scale_x = self.screen_w / dw if dw > 0 else 1.0
        scale_y = self.screen_h / dh if dh > 0 else 1.0

        screen_hash = str(imagehash.phash(display))
        cached = self.cache.get(target, screen_hash)
        if cached:
            x, y, conf, method, element_type = cached
            logger.info(f"[GROUNDING] Cache hit: '{target}' → ({x},{y}) via {method}")
            return GroundingResult(x=x, y=y, confidence=conf, found=True,
                                   latency_ms=(time.time() - start) * 1000,
                                   target=target, method=f"cache/{method}",
                                   element_type=element_type)

        img_b64 = self._encode(display)
        words = self.ocr.extract(display) if self.ocr.is_available() else []

        for attempt in range(max_retries + 1):
            # Stage 2 (VLM) only on the first attempt: on shared-VRAM machines a
            # VLM call costs a 30-60s model swap, and re-asking it the identical
            # question on an unchanged screen returns the same answer anyway.
            result = self._locate(target, display, img_b64, words, scale_x, scale_y,
                                  use_vlm=(attempt == 0))
            if result:
                x, y, conf, method, element_type = result
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                self.cache.put(target, x, y, conf, method, screen_hash, element_type)
                logger.info(
                    f"[GROUNDING] '{target}' → ({x},{y}) conf={conf:.2f} "
                    f"method={method} attempt={attempt+1} "
                    f"latency={1000*(time.time()-start):.0f}ms"
                )
                return GroundingResult(x=x, y=y, confidence=conf,
                                       found=conf >= self.min_confidence,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method=method,
                                       element_type=element_type)

        # All direct attempts failed — ask the LLM for alternative label phrasings
        alternatives = self._rephrase_targets(target)
        for alt in alternatives:
            logger.info(f"[GROUNDING] Rephrasing: trying '{alt}' for '{target}'")
            # Rephrased labels are alternative TEXT spellings — UIA/OCR are the
            # right matchers for them; skip the expensive VLM here.
            result = self._locate(alt, display, img_b64, words, scale_x, scale_y,
                                  use_vlm=False)
            if result:
                x, y, conf, method, element_type = result
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                self.cache.put(target, x, y, conf, f"rephrase/{method}", screen_hash, element_type)
                logger.info(f"[GROUNDING] Rephrasing succeeded: '{alt}' → ({x},{y})")
                return GroundingResult(x=x, y=y, confidence=conf,
                                       found=conf >= self.min_confidence,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method=f"rephrase/{method}",
                                       element_type=element_type)

        logger.warning(f"[GROUNDING] All stages failed for '{target}'")
        return GroundingResult(x=0, y=0, confidence=0.0, found=False,
                               latency_ms=(time.time() - start) * 1000,
                               target=target, method="failed")

    def ground_fast(self, target: str) -> GroundingResult:
        """
        Stage 0 + Stage 1 only (UIA + OCR) — no VLM call.

        Used during burst pre-grounding where transient elements (context-menu
        items) may not be visible yet.  If they're absent, Stage 2 would block
        for 30-50 s just to confirm not-found; this method returns immediately.
        """
        start = time.time()
        screenshot = self.capturer.capture()
        display = screenshot.copy()
        display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
        dw, dh = display.width, display.height
        scale_x = self.screen_w / dw if dw > 0 else 1.0
        scale_y = self.screen_h / dh if dh > 0 else 1.0

        words = self.ocr.extract(display) if self.ocr.is_available() else []

        if _IS_WINDOWS and _uia_ok():
            r = _uia_find(target)
            if r:
                x, y, conf = r
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                return GroundingResult(x=x, y=y, confidence=conf, found=True,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method="uia",
                                       element_type="foreground_interactive")

        if words:
            query = _strip_role_words(target)
            match = self.ocr.find_text(words, query, threshold=0.65)
            if match:
                x = int(match.cx * scale_x)
                y = int(match.cy * scale_y)
                return GroundingResult(x=x, y=y, confidence=0.95, found=True,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method="ocr_direct",
                                       element_type=match.element_type)

        return GroundingResult(x=0, y=0, confidence=0.0, found=False,
                               latency_ms=(time.time() - start) * 1000,
                               target=target, method="failed")

    def ground_multiple(self, targets: List[str]) -> List[GroundingResult]:
        return [self.ground(t) for t in targets]

    # ── grounding stages ──────────────────────────────────────────────────────

    def _locate(
        self,
        target: str,
        display: Image.Image,
        img_b64: str,
        words: List[OCRWord],
        scale_x: float,
        scale_y: float,
        use_vlm: bool = True,
    ) -> Optional[Tuple[int, int, float, str, str]]:
        # Stage 0: Windows UIAutomation — fast, pixel-perfect, no model needed
        # UIA elements are always interactive → element_type="foreground_interactive"
        if _IS_WINDOWS and _uia_ok():
            r = _uia_find(target)
            if r:
                x, y, conf = r
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                logger.info(f"[GROUNDING/S0-UIA] '{target}' → screen({x},{y}) conf={conf:.2f}")
                return (x, y, conf, "uia", "foreground_interactive")
            logger.debug(f"[GROUNDING/S0-UIA] '{target}' not found in UIA tree")

        # Stage 1: OCR direct fuzzy-match — carry element_type from the matched word
        if words:
            query = _strip_role_words(target)
            match = self.ocr.find_text(words, query, threshold=0.65)
            if match:
                x, y = int(match.cx * scale_x), int(match.cy * scale_y)
                logger.info(f"[GROUNDING/S1-OCR] '{target}' → '{match.text}' → screen({x},{y})")
                return (x, y, 0.95, "ocr_direct", match.element_type)
            logger.debug(f"[GROUNDING/S1] No OCR match for '{query}'")

        # Stage 2: VLM direct coordinate prediction
        # VLM is expected to hit interactive elements → element_type="foreground_interactive"
        if use_vlm and self.client:
            result = self._vlm_coords(target, img_b64, int(display.width), int(display.height))
            if result:
                return result

        return None

    def _vlm_coords(
        self, target: str, img_b64: str,
        display_w: int = 0, display_h: int = 0,
    ) -> Optional[Tuple[int, int, float, str, str]]:
        """Ask UI-TARS for normalized (x,y) coordinates and scale to screen pixels.

        display_w/display_h: pixel dimensions of the image sent to the VLM.
        Needed to correctly scale pixel-valued JSON coordinates back to screen space.
        """
        try:
            resp = self.client.query_vlm(
                prompt=_VLM_COORD_PROMPT.format(target=target),
                image_base64=img_b64,
                max_tokens=120,
                temperature=0.0,
                system_prompt=_UITARS_SYSTEM_PROMPT,
            )
            text = resp.content.strip()
            # Strip chain-of-thought tokens
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            result = self._parse_coords(text, self.screen_w, self.screen_h, display_w, display_h)
            if result:
                x, y, conf = result
                logger.info(f"[GROUNDING/S2-VLM] '{target}' -> screen({x},{y}) conf={conf:.2f}")
                return (x, y, conf, "vlm_coords", "foreground_interactive")
        except Exception as e:
            logger.warning(f"[GROUNDING/S2-VLM] Error: {e}")
        return None

    def _parse_coords(
        self, text: str, screen_w: int, screen_h: int,
        display_w: int = 0, display_h: int = 0,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Parse VLM output into screen pixel coordinates (x, y, confidence).

        display_w/display_h: size of the image that was sent to the VLM.
        When the model returns pixel-valued coordinates they are in the display
        image's coordinate space and must be scaled to screen space:
            screen_x = (pixel_x / display_w) * screen_w
        If display dims are not provided, pixel coords are used as-is (clipped).

        Handles the formats UI-TARS may emit:
          action: click(start_box='[[x1,y1,x2,y2]]')   0-1000 scale
          JSON:   {"x": 0.5, "y": 0.3, ...}            normalised 0-1
          JSON:   {"x": 423, "y": 706, ...}             display pixels
          Point:  <point>500 300</point>                 0-1000 or display pixels
          BBox:   (x1,y1),(x2,y2)                       same scale rules
        """
        # Explicit not-found answer — fast, clean exit (no format warning)
        if "not_found" in text.lower():
            logger.debug("[GROUNDING/S2] VLM reports element not visible")
            return None

        # Effective display dimensions for pixel-to-screen scaling.
        # Fall back to screen dims (identity scale) when not provided.
        dw = display_w if display_w > 1 else screen_w
        dh = display_h if display_h > 1 else screen_h

        def _px_to_screen(px: float, py: float) -> Tuple[int, int]:
            """Scale display-space pixels to screen pixels, clamped to screen bounds."""
            return (
                min(int(px / dw * screen_w), screen_w - 1),
                min(int(py / dh * screen_h), screen_h - 1),
            )

        # UI-TARS native action format (0-1000 scale, independent of display size):
        #   click(start_box='[[x1, y1, x2, y2]]')
        # Bracket count varies in practice ('[[', '[[[', …) — tolerate 0-4.
        m = re.search(
            r"(?:click|tap)\s*\(\s*start_box\s*=\s*'?\[{0,4}"
            r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
            r"\]{0,4}'?\s*\)",
            text,
        )
        if m:
            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            return (int(cx / 1000 * screen_w), int(cy / 1000 * screen_h), 0.90)

        # 2-value form: model emits [[cx, cy]] instead of [[x1,y1,x2,y2]].
        # Treat as center point on the 0-1000 scale.
        m = re.search(
            r"(?:click|tap)\s*\(\s*start_box\s*=\s*'?\[{0,4}"
            r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
            r"\s*[\]\)']+",
            text,
        )
        if m:
            cx, cy = float(m.group(1)), float(m.group(2))
            return (int(cx / 1000 * screen_w), int(cy / 1000 * screen_h), 0.80)

        # JSON block — x/y may be:
        #   0-1 normalised floats  (model followed instructions)
        #   1-1000 integers        (UI-TARS native 0-1000 scale leaked into JSON)
        #   > 1000                 (raw display pixels — scale by dw/dh)
        # Also handle malformed JSON like {"x": 658, 294} (missing "y": key).
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            raw_json = re.sub(r",\s*([\]}])", r"\1", m.group())
            try:
                data = json.loads(raw_json)
                if not data.get("found", True):
                    return None
                xv, yv = float(data["x"]), float(data["y"])
                conf = float(data.get("confidence", 0.7))
                if 0.0 <= xv <= 1.0 and 0.0 <= yv <= 1.0:
                    # Normalised 0-1 — most accurate
                    return (int(xv * screen_w), int(yv * screen_h), conf)
                if 1.0 < xv <= 1000 and 1.0 < yv <= 1000:
                    # UI-TARS native 0-1000 scale — treat as normalised/1000
                    return (int(xv / 1000 * screen_w), int(yv / 1000 * screen_h), conf)
                # Values > 1000 — raw display pixels, scale to screen space
                sx, sy = _px_to_screen(xv, yv)
                return (sx, sy, conf)
            except (json.JSONDecodeError, ValueError, KeyError):
                # Fallback: try to extract two numbers from the JSON string
                nums = re.findall(r'[\d]+(?:\.\d+)?', raw_json)
                if len(nums) >= 2:
                    try:
                        xv, yv = float(nums[0]), float(nums[1])
                        if 0.0 <= xv <= 1.0 and 0.0 <= yv <= 1.0:
                            return (int(xv * screen_w), int(yv * screen_h), 0.65)
                        if 1.0 < xv <= 1000 and 1.0 < yv <= 1000:
                            return (int(xv / 1000 * screen_w), int(yv / 1000 * screen_h), 0.65)
                        sx, sy = _px_to_screen(xv, yv)
                        return (sx, sy, 0.65)
                    except ValueError:
                        pass

        # <point>cx cy</point>  (0-1000 scale or display-space pixels)
        m = re.search(r'<point>\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*</point>', text)
        if m:
            px, py = float(m.group(1)), float(m.group(2))
            if px <= 1.0 and py <= 1.0:
                return (int(px * screen_w), int(py * screen_h), 0.85)
            if px <= 1000 and py <= 1000:
                # Could be 0-1000 normalised or small-display pixels; treat as 0-1000.
                return (int(px / 1000 * screen_w), int(py / 1000 * screen_h), 0.85)
            sx, sy = _px_to_screen(px, py)
            return (sx, sy, 0.75)

        # (x1,y1),(x2,y2) bounding box — take centre
        m = re.search(
            r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
            text
        )
        if m:
            x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if cx <= 1.0 and cy <= 1.0:
                return (int(cx * screen_w), int(cy * screen_h), 0.85)
            if cx <= 1000 and cy <= 1000:
                return (int(cx / 1000 * screen_w), int(cy / 1000 * screen_h), 0.85)
            sx, sy = _px_to_screen(cx, cy)
            return (sx, sy, 0.75)

        logger.debug(f"[GROUNDING/S2] Unrecognised VLM format: '{text[:100]}'")
        return None

    def _rephrase_targets(self, target: str) -> List[str]:
        """
        Ask the LLM for up to 3 alternative text labels for the same UI element.
        Used when OCR + VLM both fail to locate the original target string.
        Returns an empty list if the LLM call fails or produces nothing useful.
        """
        if not self.client:
            return []
        try:
            messages = [
                {"role": "system", "content":
                 "You are a GUI labelling assistant. Given a UI element description, "
                 "return 3 shorter alternative text labels that OCR might have detected "
                 "on screen for the same element. "
                 "Reply with ONLY a JSON array of 3 strings. No explanation."},
                {"role": "user", "content":
                 f'UI element description: "{target}"\nAlternative labels:'},
            ]
            resp = self.client.query_llm(messages, max_tokens=80, temperature=0.2)
            text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
            m = re.search(r'\[.*?\]', text, re.DOTALL)
            if m:
                candidates = json.loads(m.group())
                return [
                    str(c).strip()
                    for c in candidates
                    if str(c).strip() and str(c).strip().lower() != target.lower()
                ][:3]
        except Exception as e:
            logger.debug(f"[GROUNDING] Rephrase LLM call failed: {e}")
        return []

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode(self, image: Image.Image, quality: int = 85) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()


def _strip_role_words(text: str) -> str:
    tokens = text.lower().split()
    core = [t for t in tokens if t not in _ROLE_WORDS]
    return " ".join(core) if core else text
