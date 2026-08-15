# desktop/ocr.py
"""Reading text off the screen: RapidOCR plus the fuzzy matcher that turns a
requested label into the word box on screen that means it.

This is perception, not decision-making — grounding (agents/grounding.py) picks
which match to click, the snapshot builder (desktop/snapshot.py) tags each word
as foreground or background, and reflection reads the result back. All three
share ONE engine instance so they also share its cache.
"""
import difflib
import re
import time
from dataclasses import dataclass

import imagehash
import numpy as np
from loguru import logger
from PIL import Image


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
    """Wraps RapidOCR (pure Python ONNX, no system deps) with fuzzy text search.
    Initialised lazily on first use. Results are cached by perceptual hash so
    repeated calls on an unchanged screen skip the ONNX inference entirely.
    """

    _CACHE_TTL = 2.5   # seconds before a cached result expires
    _CACHE_MAX = 30    # maximum number of entries to keep

    def __init__(self):
        self._ocr = None
        self._available: bool | None = None
        self._cache: dict[str, tuple] = {}   # phash_str → (words, timestamp)

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

    def extract(self, image: Image.Image) -> list[OCRWord]:
        """Run OCR and return detected text boxes.
        Transparently caches by perceptual hash — unchanged screens reuse the
        previous result without running the ONNX model again (~150 ms saved).
        """
        if not self.is_available():
            return []

        # ── Cache lookup ──────────────────────────────────────────────────────
        phash_str: str | None = None
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
            # use_cls=False skips the text-angle classifier, a third ONNX model
            # that runs on every detected box to decide whether it is upside
            # down. Scanned paper needs that; a desktop screenshot never does,
            # so it only ever confirms 0°. Detection and recognition, the two
            # stages that decide WHAT is read and WHERE, are untouched.
            results, _ = self._ocr(img_np, use_cls=False)
        except Exception as e:
            logger.warning(f"[OCR] Inference error: {e}")
            return []
        if not results:
            return []

        words: list[OCRWord] = []
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

    def find_text(
        self,
        words: list[OCRWord],
        query: str,
        threshold: float = 0.60,
        foreground_only: bool = False,
    ) -> OCRWord | None:
        """Fuzzy-match query against all OCR words.
        Checks windows of 1-3 consecutive words to handle multi-word labels.
        When foreground_only=True, words with is_in_foreground=False are skipped.
        """
        if not words or not query.strip():
            return None
        q = query.strip().lower()
        best: tuple[float, OCRWord] | None = None

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
                elif combined in q and len(combined) >= 4 and re.search(
                    rf"(?:^|\s){re.escape(combined)}(?:$|\s)", q
                ):
                    # Whole-word containment only: "save" may stand in for
                    # "save button", but "meet" must NOT match "new meeting"
                    # (fragment of "meeting" — a different control entirely).
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
