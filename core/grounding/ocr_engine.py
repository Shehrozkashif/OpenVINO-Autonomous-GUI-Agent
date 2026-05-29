# core/grounding/ocr_engine.py
"""
OCR-based element locator using RapidOCR (pure Python, no system dependencies).

RapidOCR uses ONNX models for detection + recognition. It returns bounding
boxes as polygon corners, which we convert to (cx, cy) center coordinates.
This is used as Stage 1 and Stage 2 of the Hybrid Grounding Engine.
"""

import difflib
from dataclasses import dataclass
from typing import List, Optional, Tuple

from loguru import logger
from PIL import Image
import numpy as np


@dataclass
class OCRWord:
    text: str
    x: int  # left pixel (in image coords)
    y: int  # top pixel
    w: int  # width
    h: int  # height
    conf: float  # 0.0 – 1.0

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


class OCREngine:
    """
    Wraps RapidOCR with robust text extraction and fuzzy matching.
    Initialises lazily on first use to avoid slowing app startup.
    """

    def __init__(self):
        self._ocr = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._ocr = RapidOCR()
                self._available = True
                logger.info("[OCR] RapidOCR initialised successfully")
            except Exception as e:
                self._available = False
                logger.warning(f"[OCR] RapidOCR not available: {e}")
        return self._available

    def extract(self, image: Image.Image) -> List[OCRWord]:
        """
        Run OCR on the image, return all detected text boxes.
        Converts RapidOCR polygon format → OCRWord with bounding rect.
        """
        if not self.is_available():
            return []

        # RapidOCR works on numpy arrays
        img_np = np.array(image.convert("RGB"))
        try:
            results, elapse = self._ocr(img_np)
        except Exception as e:
            logger.warning(f"[OCR] Inference error: {e}")
            return []

        if not results:
            return []

        words: List[OCRWord] = []
        for item in results:
            # item = [box_points, text, confidence]
            if len(item) < 3:
                continue
            box, text, conf = item[0], item[1], item[2]
            # box is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corner points)
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x, y = min(xs), min(ys)
            w = max(xs) - x
            h = max(ys) - y
            if not str(text).strip():
                continue
            words.append(
                OCRWord(
                    text=str(text).strip(),
                    x=x,
                    y=y,
                    w=max(w, 1),
                    h=max(h, 1),
                    conf=float(conf),
                )
            )

        logger.debug(f"[OCR] Extracted {len(words)} text regions")
        return words

    def find_text(
        self,
        words: List[OCRWord],
        query: str,
        threshold: float = 0.60,
    ) -> Optional[OCRWord]:
        """
        Fuzzy-match query against all OCR words.
        Checks single words AND adjacent word-group windows (multi-word labels).
        Returns the best-matching OCRWord (or merged group) above threshold.
        """
        if not words or not query.strip():
            return None

        q = query.strip().lower()
        best: Optional[Tuple[float, OCRWord]] = None

        # Try windows of 1, 2, 3 consecutive words
        for window in range(1, 4):
            for i in range(len(words) - window + 1):
                group = words[i : i + window]
                combined = " ".join(w.text for w in group).lower()

                # Exact substring → near-perfect score
                if q == combined:
                    score = 1.0
                elif q in combined and len(q) >= 3:
                    score = 0.95
                elif combined in q and len(combined) >= 4:
                    score = 0.90
                else:
                    # Enforce length similarity to avoid "c" matching "search bar"
                    len_ratio = min(len(q), len(combined)) / max(len(q), len(combined))
                    if len_ratio < 0.4:
                        score = 0.0
                    else:
                        score = difflib.SequenceMatcher(None, q, combined).ratio()

                if score >= threshold:
                    # Merge bounding boxes for the group
                    gx = min(w.x for w in group)
                    gy = min(w.y for w in group)
                    gx2 = max(w.x + w.w for w in group)
                    gy2 = max(w.y + w.h for w in group)
                    merged = OCRWord(
                        text=" ".join(w.text for w in group),
                        x=gx,
                        y=gy,
                        w=gx2 - gx,
                        h=gy2 - gy,
                        conf=min(w.conf for w in group),
                    )
                    if best is None or score > best[0]:
                        best = (score, merged)

        if best:
            logger.debug(
                f"[OCR] Best match for '{query}': '{best[1].text}' "
                f"score={best[0]:.2f} at ({best[1].cx},{best[1].cy})"
            )
            return best[1]
        return None
