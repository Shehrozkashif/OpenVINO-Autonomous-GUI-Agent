# core/grounding/som_engine.py
"""
Set-of-Mark (SoM) Grounding Engine.

Instead of asking the VLM for raw pixel coordinates (which it refuses),
we use OpenCV to detect all candidate UI regions, number them, draw them
on the screenshot, and ask the VLM to select the correct region NUMBER.

This is the same technique used by SoM-GPT4V and modern GUI agents.

Pipeline:
  screenshot → OpenCV region detection → annotated image with numbered boxes
      → VLM picks a number → return center coords of that region
"""
import io
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger


@dataclass
class SoMRegion:
    """A candidate UI element region."""
    index: int          # display number shown on annotated image
    x: int              # top-left x
    y: int              # top-left y
    w: int              # width
    h: int              # height
    cx: int             # center x
    cy: int             # center y


class SoMEngine:
    """
    Uses OpenCV to detect interactive UI regions on a screenshot,
    numbers them, and draws the annotations so a VLM can pick one by number.
    """

    # ── tuneable parameters ──────────────────────────────────────────────────
    MIN_AREA   = 100        # ignore tiny noise blobs (px²)
    MAX_AREA_F = 0.30       # ignore regions that cover > 30% of screen
    CANNY_LO   = 50
    CANNY_HI   = 150
    DILATE_K   = 3          # kernel size for dilation (joins nearby edges)
    BOX_COLOR  = (0, 120, 255)   # BGR orange-blue for boxes
    TAG_BG     = (0, 120, 255)
    TAG_FG     = (255, 255, 255)
    FONT_SCALE = 0.55
    THICKNESS  = 2
    # ─────────────────────────────────────────────────────────────────────────

    def detect_regions(self, image: Image.Image) -> List[SoMRegion]:
        """
        Detect candidate UI element bounding boxes using OpenCV.
        Returns a deduplicated, size-filtered list of SoMRegions.
        """
        img_np = np.array(image.convert("RGB"))
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w   = gray.shape
        max_area = h * w * self.MAX_AREA_F

        # Multi-scale edge → contour pipeline
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, self.CANNY_LO, self.CANNY_HI)
        kernel  = np.ones((self.DILATE_K, self.DILATE_K), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: List[Tuple[int,int,int,int]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            boxes.append((x, y, bw, bh))

        # Merge overlapping / nested boxes
        boxes = self._merge_overlapping(boxes)

        # Build SoMRegion list (capped at 50 so VLM response stays short)
        regions: List[SoMRegion] = []
        for i, (x, y, bw, bh) in enumerate(boxes[:50], start=1):
            regions.append(SoMRegion(
                index=i,
                x=x, y=y, w=bw, h=bh,
                cx=x + bw // 2,
                cy=y + bh // 2
            ))

        logger.debug(f"[SoM] Detected {len(regions)} candidate regions")
        return regions

    def annotate(self, image: Image.Image, regions: List[SoMRegion]) -> Image.Image:
        """
        Draw numbered bounding boxes on a copy of the image.
        Returns the annotated PIL image.
        """
        annotated = image.copy().convert("RGB")
        img_np = np.array(annotated)

        for r in regions:
            # Draw rectangle
            cv2.rectangle(img_np, (r.x, r.y), (r.x + r.w, r.y + r.h),
                          self.BOX_COLOR, self.THICKNESS)
            # Draw tag background
            label  = str(r.index)
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, self.THICKNESS
            )
            tag_x2 = r.x + tw + 4
            tag_y2 = r.y + th + 6
            cv2.rectangle(img_np, (r.x, r.y), (tag_x2, tag_y2),
                          self.TAG_BG, -1)
            cv2.putText(
                img_np, label,
                (r.x + 2, r.y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE,
                self.TAG_FG, self.THICKNESS
            )

        return Image.fromarray(img_np)

    def get_region(self, regions: List[SoMRegion], index: int) -> Optional[SoMRegion]:
        """Return the SoMRegion with the given display index, or None."""
        for r in regions:
            if r.index == index:
                return r
        return None

    def parse_region_number(self, vlm_response: str, max_index: int) -> Optional[int]:
        """
        Extract the first integer from the VLM response that is a valid region index.
        Handles: "3", "Region 3", "number 3", "3.", "#3", etc.
        """
        numbers = re.findall(r'\b(\d+)\b', vlm_response)
        for n in numbers:
            val = int(n)
            if 1 <= val <= max_index:
                return val
        return None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _merge_overlapping(
        self, boxes: List[Tuple[int,int,int,int]]
    ) -> List[Tuple[int,int,int,int]]:
        """Iteratively merge bounding boxes that overlap by > 50%."""
        if not boxes:
            return []

        merged = True
        while merged:
            merged = False
            result: List[Tuple[int,int,int,int]] = []
            used = [False] * len(boxes)
            for i, a in enumerate(boxes):
                if used[i]:
                    continue
                ax, ay, aw, ah = a
                for j, b in enumerate(boxes[i+1:], start=i+1):
                    if used[j]:
                        continue
                    bx, by, bw, bh = b
                    # Compute intersection
                    ix = max(ax, bx)
                    iy = max(ay, by)
                    ix2 = min(ax+aw, bx+bw)
                    iy2 = min(ay+ah, by+bh)
                    if ix2 > ix and iy2 > iy:
                        inter = (ix2-ix) * (iy2-iy)
                        area_a = aw * ah
                        area_b = bw * bh
                        if inter / min(area_a, area_b) > 0.5:
                            # Merge into bounding union
                            nx = min(ax, bx)
                            ny = min(ay, by)
                            nw = max(ax+aw, bx+bw) - nx
                            nh = max(ay+ah, by+bh) - ny
                            ax, ay, aw, ah = nx, ny, nw, nh
                            used[j] = True
                            merged = True
                result.append((ax, ay, aw, ah))
                used[i] = True
            boxes = result

        return boxes
