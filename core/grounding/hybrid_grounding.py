# core/grounding/hybrid_grounding.py
"""
Hybrid Grounding Engine — four-stage pipeline for robust UI element location.

Stage 1 — OCR Direct Match:
  Run OCR on the screenshot and fuzzy-search for the target text directly.
  Works perfectly when the user says "the File menu" or "the Save button"
  because the text "File" and "Save" appear literally on screen.

Stage 2 — VLM Semantic → OCR:
  Ask the VLM "What exact text label does [target] show?" → get a short answer
  like "New Tab" or "Accept" → search OCR results for that label.
  Works for targets described semantically (e.g. "the accept cookies button").

Stage 3 — VLM Coordinate Prediction:
  Ask the VLM to output normalized (x, y) coordinates in [0.0–1.0].
  Qwen2.5-VL has strong spatial training; reliable for icon-only elements.

Stage 4 — VLM Zone Estimation (coarse fallback):
  Divide the screen into a 3×3 grid of named zones.
  Ask the VLM "Which zone is [target] in?" — returns zone centre pixel.
  Confidence 0.50; used only when all other stages fail.

Note: Set-of-Mark (SoM) was evaluated but removed — small models (<7B)
consistently return region #1 regardless of the actual target location.
"""
import base64
import io
import json
import re
from typing import Optional, Tuple

from loguru import logger
from PIL import Image

from core.grounding.ocr_engine import OCREngine, OCRWord
from core.grounding.som_engine import SoMEngine


# ── VLM prompt templates ──────────────────────────────────────────────────────
# These prompts are tuned for Qwen2.5-VL which is much better at spatial
# reasoning than general VLMs. Keep prompts short — the model is concise.

_VLM_TEXT_LABEL_PROMPT = (
    'In this screenshot, I need to click on: "{target}".\n'
    "What is the EXACT visible text of that UI element?\n"
    "Reply with ONLY the text, 1-4 words. E.g.: File  Save  New Tab  OK\n"
    "If it is an icon with no visible text, reply exactly: ICON"
)

_VLM_ZONE_PROMPT = (
    'In this screenshot, where is "{target}"?\n'
    "Divide the screen into 9 zones:\n"
    "1=top-left  2=top-center  3=top-right\n"
    "4=mid-left  5=center      6=mid-right\n"
    "7=bot-left  8=bot-center  9=bot-right\n"
    "Reply with ONLY a single digit (1-9)."
)

# Stage 4 coordinate prompt — works with Qwen2.5-VL's spatial training.
# The model outputs normalized floats; we multiply back to screen pixels.
_VLM_COORD_PROMPT = (
    'In this screenshot, find the UI element: "{target}"\n\n'
    "Output ONLY this JSON and nothing else:\n"
    '{{"x": <float 0.0-1.0>, "y": <float 0.0-1.0>, "confidence": <float 0.0-1.0>, "found": true|false}}\n\n'
    "Rules:\n"
    "- x=0.0 is LEFT edge, x=1.0 is RIGHT edge\n"
    "- y=0.0 is TOP edge, y=1.0 is BOTTOM edge\n"
    "- x,y must be the CENTER of the element\n"
    "- If not visible: {{\"x\":0.0,\"y\":0.0,\"confidence\":0.0,\"found\":false}}"
)


class HybridGroundingEngine:
    """
    Three-stage UI element locator combining OCR precision with VLM semantics.
    No hardcoded coordinates. Works on any screen and any application.
    """

    def __init__(self, vlm_client=None):
        self.ocr = OCREngine()
        self.vlm = vlm_client
        self.som = SoMEngine()
        self._ocr_available = self.ocr.is_available()
        if not self._ocr_available:
            logger.warning(
                "[Hybrid] tesseract not found — OCR stages disabled. "
                "Install with: sudo apt-get install tesseract-ocr"
            )

    # ── public API ────────────────────────────────────────────────────────────

    def locate(
        self,
        target: str,
        image: Image.Image,
        image_b64: str,
        screen_w: int,
        screen_h: int,
        display_w: int,
        display_h: int,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Locate a UI element by natural language description.

        Args:
            target:    e.g. "the Save button", "browser address bar"
            image:     PIL image of the screen (display_w × display_h)
            image_b64: base64-encoded JPEG of image (for VLM calls)
            screen_w/h:  actual screen resolution (for scaling back)
            display_w/h: resolution of `image` (may be smaller than screen)

        Returns:
            (x, y, confidence, method) in *screen* coordinates, or None.
            method is one of: "ocr_direct", "ocr_semantic", "vlm_zone"
        """
        scale_x = screen_w / display_w
        scale_y = screen_h / display_h

        # Extract OCR words once — reused across stages
        words = self.ocr.extract(image) if self._ocr_available else []

        # ── Stage 1: OCR direct ───────────────────────────────────────────────
        if words:
            result = self._stage1_ocr_direct(target, words, scale_x, scale_y)
            if result:
                return result

        # ── Stage 2: VLM semantic → OCR ──────────────────────────────────────
        if self.vlm and words:
            result = self._stage2_vlm_semantic_ocr(
                target, image_b64, words, scale_x, scale_y
            )
            if result:
                return result

        # ── Stage 3: VLM Coordinate Prediction ───────────────────────────────
        # Ask VLM directly for normalized (0-1) coordinates of the target.
        # More reliable than SoM for small models — no region-number reasoning needed.
        # NOTE: SoM removed — it consistently fails with models <7B because the model
        # defaults to region #1 regardless of the actual target location.
        if self.vlm:
            result = self._stage4_vlm_coords(
                target, image_b64, screen_w, screen_h
            )
            if result:
                return result

        # ── Stage 4: VLM zone estimation (coarse fallback) ───────────────────
        if self.vlm:
            result = self._stage5_vlm_zone(
                target, image_b64, screen_w, screen_h, display_w, display_h
            )
            if result:
                return result

        return None

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage1_ocr_direct(
        self,
        target: str,
        words: list,
        scale_x: float,
        scale_y: float,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Try to find the target text directly on screen via OCR.
        Works for "File menu", "Save button", "Cancel" etc.
        """
        # Strip common action/role words to get the core text
        query = _strip_role_words(target)
        match = self.ocr.find_text(words, query, threshold=0.65)
        if match:
            x = int(match.cx * scale_x)
            y = int(match.cy * scale_y)
            logger.info(
                f"[Hybrid/S1-OCR-Direct] '{target}' → '{match.text}' "
                f"at display({match.cx},{match.cy}) → screen({x},{y})"
            )
            return (x, y, 0.95, "ocr_direct")
        logger.debug(f"[Hybrid/S1] No direct OCR match for '{query}'")
        return None

    def _stage2_vlm_semantic_ocr(
        self,
        target: str,
        image_b64: str,
        words: list,
        scale_x: float,
        scale_y: float,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Ask VLM for the text label of the target, then locate that label via OCR.
        Works for semantically described elements ("the accept cookies button").
        """
        try:
            prompt = _VLM_TEXT_LABEL_PROMPT.format(target=target)
            resp = self.vlm.query_vlm(
                prompt=prompt,
                image_base64=image_b64,
                max_tokens=15,
                temperature=0.05,
            )
            label = resp.content.strip().strip('"\'').strip()
            logger.debug(f"[Hybrid/S2] VLM text label for '{target}': '{label}'")

            if not label or label.upper() == "ICON" or len(label) > 40:
                return None

            # Remove VLM verbosity if it gave a sentence instead of a label
            label = label.split("\n")[0].split(".")[0].strip()

            match = self.ocr.find_text(words, label, threshold=0.60)
            if match:
                x = int(match.cx * scale_x)
                y = int(match.cy * scale_y)
                logger.info(
                    f"[Hybrid/S2-VLM-OCR] '{target}' → VLM said '{label}' "
                    f"→ OCR found '{match.text}' at screen({x},{y})"
                )
                return (x, y, 0.85, "ocr_semantic")
        except Exception as e:
            logger.warning(f"[Hybrid/S2] Error: {e}")
        return None

    def _stage3_som(
        self,
        target: str,
        image: Image.Image,
        screen_w: int,
        screen_h: int,
        display_w: int,
        display_h: int,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Stage 3: Set-of-Mark (SoM) region selection.
        """
        try:
            regions = self.som.detect_regions(image)
            if not regions:
                logger.debug("[Hybrid/S3-SoM] No candidate regions detected.")
                return None
            
            annotated_img = self.som.annotate(image, regions)
            buf = io.BytesIO()
            annotated_img.save(buf, format="JPEG", quality=85)
            annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            prompt = (
                f"Look at this screenshot where interactive elements are outlined and numbered.\n"
                f"I want to select the element described as: \"{target}\".\n"
                f"Which number box (1 to {len(regions)}) corresponds to that element?\n"
                f"Reply with ONLY the integer number, e.g. '7'."
            )
            
            resp = self.vlm.query_vlm(
                prompt=prompt,
                image_base64=annotated_b64,
                max_tokens=5,
                temperature=0.05
            )
            
            index = self.som.parse_region_number(resp.content, len(regions))
            if index is not None:
                region = self.som.get_region(regions, index)
                if region:
                    scale_x = screen_w / display_w
                    scale_y = screen_h / display_h
                    x = int(region.cx * scale_x)
                    y = int(region.cy * scale_y)
                    logger.info(
                        f"[Hybrid/S3-SoM] '{target}' → Selected Region #{index} "
                        f"at screen({x},{y})"
                    )
                    return (x, y, 0.75, "som")
            else:
                logger.debug(f"[Hybrid/S3-SoM] VLM responded with invalid index: '{resp.content}'")
        except Exception as e:
            logger.warning(f"[Hybrid/S3-SoM] Error: {e}")
        return None

    def _stage4_vlm_coords(
        self,
        target: str,
        image_b64: str,
        screen_w: int,
        screen_h: int,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Stage 4: VLM direct coordinate prediction with normalized coordinates [0.0-1.0].
        Qwen2.5-VL has strong spatial training; this stage works well with it.
        """
        try:
            prompt = _VLM_COORD_PROMPT.format(target=target)
            resp = self.vlm.query_vlm(
                prompt=prompt,
                image_base64=image_b64,
                max_tokens=80,
                temperature=0.0,
            )

            text = resp.content.strip()

            # Strip reasoning tokens that thinking models emit before the answer
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if not json_match:
                logger.debug(f"[Hybrid/S4] No JSON in response: '{text[:80]}'")
                return None

            # Tolerate trailing commas before closing brace
            json_str = re.sub(r",\s*([\]}])", r"\1", json_match.group())
            data = json.loads(json_str)

            if not data.get("found", False):
                return None

            x_norm = float(data.get("x", 0.0))
            y_norm = float(data.get("y", 0.0))
            conf = float(data.get("confidence", 0.65))

            # Reject obviously wrong coordinates
            if not (0.0 < x_norm < 1.0 and 0.0 < y_norm < 1.0):
                logger.debug(f"[Hybrid/S4] Coordinates out of range: ({x_norm},{y_norm})")
                return None

            x = int(x_norm * screen_w)
            y = int(y_norm * screen_h)
            logger.info(
                f"[Hybrid/S4-VLM-Coords] '{target}' → norm({x_norm:.3f},{y_norm:.3f}) "
                f"→ screen({x},{y}) conf={conf:.2f}"
            )
            return (x, y, conf, "vlm_coords")
        except Exception as e:
            logger.warning(f"[Hybrid/S4-VLM-Coords] Error: {e}")
        return None

    def _stage5_vlm_zone(
        self,
        target: str,
        image_b64: str,
        screen_w: int,
        screen_h: int,
        display_w: int,
        display_h: int,
    ) -> Optional[Tuple[int, int, float, str]]:
        """
        Ask VLM which 3×3 zone the target is in, return zone center.
        Coarse but honest — confidence 0.50.
        """
        try:
            prompt = _VLM_ZONE_PROMPT.format(target=target)
            resp = self.vlm.query_vlm(
                prompt=prompt,
                image_base64=image_b64,
                max_tokens=5,
                temperature=0.05,
            )
            raw = resp.content.strip()
            nums = re.findall(r'\b([1-9])\b', raw)
            if not nums:
                logger.warning(f"[Hybrid/S5] No zone number in VLM response: '{raw}'")
                return None

            zone = int(nums[0])
            x, y = _zone_center(zone, screen_w, screen_h)
            logger.info(f"[Hybrid/S5-VLM-Zone] '{target}' → zone {zone} → screen({x},{y})")
            return (x, y, 0.50, "vlm_zone")
        except Exception as e:
            logger.warning(f"[Hybrid/S5] Error: {e}")
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

# Common words that describe the ROLE of an element, not its text label
_ROLE_WORDS = {
    "the", "a", "an", "button", "btn", "menu", "bar", "tab", "panel",
    "icon", "link", "field", "box", "input", "area", "window", "dialog",
    "click", "open", "close", "toggle", "select", "choose", "find", "locate",
    "at", "on", "in", "of", "for", "with", "to", "from", "top", "bottom",
    "left", "right", "center", "middle", "upper", "lower", "primary", "main",
    "current", "active", "focused", "corner", "side", "edge", "near",
}


def _strip_role_words(text: str) -> str:
    """Remove common role/position descriptors to get the core text label."""
    tokens = text.lower().split()
    core = [t for t in tokens if t not in _ROLE_WORDS]
    return " ".join(core) if core else text


def _zone_center(zone: int, w: int, h: int) -> Tuple[int, int]:
    """Return the screen-coordinate center pixel of the given 1-9 zone."""
    col = (zone - 1) % 3       # 0=left, 1=center, 2=right
    row = (zone - 1) // 3      # 0=top,  1=middle, 2=bottom
    x = int(w * (col * 2 + 1) / 6)
    y = int(h * (row * 2 + 1) / 6)
    return x, y
