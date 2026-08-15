# agents/coords.py
"""Turning a VLM's answer into a screen pixel.

UI-TARS answers a grounding question in whichever format it feels like — an
action call, JSON, a point tag, a bare bounding box — and in whichever value
scale (0-1, 0-1000, or absolute pixels of the resized image it was shown).
Getting this conversion wrong puts every click in the wrong place, so the whole
conversion lives here, apart from the grounding cascade that uses it, with one
parser per format and a single shared scaling rule (`CoordSpace`).

The scale is pinned by config.VLM_COORD_SPACE; re-calibrate with
tests/live/test_vlm_coordinates.py after changing the served model.
"""
import json
import re
from dataclasses import dataclass

from loguru import logger

# '[[', '[[[', '(', '[(', etc.
_BOX_OPEN = r"[\[\(]{0,4}"
_BOX_CLOSE = r"[\]\)]{0,4}"


class _VLMSaysNotFound(Exception):
    """VLM answered found=false — stop parsing, the element is not on screen."""


def _qwen_resize_dim(dim: int, factor: int = 28) -> int:
    """qwen2.5-VL smart_resize rounds each image side to a multiple of `factor`
    (28 = patch_size 14 × merge_size 2). Our screenshots are always far below the
    model's max_pixels (12,845,056 px, per its preprocessor_config.json), so the
    pixel-budget branch of smart_resize never fires — the model simply sees the
    sent image rounded to the nearest /28, and (being qwen2.5-VL based) emits
    ABSOLUTE pixel coordinates in THAT resized space. Mapping back to screen
    therefore divides by the /28-rounded dimension, not the raw sent dimension.
    """
    return max(factor, int(round(dim / factor)) * factor)


@dataclass
class CoordSpace:
    """Maps coordinate values from one VLM answer to screen pixels.

    UI-TARS-1.5 (qwen2.5-VL) emits ABSOLUTE pixel coordinates in the smart-resized
    image space. This object carries the dimensions + configured convention so
    every parser applies the exact same conversion rules. Pixel divisions use the
    /28-rounded (smart-resized) dimension via _disp(), matching what the model saw.
    """

    screen_w: int
    screen_h: int
    display_w: int   # size of the image sent to the VLM (fallback: screen)
    display_h: int
    mode: str        # config.VLM_COORD_SPACE: "auto" | "pixels" | "norm1000"

    def _disp(self, dim: int) -> int:
        """Divisor for pixel-valued coords: the /28 smart-resized dimension."""
        return _qwen_resize_dim(dim) if dim > 1 else dim

    def px_to_screen(self, px: float, py: float) -> tuple[int, int]:
        """Scale resized-image pixels to screen pixels, clamped to screen bounds."""
        return (
            min(int(px / self._disp(self.display_w) * self.screen_w), self.screen_w - 1),
            min(int(py / self._disp(self.display_h) * self.screen_h), self.screen_h - 1),
        )

    def scale_x(self, val: float) -> int:
        return self._scale(val, self.screen_w, self.display_w)

    def scale_y(self, val: float) -> int:
        return self._scale(val, self.screen_h, self.display_h)

    def _scale(self, val: float, screen_dim: int, display_dim: int) -> int:
        """Convert a single VLM coordinate to screen pixels.

        A pinned mode ("pixels" / "norm1000") is applied deterministically.
        In "auto": if val > 1000 it must be a resized-space pixel, not 0-1000;
        if display_dim is available and val fits within it, treat as pixel;
        otherwise fall back to 0-1000 normalised interpretation. All pixel
        divisions use the /28 smart-resized dimension (_disp).
        """
        rd = self._disp(display_dim)
        if self.mode == "pixels" and display_dim > 1:
            return min(int(val / rd * screen_dim), screen_dim - 1)
        if self.mode == "norm1000" and val <= 1000:
            return min(int(val / 1000 * screen_dim), screen_dim - 1)
        if val > 1000:
            return min(int(val / rd * screen_dim), screen_dim - 1)
        if display_dim > 1 and val <= display_dim:
            return min(int(val / rd * screen_dim), screen_dim - 1)
        return int(val / 1000 * screen_dim)




def parse_coords(
    text: str, screen_w: int, screen_h: int,
    display_w: int = 0, display_h: int = 0,
) -> tuple[int, int, float] | None:
    """Parse VLM output into screen pixel coordinates (x, y, confidence).

    display_w/display_h: size of the image that was sent to the VLM.
    When the model returns pixel-valued coordinates they are in the display
    image's coordinate space and must be scaled to screen space:
        screen_x = (pixel_x / display_w) * screen_w
    If display dims are not provided, pixel coords are used as-is (clipped).

    Handles the formats UI-TARS may emit — one parser method per format:
      action: click(start_box='[[x1,y1,x2,y2]]')   → _parse_click_box
      JSON:   {"x": 0.5, "y": 0.3, ...}            → _parse_json_coords
      Point:  <point>500 300</point>               → _parse_point_tag
      BBox:   (x1,y1),(x2,y2)                      → _parse_paren_bbox
    Value-scale interpretation (0-1 / 0-1000 / display pixels) is shared:
    see CoordSpace and _xy_to_screen.
    """
    # Explicit not-found answer — fast, clean exit (no format warning)
    if "not_found" in text.lower():
        logger.debug("[GROUNDING/S2] VLM reports element not visible")
        return None

    space = coord_space(screen_w, screen_h, display_w, display_h)
    try:
        for parse in (
            _parse_click_box,
            _parse_json_coords,
            _parse_point_tag,
            _parse_paren_bbox,
        ):
            result = parse(text, space)
            if result:
                return result
    except _VLMSaysNotFound:
        return None

    logger.debug(f"[GROUNDING/S2] Unrecognised VLM format: '{text[:100]}'")
    return None

def coord_space(
    screen_w: int, screen_h: int, display_w: int, display_h: int,
) -> "CoordSpace":
    """Build the coordinate-mapping context for one VLM answer.

    The coordinate convention of the served model comes from config.
    "auto" keeps the value-range heuristics in CoordSpace/_xy_to_screen;
    pin to "pixels" or "norm1000" in config.py (calibrate once with
    tests/live/test_vlm_coordinates.py) for deterministic parsing.
    """
    try:
        import config as _cfg
        mode = str(getattr(_cfg, "VLM_COORD_SPACE", "auto")).lower()
    except Exception:
        mode = "auto"
    # Effective display dimensions for pixel-to-screen scaling.
    # Fall back to screen dims (identity scale) when not provided.
    return CoordSpace(
        screen_w=screen_w, screen_h=screen_h,
        display_w=display_w if display_w > 1 else screen_w,
        display_h=display_h if display_h > 1 else screen_h,
        mode=mode,
    )

def _xy_to_screen(
    space: "CoordSpace", xv: float, yv: float,
    conf: float, conf_px: float, *,
    strict_1000: bool, honor_pinned: bool = True,
) -> tuple[int, int, float]:
    """Map one (x, y) pair to screen pixels via the value-range tiers.

    Tier order (first match wins):
      1. both values in 0-1     → normalised floats (most accurate)
      2. pinned config mode     → deterministic per-axis scaling
      3. both values in 0-1000  → UI-TARS native 0-1000 scale
      4. anything larger        → raw display pixels (conf_px applies)
    strict_1000: the 0-1000 tier additionally requires values > 1
    (used by the JSON parser, whose 0-1 tier is exact).
    honor_pinned: the salvaged-numbers JSON fallback skips tier 2,
    matching the original behaviour of that path.
    """
    if 0.0 <= xv <= 1.0 and 0.0 <= yv <= 1.0:
        return (int(xv * space.screen_w), int(yv * space.screen_h), conf)
    if honor_pinned and space.mode in ("pixels", "norm1000"):
        return (space.scale_x(xv), space.scale_y(yv), conf)
    in_thousand = (
        (1.0 < xv <= 1000 and 1.0 < yv <= 1000) if strict_1000
        else (xv <= 1000 and yv <= 1000)
    )
    if in_thousand:
        return (
            int(xv / 1000 * space.screen_w),
            int(yv / 1000 * space.screen_h),
            conf,
        )
    sx, sy = space.px_to_screen(xv, yv)
    return (sx, sy, conf_px)

def _parse_click_box(text: str, space: "CoordSpace",
) -> tuple[int, int, float] | None:
    """UI-TARS native action format: click(start_box='[[x1, y1, x2, y2]]').

    Bracket style varies wildly: '[[', '[[[', '(', '[(', etc., and the
    model sometimes emits a 2-value centre point instead of a full bbox.
    Scale interpretation: the prompt asks for 0-1000 but the OVMS-served
    INT4 UI-TARS often emits coordinates in the screenshot's pixel space
    instead — CoordSpace.scale_x/y applies the per-value heuristic.
    """
    m = re.search(
        r"(?:click|tap)\s*\(\s*start_box\s*=\s*'?" + _BOX_OPEN +
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
        + _BOX_CLOSE + r"'?\s*\)",
        text,
    )
    if m:
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (space.scale_x(cx), space.scale_y(cy), 0.90)

    # 2-value form: model emits [[cx, cy]] or (cx, cy) instead of a full bbox.
    m = re.search(
        r"(?:click|tap)\s*\(\s*start_box\s*=\s*'?" + _BOX_OPEN +
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
        r"\s*[\]\)\s']+",
        text,
    )
    if m:
        cx, cy = float(m.group(1)), float(m.group(2))
        return (space.scale_x(cx), space.scale_y(cy), 0.80)
    return None

def _parse_json_coords(text: str, space: "CoordSpace",
) -> tuple[int, int, float] | None:
    """JSON block — x/y may be:
      0-1 normalised floats  (model followed instructions)
      1-1000 integers        (UI-TARS native 0-1000 scale leaked into JSON)
      > 1000                 (raw display pixels — scale by display dims)
    Also handles malformed JSON like {"x": 658, 294} (missing "y": key)
    by salvaging the first two numbers at reduced confidence.
    """
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        return None
    raw_json = re.sub(r",\s*([\]}])", r"\1", m.group())
    try:
        data = json.loads(raw_json)
        if not data.get("found", True):
            raise _VLMSaysNotFound
        xv, yv = float(data["x"]), float(data["y"])
        conf = float(data.get("confidence", 0.7))
        return _xy_to_screen(space, xv, yv, conf, conf, strict_1000=True)
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: try to extract two numbers from the JSON string
        nums = re.findall(r'[\d]+(?:\.\d+)?', raw_json)
        if len(nums) >= 2:
            try:
                xv, yv = float(nums[0]), float(nums[1])
                return _xy_to_screen(
                    space, xv, yv, 0.65, 0.65,
                    strict_1000=True, honor_pinned=False,
                )
            except ValueError:
                pass
        return None

def _parse_point_tag(text: str, space: "CoordSpace",
) -> tuple[int, int, float] | None:
    """<point>cx cy</point> — 0-1000 scale or display-space pixels."""
    m = re.search(r'<point>\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*</point>', text)
    if not m:
        return None
    px, py = float(m.group(1)), float(m.group(2))
    return _xy_to_screen(space, px, py, 0.85, 0.75, strict_1000=False)

def _parse_paren_bbox(text: str, space: "CoordSpace",
) -> tuple[int, int, float] | None:
    """(x1,y1),(x2,y2) bounding box — take the centre point."""
    m = re.search(
        r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\),\s*\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        text
    )
    if not m:
        return None
    x1, y1, x2, y2 = (float(m.group(i)) for i in range(1, 5))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return _xy_to_screen(space, cx, cy, 0.85, 0.75, strict_1000=False)

