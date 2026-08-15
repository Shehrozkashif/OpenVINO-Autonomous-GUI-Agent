# agents/grounding.py
"""UI Grounding Agent — locates UI elements by natural language description.

Three-stage pipeline:
  Stage 0 — Windows UIA:  Accessibility tree bounding box lookup    (conf 1.0 / 0.85+)
  Stage 1 — OCR Direct:   Fuzzy-match target text in live OCR       (conf 0.95)
  Stage 2 — VLM Coords:   UI-TARS predicts normalized (x,y) directly (conf ~0.80)

If all stages fail, returns found=False so the orchestrator can retry.
"""
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import imagehash
from loguru import logger
from PIL import Image

from agents import coords
from core.inference import InferenceClient
from desktop.capture import OCR_THUMB, ScreenCapture, _screen_size
from desktop.ocr import OCREngine
from desktop.uia import find_element as _uia_find
from desktop.uia import is_available as _uia_ok

_ocr_grounding_disabled_logged = False


def _once(fn):
    """Wrap fn so it runs at most once, on first call, and remembers its result.

    Lets an expensive value (an OCR pass, a base64 encode of the screen) be
    handed to a cascade of stages that may never ask for it, without any stage
    having to know whether an earlier one already paid for it.
    """
    box: list = []

    def get():
        if not box:
            box.append(fn())
        return box[0]

    return get


def _ocr_grounding_disabled() -> bool:
    """True when AGENT_DISABLE_OCR=1 turns off the OCR (Stage 1) grounding stage.

    Mirror of AGENT_DISABLE_UIA: set BOTH to force every grounding call down to
    the VLM (Stage 2), so the visual grounding path can be exercised in
    isolation on a use case where UIA/OCR would otherwise win first. Only the
    grounding STAGE is gated — OCR is still used for planner screen context and
    reflection, which are not grounding.
    """
    if os.environ.get("AGENT_DISABLE_OCR", "").strip().lower() not in ("1", "true", "yes"):
        return False
    global _ocr_grounding_disabled_logged
    if not _ocr_grounding_disabled_logged:
        _ocr_grounding_disabled_logged = True
        logger.info("[OCR] Stage 1 grounding disabled via AGENT_DISABLE_OCR — VLM takes over")
    return True


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


# ── Agent ─────────────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    """Where an element was found on screen (or found=False if it wasn't).

    (x, y) is the click point; `method` records which stage found it
    (uia / ocr / vlm) for logging and dead-point bookkeeping.
    """

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
        self._cache: dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, target: str, screen_hash: str) -> tuple[int, int, float, str, str] | None:
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

    def drop(self, target: str):
        """Remove one target's entry (e.g. its coordinate was proven inert)."""
        self._cache.pop(target, None)


# Bracket styles UI-TARS uses around click(start_box=...) coordinates:
class UIGroundingAgent:
    """Locates UI elements by natural language description.

    Stage 1 — OCR:  fast, free, pixel-perfect for text-labeled elements.
    Stage 2 — VLM:  UI-TARS-1.5-7B direct coordinate prediction for everything else.

    Accepts any client that implements InferenceClient (OVMSClient, etc.).
    """

    # Must match every other OCR pass (OCR_THUMB) so cache entries are shared
    # and element_type tags set by capture_snapshot() flow into grounding.
    _DISPLAY_W = OCR_THUMB[0]
    _DISPLAY_H = OCR_THUMB[1]

    def __init__(
        self,
        client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.5,
        ocr: Optional["OCREngine"] = None,
    ):
        self.client = client
        self.capturer = capturer
        self.cache = ElementCache()
        self.ocr = ocr if ocr is not None else OCREngine()
        self.min_confidence = min_confidence
        self.screen_w, self.screen_h = _screen_size()
        # Coordinates proven inert by the reflector (click → phash delta=0),
        # keyed by (target, screen phash). While the screen is in that exact
        # state, these points are never served again — grounding falls through
        # to the next stage instead. See mark_dead().
        self._dead: dict[tuple[str, str], list[tuple[int, int]]] = {}
        # Screen hash each target was last grounded on — mark_dead() blacklists
        # against THIS state, so a wrong click that changed the screen (e.g.
        # opened another app's window) still poisons the point on the screen
        # where it will be looked up again.
        self._ground_hash: dict[str, str] = {}
        # Negative cache: (target, screen phash) pairs where EVERY stage
        # already failed. Grounding is deterministic for an unchanged screen,
        # so re-running the cascade (VLM call + rephrase LLM + tree searches,
        # 15-20 s) to reproduce a known miss is pure latency. A new screen
        # state is a new hash; mark_dead() also invalidates the entry.
        self._no_find: dict[tuple[str, str], float] = {}
        # Points proven to land in ANOTHER APP's window (clicking them flipped
        # the foreground away from the task's anchor app). Unlike delta=0
        # points these are wrong regardless of screen state — the foreign
        # window doesn't stop being foreign because pixels elsewhere changed —
        # so they are blacklisted for EVERY target until the next task starts.
        # Live: the VLM re-emitted the exact same Edge-window coordinate for
        # 'New meeting' across three screen changes, opening Edge each time.
        self._dead_foreign: list[tuple[int, int]] = []
        logger.info(
            f"[GROUNDING] Ready. Screen: {self.screen_w}×{self.screen_h}. "
            f"OCR: {'on' if self.ocr.is_available() else 'off (pip install rapidocr-onnxruntime)'}"
        )

    # ── public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _near_dead(x: int, y: int, dead, radius: int = 10) -> bool:
        return any(abs(x - dx) <= radius and abs(y - dy) <= radius for dx, dy in dead)

    def is_dead_point(self, x: int, y: int) -> bool:
        """Was (x,y) proven inert (delta=0) on the CURRENT screen, or proven
        to belong to another app's window at any point in this task?

        Read path for EXPLICIT-coordinate clicks (visual planner): they carry
        raw pixels and never call ground(), so they bypassed the dead-point
        blacklist on both read and write — live: the VLM proposed the same
        inert point across three replanned subtasks and the agent clicked it
        NINE times, delta=0 every time. Their marks live under the shared
        '[visual]' pseudo-target (a dead pixel is dead regardless of intent).
        """
        foreign = getattr(self, "_dead_foreign", [])
        if foreign and self._near_dead(x, y, foreign):
            return True
        try:
            display = self.capturer.capture()
            display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
            screen_hash = str(imagehash.phash(display))
        except Exception:
            return False
        dead = self._dead.get(("[visual]", screen_hash), [])
        return bool(dead) and self._near_dead(x, y, dead)

    def mark_dead(self, target: str, x: int, y: int, foreign: bool = False):
        """Record that clicking (x,y) for `target` provably changed nothing.

        Called by the orchestrator when the reflector's frame comparison shows
        phash delta=0 after the click — hard evidence the point is inert in the
        CURRENT screen state. The screen is still in that state (nothing
        changed), so hashing it now keys the blacklist to exactly the state
        where the click was proven dead. On any other screen the same
        coordinate stays usable.

        foreign=True: the click flipped the foreground to a DIFFERENT app —
        the point belongs to another app's window. That fact survives screen
        changes, so the point is additionally blacklisted for ALL targets on
        ALL screens until clear_dead_points() (next task).
        """
        if not target:
            return
        if foreign:
            fdead = getattr(self, "_dead_foreign", None)
            if fdead is None:
                fdead = self._dead_foreign = []
            if not self._near_dead(x, y, fdead):
                fdead.append((x, y))
                del fdead[:-32]   # bound task-lifetime growth
        # Prefer the hash of the screen this target was GROUNDED on: when the
        # wrong click changed the screen (opened another window), hashing the
        # current screen would key the blacklist to the aftermath state — a
        # state where the point is never looked up — making it useless.
        screen_hash = self._ground_hash.get(target.lower())
        if screen_hash is None:
            try:
                display = self.capturer.capture()
                display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
                screen_hash = str(imagehash.phash(display))
            except Exception:
                return
        self._dead.setdefault((target.lower(), screen_hash), []).append((x, y))
        if len(self._dead) > 64:
            self._dead.pop(next(iter(self._dead)))
        # The cached entry points at the dead coordinate — drop it so the
        # retry re-grounds instead of re-serving the proven-inert point.
        # The negative cache too: with a new dead point the stages may now
        # fall through to a DIFFERENT (live) candidate.
        self.cache.drop(target)
        self._no_find.pop((target.lower(), screen_hash), None)
        logger.info(
            f"[GROUNDING] ({x},{y}) marked DEAD for '{target}' on current "
            f"screen — next attempt must find a different point"
        )

    def clear_dead_points(self):
        """Forget every blacklist at the start of a new task.

        Dead points, foreign-window points, negative-cache misses and
        grounding-origin hashes all describe THIS task's screens; letting them
        leak into the next task can veto a coordinate that is perfectly valid
        there (the foreign-window list especially — the next task may be
        anchored to the very app these points belong to).
        """
        self._dead.clear()
        self._no_find.clear()
        self._ground_hash.clear()
        fdead = getattr(self, "_dead_foreign", None)
        if fdead:
            fdead.clear()

    def _prepare_screen(self, target: str):
        """Shared prelude of ground()/ground_fast(): capture, thumbnail,
        scale factors, screen hash (recorded as the target's grounding
        origin — mark_dead() keys the blacklist to it), and that screen's
        dead points for this target.
        """
        display = self.capturer.capture().copy()
        display.thumbnail((self._DISPLAY_W, self._DISPLAY_H), Image.LANCZOS)
        dw, dh = display.width, display.height
        # Guard against zero-sized thumbnails (can happen on headless/virtual displays)
        scale_x = self.screen_w / dw if dw > 0 else 1.0
        scale_y = self.screen_h / dh if dh > 0 else 1.0

        screen_hash = str(imagehash.phash(display))
        self._ground_hash[target.lower()] = screen_hash
        if len(self._ground_hash) > 128:
            self._ground_hash.pop(next(iter(self._ground_hash)))
        # Screen-scoped inert points for this target PLUS the task-scoped
        # foreign-window points (wrong for every target on every screen).
        dead = list(self._dead.get((target.lower(), screen_hash), []))
        dead += getattr(self, "_dead_foreign", [])
        return display, scale_x, scale_y, screen_hash, dead

    def ground(self, target: str, max_retries: int = 1) -> GroundingResult:
        """Locate a UI element by natural language description.
        Returns GroundingResult with screen (x, y). found=False if all stages fail.

        On failure, asks the LLM for 3 alternative phrasings and retries each
        before giving up — covers cases where the model used different label text
        than what OCR actually detected on screen.
        """
        start = time.time()
        display, scale_x, scale_y, screen_hash, dead = self._prepare_screen(target)
        if (target.lower(), screen_hash) in self._no_find:
            logger.info(
                f"[GROUNDING] '{target}' already failed every stage on this "
                f"exact screen — cached miss (skipping the cascade)"
            )
            return GroundingResult(x=0, y=0, confidence=0.0, found=False,
                                   latency_ms=(time.time() - start) * 1000,
                                   target=target, method="cached_miss")
        cached = self.cache.get(target, screen_hash)
        if cached and dead and self._near_dead(cached[0], cached[1], dead):
            self.cache.drop(target)
            cached = None
        if cached:
            x, y, conf, method, element_type = cached
            # Canonical "[GROUNDING] '…' → (x,y) conf=… method=…" format — the
            # UI event bus parses these lines (ui/events.py _RX_LOCATED).
            logger.info(f"[GROUNDING] '{target}' -> ({x},{y}) "
                        f"conf={conf:.2f} method=cache/{method}")
            return GroundingResult(x=x, y=y, confidence=conf, found=True,
                                   latency_ms=(time.time() - start) * 1000,
                                   target=target, method=f"cache/{method}",
                                   element_type=element_type)

        # Deferred, not skipped. Both of these used to run right here, before
        # the cascade — so every grounding call paid for a full OCR pass (1-2 s
        # of ONNX) and a base64 encode of the whole screen even when Stage 0
        # answered from the accessibility tree in 80 ms and never looked at
        # either. Stage 1 and Stage 2 still get exactly the same values; they
        # are just computed at the moment a stage actually asks for them, and
        # memoised so a retry round does not repeat the work.
        get_b64 = _once(lambda: self._encode(display))
        get_words = _once(lambda: self.ocr.extract(display) if self.ocr.is_available() else [])

        for attempt in range(max_retries + 1):
            # Stage 2 (VLM) only on the first attempt: on shared-VRAM machines a
            # VLM call costs a 30-60s model swap, and re-asking it the identical
            # question on an unchanged screen returns the same answer anyway.
            result = self._locate(target, display, get_b64, get_words, scale_x, scale_y,
                                  use_vlm=(attempt == 0), dead=dead)
            if result:
                x, y, conf, method, element_type = result
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                self.cache.put(target, x, y, conf, method, screen_hash, element_type)
                logger.info(
                    f"[GROUNDING] '{target}' -> ({x},{y}) conf={conf:.2f} "
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
            # A rephrase is ALREADY a guess, so its OCR match must be near-
            # exact: every live rephrase hit below 0.9 was a different control
            # ('Create Meeting' -> 'Create a meeting link' @0.80 opened the
            # wrong dialog; 'Meeting Details New' -> 'New meeting Details
            # Save' @0.79 was a dead click), while correct hits score >=0.95.
            # Fuzzy-on-top-of-fuzzy compounds into confidently-wrong clicks.
            result = self._locate(alt, display, get_b64, get_words, scale_x, scale_y,
                                  use_vlm=False, dead=dead, ocr_threshold=0.9)
            if result:
                x, y, conf, method, element_type = result
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                self.cache.put(target, x, y, conf, f"rephrase/{method}", screen_hash, element_type)
                logger.info(f"[GROUNDING] '{target}' -> ({x},{y}) "
                            f"conf={conf:.2f} method=rephrase/{method} "
                            f"(as '{alt}')")
                return GroundingResult(x=x, y=y, confidence=conf,
                                       found=conf >= self.min_confidence,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method=f"rephrase/{method}",
                                       element_type=element_type)

        logger.warning(f"[GROUNDING] All stages failed for '{target}'")
        self._no_find[(target.lower(), screen_hash)] = time.time()
        if len(self._no_find) > 128:
            self._no_find.pop(next(iter(self._no_find)))
        return GroundingResult(x=0, y=0, confidence=0.0, found=False,
                               latency_ms=(time.time() - start) * 1000,
                               target=target, method="failed")

    def ground_fast(self, target: str, ocr_threshold: float = 0.65) -> GroundingResult:
        """Stage 0 + Stage 1 only (UIA + OCR) — no VLM call.

        Used where a target may legitimately be absent (e.g. scroll-to-find
        probing): a full ground() would spend 30-50 s in Stage 2 just to confirm
        not-found, whereas this returns immediately when UIA + OCR miss.

        ocr_threshold: minimum fuzzy-match score for the OCR stage. Callers
        whose hit leads to typing (set_value/select field grounding) pass a
        stricter bar — live, 'date' matched taskbar garbage 'Wd TE' at 0.67
        and the fallback ctrl+a-typed a date into a non-Teams surface.
        """
        start = time.time()
        display, scale_x, scale_y, _screen_hash, dead = self._prepare_screen(target)

        if _uia_ok():
            r = _uia_find(target)
            if r:
                x, y, conf = r
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                if self._near_dead(x, y, dead):
                    logger.info(
                        f"[GROUNDING/FAST] '{target}' → ({x},{y}) is a "
                        "known-dead point — skipping UIA hit"
                    )
                else:
                    return GroundingResult(x=x, y=y, confidence=conf, found=True,
                                           latency_ms=(time.time() - start) * 1000,
                                           target=target, method="uia",
                                           element_type="foreground_interactive")

        # OCR runs only now, past Stage 0. Every set_value/select step grounds
        # its field label through here purely to have a pixel fallback ready in
        # case the accessibility tree cannot focus the control — so on the
        # normal path, where UIA does focus it, this OCR was pure waste.
        words = self.ocr.extract(display) if self.ocr.is_available() else []
        if words and not _ocr_grounding_disabled():
            query = _strip_role_words(target)
            match = self.ocr.find_text(words, query, threshold=ocr_threshold)
            if match:
                x = int(match.cx * scale_x)
                y = int(match.cy * scale_y)
                if self._near_dead(x, y, dead):
                    logger.info(
                        f"[GROUNDING/FAST] '{target}' → ({x},{y}) is a "
                        "known-dead point — skipping OCR hit"
                    )
                    return GroundingResult(
                        x=0, y=0, confidence=0.0, found=False,
                        latency_ms=(time.time() - start) * 1000,
                        target=target, method="fast_dead")
                return GroundingResult(x=x, y=y, confidence=0.95, found=True,
                                       latency_ms=(time.time() - start) * 1000,
                                       target=target, method="ocr_direct",
                                       element_type=match.element_type)

        return GroundingResult(x=0, y=0, confidence=0.0, found=False,
                               latency_ms=(time.time() - start) * 1000,
                               target=target, method="failed")

    # ── grounding stages ──────────────────────────────────────────────────────

    def _locate(
        self,
        target: str,
        display: Image.Image,
        get_b64,
        get_words,
        scale_x: float,
        scale_y: float,
        use_vlm: bool = True,
        dead=(),
        ocr_threshold: float = 0.65,
    ) -> tuple[int, int, float, str, str] | None:
        # `dead` holds coordinates proven inert on this exact screen (a click
        # there changed zero pixels). A stage that lands on a dead point is
        # skipped so the next stage gets a chance to find a different point.

        # Stage 0: Windows UIAutomation — fast, pixel-perfect, no model needed
        # UIA elements are always interactive → element_type="foreground_interactive"
        if _uia_ok():
            r = _uia_find(target)
            if r:
                x, y, conf = r
                x = max(0, min(x, self.screen_w - 1))
                y = max(0, min(y, self.screen_h - 1))
                if self._near_dead(x, y, dead):
                    logger.info(f"[GROUNDING/S0-UIA] '{target}' → ({x},{y}) is a known-dead point — skipping stage")
                else:
                    logger.info(f"[GROUNDING/S0-UIA] '{target}' → screen({x},{y}) conf={conf:.2f}")
                    return (x, y, conf, "uia", "foreground_interactive")
            else:
                logger.debug(f"[GROUNDING/S0-UIA] '{target}' not found in UIA tree")

        # Stage 1: OCR direct fuzzy-match — carry element_type from the matched word
        # get_words() is where OCR actually runs. Reaching this line means Stage 0
        # had no answer, which is the only case that needs it.
        if not _ocr_grounding_disabled() and (words := get_words()):
            query = _strip_role_words(target)
            match = self.ocr.find_text(words, query, threshold=ocr_threshold)
            if match:
                x, y = int(match.cx * scale_x), int(match.cy * scale_y)
                if self._near_dead(x, y, dead):
                    logger.info(f"[GROUNDING/S1-OCR] '{target}' → ({x},{y}) is a known-dead point — skipping stage")
                else:
                    logger.info(f"[GROUNDING/S1-OCR] '{target}' → '{match.text}' → screen({x},{y})")
                    return (x, y, 0.95, "ocr_direct", match.element_type)
            else:
                logger.debug(f"[GROUNDING/S1] No OCR match for '{query}'")

        # Stage 2: VLM direct coordinate prediction
        # VLM is expected to hit interactive elements → element_type="foreground_interactive"
        if use_vlm and self.client:
            result = self._vlm_coords(target, get_b64(), int(display.width), int(display.height),
                                      avoid=dead)
            if result and not self._near_dead(result[0], result[1], dead):
                return result

        return None

    def _vlm_coords(
        self, target: str, img_b64: str,
        display_w: int = 0, display_h: int = 0,
        avoid: list | tuple = (),
    ) -> tuple[int, int, float, str, str] | None:
        """Ask UI-TARS for normalized (x,y) coordinates and scale to screen pixels.

        display_w/display_h: pixel dimensions of the image sent to the VLM.
        Needed to correctly scale pixel-valued JSON coordinates back to screen space.

        avoid: screen-pixel points already proven wrong for this target. The
        model runs at temperature 0, so re-asking the identical question
        returns the identical wrong point forever — the only way to a
        different answer is to change the question. The points are embedded
        in the prompt on the model's own 0-1000 scale.
        """
        prompt = _VLM_COORD_PROMPT.format(target=target)
        if avoid and self.screen_w and self.screen_h:
            pts = ", ".join(
                f"({int(ax / self.screen_w * 1000)}, {int(ay / self.screen_h * 1000)})"
                for ax, ay in list(avoid)[:8]
            )
            prompt += (
                f"\nIMPORTANT: the element is NOT at these 0-1000 scale "
                f"locations (already tried, wrong): {pts}. Pick a clearly "
                f"different location, or answer not_found() if the element "
                f"is not visible anywhere else."
            )
        try:
            resp = self.client.query_vlm(
                prompt=prompt,
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

            result = coords.parse_coords(text, self.screen_w, self.screen_h, display_w, display_h)
            if result:
                x, y, conf = result
                logger.info(f"[GROUNDING/S2-VLM] '{target}' -> screen({x},{y}) conf={conf:.2f}")
                return (x, y, conf, "vlm_coords", "foreground_interactive")
        except Exception as e:
            logger.warning(f"[GROUNDING/S2-VLM] Error: {e}")
        return None

    def _rephrase_targets(self, target: str) -> list[str]:
        """Ask the LLM for up to 3 alternative text labels for the same UI element.
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
