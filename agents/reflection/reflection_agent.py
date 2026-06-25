# agents/reflection/reflection_agent.py
"""
Reflection Agent — verifies that each action step succeeded.

Primary path: OCR the after-screenshot, pass visible text to the LLM
(config.LLM_MODEL).
  - A reasoning LLM handles conditional success logic accurately.
  - No image encoding needed — fast, cheap, high quality.

Fallback path: the VLM (UI-TARS) is used only when OCR returns fewer
than 3 meaningful words (icon-heavy screens, blank desktops).
"""
import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from PIL import Image

from agents.grounding.grounding_agent import OCREngine
from core.capture.screenshot import ScreenCapture
from core.protocols.a2a import ActionStep, InferenceClient


@dataclass
class ReflectionResult:
    success: bool
    confidence: float
    observation: str
    error_description: str
    should_retry: bool
    recovery_hint: str
    ocr_text: str = ""   # raw OCR from post-action screenshot, reused by orchestrator


# ── LLM reflection prompt (primary path) ─────────────────────────────────────
# Sent to the LLM with OCR text. Much more reliable than asking a grounding
# VLM to reason about conditional success criteria.

_LLM_SYSTEM = "You are a desktop automation verifier. Reply with valid JSON only."

_LLM_REFLECTION_PROMPT = """\
Verify whether a desktop automation step succeeded.

STEP     : {action_type} — {description}
EXPECTED : {verification}
SCREEN TEXT (OCR after action): {ocr_text}

Rules — base your verdict on evidence in the screen text:
  click/right_click  : SUCCEED only if a new menu, dialog, window title, or
                       focused element appeared that is new after the action.
                       FAIL if screen text shows no new UI elements.
  double_click       : SUCCEED if a file opened, app launched, or item selected
                       (window title changed or new content appeared).
  type               : SUCCEED if the typed text (or close variant) is visible
                       in a context consistent with an active input field.
                       FAIL if the text appears only in unrelated logs or chat.
  key_press "enter"  : SUCCEED if the intended effect occurred (command ran,
                       dialog confirmed, folder renamed, search executed).
                       In a TERMINAL: shell silence means success — a new empty
                       prompt line with NO error text = the command SUCCEEDED.
                       Only FAIL if error text is visible (denied, not found, …).
  key_press "super"  : SUCCEED if a search bar or launcher panel is visible.
  key_press "escape" : SUCCEED if a dialog or menu closed.
  hotkey ctrl+s      : SUCCEED if no error dialog appeared. Silence = success —
                       a named file saves with no visible dialog. A Save-As dialog
                       appearing is also success. FAIL only if an error is visible.
  hotkey ctrl+l      : SUCCEED if an address bar or URL field is highlighted.
  scroll             : SUCCEED if any text content changed (page shifted).
  Ambiguous case     : success=false, confidence=0.5, should_retry=true.
                       Do NOT default to success when uncertain.

Reply JSON only:
{{"success": true/false, "confidence": 0.0-1.0, "observation": "one sentence", \
"error_description": "", "should_retry": true/false, "recovery_hint": ""}}"""


# ── VLM reflection prompt (fallback for icon-heavy / sparse screens) ──────────
_VLM_REFLECTION_PROMPT = """\
You are verifying whether a desktop automation action succeeded.

ACTION PERFORMED:
  Type        : {action_type}
  Description : {description}
  Expected    : {verification}

Study the screenshot carefully. Did this action succeed?

SUCCESS CRITERIA:
  click/right_click/double_click: succeed if a new menu, window, or focused element
    appeared. FAIL if the screen is completely unchanged after the click.
  type: succeed if typed text appears in an input field or form area.
  key_press: succeed if the expected screen change occurred.
  hotkey: succeed if the expected effect occurred or no error appeared.
  scroll: succeed if content shifted at all.

Return success=false with confidence=0.5 when the outcome is uncertain.
Only return success=false with high confidence when you are CERTAIN the
action had zero visible effect.

Reply JSON only:
{{"success": true/false, "confidence": 0.0-1.0, "observation": "one sentence", \
"error_description": "", "should_retry": true/false, "recovery_hint": ""}}"""


# Actions whose success is judged visually (a menu, selection highlight, toggle
# state, icon change) rather than by text. For these, an uncertain OCR→LLM verdict
# is escalated to a pixel-level VLM check — the LLM never saw the screenshot.
_VISUAL_ACTIONS = ("click", "right_click", "double_click", "scroll", "drag")


class ReflectionAgent:
    def __init__(
        self,
        client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.75,
        ocr: OCREngine | None = None,
        escalate_uncertain: bool = True,
    ):
        self.client = client
        self.capturer = capturer
        self.min_confidence = min_confidence
        self._ocr = ocr if ocr is not None else OCREngine()
        # When True, an uncertain OCR→LLM verdict on a visual action is resolved
        # by a second look with the VLM on the actual screenshot. This is the main
        # accuracy lever for visual-only changes the text path can't see. It is
        # read via getattr() in verify() so agents built with __new__ (some tests)
        # safely default to disabled.
        self._escalate_uncertain = escalate_uncertain

    # ── public API ────────────────────────────────────────────────────────────

    def verify(self, step: ActionStep, wait_s: float = 1.5,
               pre_hash=None) -> ReflectionResult:
        """
        Verify if an action step succeeded.
        1. For click actions: use pre_hash (captured by orchestrator BEFORE the action)
           as the baseline. Falls back to capturing after-action if pre_hash is None.
        2. Adaptive wait for UI to settle.
        3. Capture after-screenshot; if hash unchanged for click → immediate failure.
        4. OCR → LLM (primary) or VLM (fallback if screen is icon-heavy).
        """
        if step.action_type == "wait":
            try:
                duration = float(step.value or "1.0")
            except ValueError:
                duration = 1.0
            time.sleep(duration)
            return ReflectionResult(
                success=True, confidence=1.0,
                observation="Wait completed.", error_description="",
                should_retry=False, recovery_hint=""
            )

        from core.capture.screenshot import frame_phash

        _CLICK_ACTIONS = ("click", "right_click", "double_click")
        # Use the pre-action hash captured by the orchestrator before execute() fires.
        # Fallback to capturing now only when no pre-hash was supplied.
        _before_hash = pre_hash
        if _before_hash is None and step.action_type in _CLICK_ACTIONS:
            _before_hash = frame_phash(self.capturer.capture())

        self._adaptive_wait(*self._wait_bounds(step))

        # Capture once — reuse for both paths
        screenshot = self.capturer.capture()

        if _before_hash is not None:
            # High-fidelity frame comparison (H2). delta==0 now genuinely means
            # "nothing changed on screen", so a click that opened a small menu is
            # no longer mis-scored as a no-op.
            if (_before_hash - frame_phash(screenshot)) == 0:
                result = ReflectionResult(
                    success=False,
                    confidence=0.98,
                    observation="Screen unchanged after click — element may not be interactive",
                    error_description="Screen unchanged after click",
                    should_retry=True,
                    recovery_hint="Re-ground the target element; it may have moved or become inactive",
                )
                logger.info(
                    f"[REFLECTION] Step {step.id} '{step.description[:40]}' → "
                    f"FAILED (delta=0, conf={result.confidence:.2f})"
                )
                logger.warning(
                    f"[REFLECTION] {result.error_description} | hint: {result.recovery_hint}"
                )
                return result
        thumb = screenshot.copy()
        thumb.thumbnail((960, 540))
        words = self._ocr.extract(thumb)
        meaningful = [w for w in words if len(w.text) >= 3 and w.conf >= 0.65]

        verification = _infer_verification(step) if step.action_type == "type" else (
            step.verification or _infer_verification(step)
        )

        # key_press and hotkey steps: always use LLM (fast), even with sparse OCR.
        # Falling back to VLM for a key-press forces a slow model swap on 6GB VRAM.
        _force_llm = step.action_type in ("key_press", "hotkey", "type", "wait")

        _used_llm = len(meaningful) >= 3 or _force_llm
        if _used_llm:
            ocr_text = ", ".join(w.text for w in meaningful[:40]) if meaningful else "(screen appears blank or icon-only)"
            result = self._verify_with_llm(step, ocr_text, verification)
            result.ocr_text = ocr_text
        else:
            ocr_text = ""
            img_b64 = self._encode(screenshot)
            result = self._verify_with_vlm(step, img_b64, verification)

        # ── Escalate uncertain visual verdicts to the VLM (pixel-level check) ──
        # The OCR→LLM path can confirm text but is blind to visual-only outcomes:
        # a selection highlight, a toggled switch, an icon state change, a menu
        # that contains no new OCR-able text. When that path is uncertain (neither
        # a confident success nor a confident failure) on a visual action, take a
        # second look with the VLM on the real screenshot and reconcile.
        if (
            getattr(self, "_escalate_uncertain", False)
            and _used_llm                                   # primary was the text path
            and step.action_type in _VISUAL_ACTIONS
            and result.confidence < self.min_confidence     # uncertain either way
            and not (result.success and result.confidence >= 0.85)
        ):
            try:
                logger.info(
                    f"[REFLECTION] Uncertain {step.action_type} verdict "
                    f"(conf={result.confidence:.2f}) — escalating to VLM screenshot check"
                )
                vlm_result = self._verify_with_vlm(
                    step, self._encode(screenshot), verification
                )
                result = self._reconcile(result, vlm_result)
                result.ocr_text = ocr_text   # preserve OCR for orchestrator reuse
            except Exception as e:
                logger.debug(
                    f"[REFLECTION] VLM escalation failed ({e}) — keeping text verdict"
                )

        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            f"[REFLECTION] Step {step.id} '{step.description[:40]}' → "
            f"{status} (conf={result.confidence:.2f})"
        )
        if not result.success:
            logger.warning(f"[REFLECTION] {result.error_description} | hint: {result.recovery_hint}")
        return result

    # ── verification paths ────────────────────────────────────────────────────

    def _verify_with_llm(
        self, step: ActionStep, ocr_text: str, verification: str
    ) -> ReflectionResult:
        prompt = _LLM_REFLECTION_PROMPT.format(
            action_type=step.action_type,
            description=step.description,
            verification=verification,
            ocr_text=ocr_text,
        )
        messages = [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.query_llm(messages, max_tokens=120, temperature=0.1)
        return self._parse(resp.content)

    def _verify_with_vlm(
        self, step: ActionStep, img_b64: str, verification: str
    ) -> ReflectionResult:
        prompt = _VLM_REFLECTION_PROMPT.format(
            action_type=step.action_type,
            description=step.description,
            verification=verification,
        )
        resp = self.client.query_vlm(
            prompt=prompt,
            image_base64=img_b64,
            max_tokens=120,
            temperature=0.1,
        )
        return self._parse(resp.content)

    def _reconcile(self, primary: ReflectionResult, vlm: ReflectionResult) -> ReflectionResult:
        """Combine an uncertain text verdict with a VLM screenshot verdict.

        The VLM actually saw the pixels, so it is authoritative WHEN it is
        confident. If the VLM is itself uncertain we keep the primary verdict
        (escalation didn't add information). This both rescues false failures
        (text path missed a visual change the VLM confirms) and catches false
        successes (text matched but the VLM sees nothing happened).
        """
        # VLM confidently resolves the outcome either way → trust it.
        if vlm.confidence >= self.min_confidence:
            vlm.observation = f"[VLM] {vlm.observation}".strip()
            return vlm
        # Both uncertain — keep whichever is more confident, preferring primary
        # on ties so behaviour stays stable.
        return vlm if vlm.confidence > primary.confidence else primary

    # ── helpers ───────────────────────────────────────────────────────────────

    def _wait_bounds(self, step: ActionStep):
        key = (step.key or "").lower()
        is_app_launch = (
            step.action_type == "key_press" and key == "enter" and
            any(w in step.description.lower() for w in ("launch", "open", "start", "run"))
        )
        is_snap_launch = (
            step.action_type == "type" and
            any(w in (step.value or "").lower() for w in ("firefox", "thunderbird"))
        )
        if is_app_launch:
            return 1.5, 3.0
        if is_snap_launch:
            return 1.0, 2.0
        if step.action_type in ("key_press", "hotkey", "type"):
            return 0.5, 1.5
        return 0.8, 2.0

    def _adaptive_wait(self, min_wait: float, max_wait: float, poll: float = 0.15) -> None:
        """Sleep at least min_wait, then exit when screen settles.

        Hard cap of min_wait + max_wait + 5s prevents infinite hang on
        animated content (video players, loading spinners, live GIFs).
        """
        time.sleep(min_wait)
        deadline = time.time() + (max_wait - min_wait) + 5.0   # hard max
        self.capturer.has_changed()      # prime the baseline hash
        while time.time() < deadline:
            time.sleep(poll)
            if not self.capturer.has_changed(threshold=0.02):
                break

    def _parse(self, text: str) -> ReflectionResult:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        data = {}
        start = text.find('{')
        if start != -1:
            depth, end = 0, -1
            for i, ch in enumerate(text[start:], start):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                try:
                    json_str = re.sub(r",\s*([\]}])", r"\1", text[start:end + 1])
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        if not data:
            # Fix C3: when the verdict JSON can't be parsed, DO NOT keyword-sniff
            # for success words. The old heuristic returned success=True for any
            # response containing "succeed"/"changed" — including negated
            # sentences like "this did NOT succeed". An unparseable verdict means
            # the outcome is UNKNOWN, which must read as uncertain (retry), never
            # as success. Only an explicit, parsed success can pass a step.
            text_lower = text.lower()
            negated = any(neg in text_lower for neg in (
                "not ", "n't", "no ", "fail", "didn't", "did not", "unchanged",
                "no change", "nothing", "error", "unable", "cannot", "can't",
            ))
            looks_positive = any(w in text_lower for w in [
                "success", "succeeded", "appeared", "opened", "launched",
                "displayed", "confirmed",
            ])
            success = looks_positive and not negated
            data = {
                "success": success,
                # Uncertain by construction — keep confidence low so the
                # orchestrator's threshold logic treats it as "retry", not "done".
                "confidence": 0.5,
                "observation": text[:120],
                "error_description": "" if success else text[:120],
                "should_retry": True,
                "recovery_hint": "",
            }

        success = bool(data.get("success", False))
        conf = float(data.get("confidence", 0.85 if success else 0.3))
        # Preserve the LLM's raw confidence — do not clamp in either direction.
        # Clamping destroyed calibration: failures were forced to ≤0.60 which,
        # combined with the old fail_threshold logic, made every step pass.

        return ReflectionResult(
            success=success,
            confidence=conf,
            observation=data.get("observation", ""),
            error_description=data.get("error_description", ""),
            should_retry=bool(data.get("should_retry", not success)),
            recovery_hint=data.get("recovery_hint", ""),
        )

    def _encode(self, image: Image.Image, quality: int = 85) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()


# ── verification hint generator ───────────────────────────────────────────────

# Hotkey combo → expected observable effect, checked in order (first substring
# match wins, so more specific combos must come before their prefixes).
_HOTKEY_HINTS = {
    "ctrl+shift+s": "Save As dialog appeared",
    "ctrl+alt+t":   "terminal window opened",
    "ctrl+s": (
        "File saved — silence is success (named file saves with no dialog). "
        "A Save-As dialog appearing is also success. Fail only if an error message appears."
    ),
    "ctrl+l":       "address bar focused and highlighted",
    "ctrl+t":       "new tab opened",
    "ctrl+w":       "current tab or window closed",
    "ctrl+f":       "find bar appeared",
    "ctrl+a":       "all content selected",
    "ctrl+c":       "content copied to clipboard",
    "ctrl+v":       "clipboard content pasted and visible",
    "ctrl+z":       "last action undone",
    "alt+f4":       "active window closed",
    "alt+tab":      "window switcher appeared or focus changed",
}


def _infer_verification(step: ActionStep) -> str:
    key = (step.key or "").lower()
    val = (step.value or "")
    desc = step.description.lower()

    if step.action_type == "wait":
        return "wait completed — always success"
    if step.action_type == "key_press":
        if key == "super":
            return "GNOME Activities overlay appeared with search bar visible"
        if key == "enter":
            if any(w in desc for w in ("launch", "open", "start", "run")):
                return "target application window opened and is visible on screen"
            return "enter key accepted — form submitted or screen changed"
        if key in ("escape", "esc"):
            return "dialog or overlay dismissed"
        if key == "tab":
            return "focus moved to next field"
        return f"pressing '{key}' caused a visible change"
    if step.action_type == "hotkey":
        for combo, hint in _HOTKEY_HINTS.items():
            if combo in key:
                return hint
        return f"hotkey '{key}' produced expected effect"
    if step.action_type == "type":
        if val:
            return (
                f"Text typed — any version of '{val[:40]}' visible in active field. "
                f"Accept autocorrect, capitalization changes, or partial match."
            )
        return "typed text visible somewhere on screen"
    if step.action_type in ("click", "double_click"):
        return f"'{(step.target or step.description)[:60]}' responded — visual change visible"
    if step.action_type == "right_click":
        return "context menu appeared"
    if step.action_type == "scroll":
        return f"content scrolled {val or 'down'}"
    return "action completed — any visible change counts as success"
