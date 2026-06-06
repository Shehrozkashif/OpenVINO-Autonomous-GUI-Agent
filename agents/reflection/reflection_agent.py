# agents/reflection/reflection_agent.py
"""
Reflection Agent — verifies that each action step succeeded.

Primary path: OCR the after-screenshot, pass visible text to the LLM (qwen3:14b).
  - qwen3 is a reasoning model and handles conditional success logic accurately.
  - No image encoding needed — fast, cheap, high quality.

Fallback path: VLM (qwen2.5vl / UI-TARS) is used only when OCR returns fewer
than 3 meaningful words (icon-heavy screens, blank desktops).
"""
import base64
import io
import json
import re
import time
from dataclasses import dataclass

import numpy as np
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
# Sent to qwen3:14b with OCR text. Much more reliable than asking a grounding
# VLM to reason about conditional success criteria.

_LLM_SYSTEM = "You are a desktop automation verifier. Reply with valid JSON only."

_LLM_REFLECTION_PROMPT = """\
Verify whether a desktop automation step succeeded.

STEP     : {action_type} — {description}
EXPECTED : {verification}
SCREEN TEXT (OCR after action): {ocr_text}

Rules — be lenient, only fail when clearly nothing happened:
  click/double_click : succeed if any new label appeared or existing one changed
  type               : succeed if any version of the typed text is in screen text
  key_press "super"  : succeed if "Activities", "Type to search", or "Search" visible
  key_press "enter"  : succeed if expected result is visible anywhere on screen
  hotkey ctrl+l      : succeed if address bar or URL text visible
  hotkey ctrl+s      : succeed if no error dialog; title change optional
  scroll             : succeed unless content is provably unchanged
  Any ambiguous case → success=true, lower confidence

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
  click/right_click/double_click: succeed if element shows focus, a menu/window opened,
    or any visual change occurred. FAIL only if screen is completely unchanged.
  type: succeed if ANY text appears in any field, even with autocorrect changes.
  key_press: succeed if ANY screen change occurred (window, menu, cursor, app launch).
  hotkey: succeed if expected effect occurred OR no error dialog appeared.
  scroll: succeed if content shifted at all.

When confidence is between 0.5 and 0.8, return success=true.
Only return FAILED when you are CERTAIN the action had zero effect.

Reply JSON only:
{{"success": true/false, "confidence": 0.0-1.0, "observation": "one sentence", \
"error_description": "", "should_retry": true/false, "recovery_hint": ""}}"""


class ReflectionAgent:
    def __init__(
        self,
        ovms_client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.75,
    ):
        self.ovms = ovms_client
        self.capturer = capturer
        self.min_confidence = min_confidence
        self._ocr = OCREngine()

    # ── public API ────────────────────────────────────────────────────────────

    def verify(self, step: ActionStep, wait_s: float = 1.5) -> ReflectionResult:
        """
        Verify if an action step succeeded.
        1. Adaptive wait for UI to settle.
        2. Capture screenshot.
        3. OCR → LLM (primary) or VLM (fallback if screen is icon-heavy).
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

        self._adaptive_wait(*self._wait_bounds(step))

        # Capture once — reuse for both paths
        screenshot = self.capturer.capture()
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

        if len(meaningful) >= 3 or _force_llm:
            ocr_text = ", ".join(w.text for w in meaningful[:40]) if meaningful else "(screen appears blank or icon-only)"
            result = self._verify_with_llm(step, ocr_text, verification)
            result.ocr_text = ocr_text
        else:
            img_b64 = self._encode(screenshot)
            result = self._verify_with_vlm(step, img_b64, verification)

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
        resp = self.ovms.query_llm(messages, max_tokens=120, temperature=0.1)
        return self._parse(resp.content)

    def _verify_with_vlm(
        self, step: ActionStep, img_b64: str, verification: str
    ) -> ReflectionResult:
        prompt = _VLM_REFLECTION_PROMPT.format(
            action_type=step.action_type,
            description=step.description,
            verification=verification,
        )
        resp = self.ovms.query_vlm(
            prompt=prompt,
            image_base64=img_b64,
            max_tokens=120,
            temperature=0.1,
        )
        return self._parse(resp.content)

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
            text_lower = text.lower()
            success = any(w in text_lower for w in [
                "success", "succeeded", "yes", "correct", "appeared", "visible",
                "typed", "entered", "opened", "clicked", "focused", "launched",
                "showing", "displayed", "changed",
            ])
            data = {
                "success": success,
                "confidence": 0.85 if success else 0.5,
                "observation": text[:120],
                "error_description": "" if success else text[:120],
                "should_retry": not success,
                "recovery_hint": "",
            }

        success = bool(data.get("success", False))
        conf = float(data.get("confidence", 0.85 if success else 0.3))
        # Keep success/confidence consistent: clamp each direction
        if success and conf < 0.75:
            conf = 0.75
        elif not success and conf > 0.6:
            conf = 0.6

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
        if "ctrl+s" in key:       return "file saved — title bar updated"
        if "ctrl+shift+s" in key: return "Save As dialog appeared"
        if "ctrl+l" in key:       return "address bar focused and highlighted"
        if "ctrl+t" in key:       return "new tab opened"
        if "ctrl+w" in key:       return "current tab or window closed"
        if "ctrl+alt+t" in key:   return "terminal window opened"
        if "ctrl+f" in key:       return "find bar appeared"
        if "ctrl+a" in key:       return "all content selected"
        if "ctrl+c" in key:       return "content copied to clipboard"
        if "ctrl+v" in key:       return "clipboard content pasted and visible"
        if "ctrl+z" in key:       return "last action undone"
        if "alt+f4" in key:       return "active window closed"
        if "alt+tab" in key:      return "window switcher appeared or focus changed"
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
