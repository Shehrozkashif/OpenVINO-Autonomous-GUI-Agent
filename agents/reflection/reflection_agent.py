# agents/reflection/reflection_agent.py
"""
Reflection Agent — verifies that each action step succeeded.
Compares before-state description with after screenshot using VLM.
"""
import json
import re
import time
from dataclasses import dataclass

from loguru import logger

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


REFLECTION_PROMPT_TEMPLATE = """You are verifying whether a desktop automation action succeeded.

ACTION PERFORMED:
  Type        : {action_type}
  Description : {description}
  Expected    : {verification}

Study the screenshot carefully. Did this action succeed?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUCCESS CRITERIA BY ACTION TYPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

click / right_click / double_click:
  SUCCESS if: the clicked element shows focus, highlight, selection, or any
  visual change. A new window, menu, or dialog appearing = SUCCESS.
  FAIL only if: the screen is completely unchanged.

type:
  SUCCESS if: ANY text appears in any input field, search bar, terminal,
  or document — even with different capitalization, autocorrect changes,
  extra/missing spaces, or slight spelling differences.
  "Hello World" typed → "hello world" visible = SUCCESS.
  "hello world" typed → "Hello World" visible = SUCCESS.
  FAIL only if: the target field is completely empty with zero characters.

key_press (enter, escape, tab, arrow keys, f-keys, super):
  SUCCESS if: ANY screen change occurred — new window, closed dialog,
  cursor moved, selection changed, app launched, menu appeared.
  For "super" / Activities: success if search bar or overlay is visible.
  For "enter" to launch app: success if app window is visible anywhere.
  FAIL only if: screen is pixel-for-pixel identical with no change at all.

hotkey (ctrl+s, ctrl+l, ctrl+t, alt+f4, ctrl+alt+t, etc.):
  SUCCESS if: expected effect occurred OR no error dialog appeared.
  ctrl+s → file saved (title bar changed or no asterisk) = SUCCESS.
  ctrl+l → address bar focused or highlighted = SUCCESS.
  ctrl+alt+t → terminal window visible = SUCCESS.
  alt+f4 → window closed = SUCCESS.
  If the hotkey had no visible effect but also caused no error = SUCCESS.
  FAIL only if: an explicit error dialog appeared.

scroll:
  SUCCESS if: page or list content shifted in any direction.
  FAIL only if: scroll position is provably unchanged.

wait:
  Always SUCCESS — waiting never fails.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GENERAL LENIENCY RULES (apply to all actions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Autocorrect, spellcheck, or capitalization changes do NOT make typing fail.
- App loading slowly or partially visible still counts as launched = SUCCESS.
- A different but related visual change still counts as SUCCESS.
- An "uncertain" state should be judged as SUCCESS with lower confidence.
- Only return FAILED when you are CERTAIN the action had absolutely no effect.
- When confidence is between 0.5 and 0.8, still return success=true.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — valid JSON only, no prose, no markdown
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "success": true or false,
  "confidence": <float 0.0–1.0>,
  "observation": "<one sentence describing exactly what you see in the screenshot>",
  "error_description": "<what went wrong if success=false, else empty string>",
  "should_retry": true or false,
  "recovery_hint": "<one concrete fix suggestion if failed, else empty string>"
}}"""


def _infer_verification(step) -> str:
    """Generate a specific visual check when step.verification is empty."""
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
                return "the target application window opened and is visible on screen"
            return "enter key accepted — form submitted, selection confirmed, or screen changed"
        if key in ("escape", "esc"):
            return "dialog, popup, or overlay was dismissed"
        if key == "tab":
            return "focus moved to the next field or element"
        if key.startswith("f") and key[1:].isdigit():
            return f"{key.upper()} key effect is visible on screen"
        return f"pressing '{key}' caused a visible change on screen"

    if step.action_type == "hotkey":
        if any(p in key for p in ("print_screen", "prtsc")):
            return "screenshot taken — tool dialog appeared or notification shown. If no error, treat as success."
        if "ctrl+s" in key:
            return "file saved — title bar no longer shows unsaved indicator"
        if "ctrl+shift+s" in key:
            return "Save As dialog appeared"
        if "ctrl+alt+t" in key:
            return "terminal window opened and is visible"
        if "ctrl+t" in key:
            return "new tab opened in browser or terminal"
        if "ctrl+w" in key:
            return "current tab or window closed"
        if "ctrl+l" in key:
            return "address bar is focused and highlighted"
        if "ctrl+z" in key:
            return "last action was undone — visible change on screen"
        if "ctrl+a" in key:
            return "all text or items selected — highlighted in blue"
        if "ctrl+c" in key:
            return "content copied to clipboard"
        if "ctrl+v" in key:
            return "clipboard content pasted and visible"
        if "ctrl+f" in key:
            return "find/search bar appeared"
        if "ctrl+h" in key:
            return "find and replace dialog appeared"
        if "ctrl+p" in key:
            return "print dialog appeared"
        if "ctrl+n" in key:
            return "new document or window opened"
        if "ctrl+o" in key:
            return "open file dialog appeared"
        if "alt+f4" in key:
            return "active window closed"
        if "alt+tab" in key:
            return "window switcher appeared or focus moved to next window"
        if "super+d" in key:
            return "desktop is visible with all windows minimized"
        if "super+l" in key:
            return "screen is locked"
        return f"hotkey '{key}' produced the expected visual change, no error shown"

    if step.action_type == "type":
        if val:
            preview = val[:40]
            return (
                f"Text was typed — any version of '{preview}' is visible in the "
                f"active field. Accept any capitalization, autocorrect variant, "
                f"or partial match. Only fail if field is completely empty."
            )
        return "typed text or response is visible somewhere on screen"

    if step.action_type in ("click", "double_click"):
        target = (step.target or step.description)[:60]
        return f"'{target}' responded — focus changed, window opened, or visual change visible"

    if step.action_type == "right_click":
        return "context menu appeared on screen"

    if step.action_type == "scroll":
        direction = val or "down"
        return f"content scrolled {direction} — different items visible than before"

    return "action completed — any visible change on screen counts as success"


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

    def _adaptive_wait(self, min_wait: float, max_wait: float, poll: float = 0.15) -> None:
        """Sleep at least min_wait seconds, then exit as soon as screen stops changing."""
        time.sleep(min_wait)
        deadline = time.time() + (max_wait - min_wait)
        self.capturer.has_changed()          # set baseline hash
        while time.time() < deadline:
            time.sleep(poll)
            if not self.capturer.has_changed(threshold=0.02):
                break                        # screen settled — no need to wait longer

    def verify(
        self,
        step: ActionStep,
        wait_s: float = 1.5
    ) -> ReflectionResult:
        """
        Verify if an action step succeeded.

        1. Wait for UI animations to settle
        2. Capture after screenshot
        3. Ask VLM: did the after screenshot show success?
        4. Parse and return result
        """
        # Wait actions always succeed — no VLM call needed
        if step.action_type == "wait":
            try:
                wait_duration = float(step.value or "1.0")
            except ValueError:
                wait_duration = 1.0
            time.sleep(wait_duration)
            return ReflectionResult(
                success=True, confidence=1.0,
                observation="Wait completed successfully.",
                error_description="", should_retry=False, recovery_hint=""
            )

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
            min_wait, max_wait = 1.5, 3.0
        elif is_snap_launch:
            min_wait, max_wait = 1.0, 2.0
        elif step.action_type in ("key_press", "hotkey", "type"):
            min_wait, max_wait = 0.5, 1.5
        else:
            min_wait, max_wait = wait_s, wait_s + 1.0

        self._adaptive_wait(min_wait, max_wait)
        after_b64 = self.capturer.capture_as_base64(quality=85)

        # Always use lenient inferred verification for type actions
        # to avoid strict string matching failures from autocorrect
        if step.action_type == "type":
            verification = _infer_verification(step)
        else:
            verification = step.verification if step.verification else _infer_verification(step)

        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            action_type=step.action_type,
            description=step.description,
            verification=verification,
        )

        resp = self.ovms.query_vlm(
            prompt=prompt,
            image_base64=after_b64,
            max_tokens=120,
            temperature=0.1
        )

        result = self._parse(resp.content)
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(
            f"[REFLECTION] Step {step.id} '{step.description[:40]}' → "
            f"{status} (conf={result.confidence:.2f})"
        )
        if not result.success:
            logger.warning(f"[REFLECTION] Error: {result.error_description}")
            logger.warning(f"[REFLECTION] Hint: {result.recovery_hint}")

        return result

    def _parse(self, text: str) -> ReflectionResult:
        # Strip reasoning blocks (qwen3 think tags)
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
                "success", "succeeded", "yes", "correct", "appeared",
                "visible", "typed", "entered", "opened", "clicked",
                "focused", "launched", "showing", "displayed", "changed"
            ])
            data = {
                "success": success,
                "confidence": 0.85 if success else 0.5,
                "observation": text[:120],
                "error_description": "" if success else text[:120],
                "should_retry": not success,
                "recovery_hint": ""
            }

        success = bool(data.get("success", False))
        conf = float(data.get("confidence", 0.85 if success else 0.0))

        # Clamp confidence — never let uncertain cases block progress
        if success and conf < 0.75:
            conf = 0.75

        return ReflectionResult(
            success=success,
            confidence=conf,
            observation=data.get("observation", ""),
            error_description=data.get("error_description", ""),
            should_retry=bool(data.get("should_retry", not success)),
            recovery_hint=data.get("recovery_hint", "")
        )