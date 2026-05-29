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


REFLECTION_PROMPT_TEMPLATE = """An automated desktop agent just performed this action:

  Action type : {action_type}
  Description : {description}
  What to verify: {verification}

Look carefully at the screenshot taken RIGHT AFTER the action.
Answer: did the action succeed based on what you see?

Output ONLY valid JSON — no prose:
{{
  "success": true or false,
  "confidence": <float 0.0–1.0>,
  "observation": "<one sentence: exactly what you see that supports your answer>",
  "error_description": "<what is wrong if success=false, else empty string>",
  "should_retry": true or false,
  "recovery_hint": "<one concrete suggestion if failed, else empty string>"
}}"""


def _infer_verification(step) -> str:
    """Generate a specific visual check when step.verification is empty."""
    key = (step.key or "").lower()
    val = (step.value or "")
    desc = step.description.lower()

    if step.action_type == "key_press":
        if key == "super":
            return "GNOME Activities search overlay appeared (search bar is visible at top)"
        if key == "enter":
            if any(w in desc for w in ("launch", "open", "start", "run")):
                return "the target application window opened and is visible on screen"
            return "the Enter key was accepted (form submitted, selection confirmed, or next screen appeared)"
        if key in ("escape", "esc"):
            return "the dialog, popup, or overlay was dismissed and is no longer visible"
        if key.startswith("f") and key[1:].isdigit():
            return f"the {key.upper()} key action took effect on screen"
        return f"pressing '{key}' had the expected visual effect on screen"

    if step.action_type == "hotkey":
        if any(p in key for p in ("print_screen", "print", "prtsc")):
            return ("the print screen key was sent — success means either: a screenshot tool "
                    "dialog appeared, or a notification appeared, or the screen briefly flashed. "
                    "If no ERROR dialog appeared, treat it as success.")
        if "ctrl+s" in key:
            return "the file was saved (title bar no longer shows an unsaved indicator)"
        if "ctrl+alt+t" in key or "ctrl+t" in key:
            return "a terminal or new tab window opened and is visible"
        if "ctrl+z" in key:
            return "the previous action was undone (visible change on screen)"
        if "ctrl+c" in key or "ctrl+v" in key:
            return "the clipboard operation completed"
        if "alt+f4" in key:
            return "the active window closed"
        return f"the keyboard shortcut '{key}' produced the expected visual change"

    if step.action_type == "type":
        if val:
            preview = val[:30]
            return (
                f"After typing '{preview}': the text appears in the active field, OR "
                f"the interface responded (search results appeared, suggestions showed up, "
                f"or the display changed to reflect the input). Any visible response counts as success."
            )
        return "the typed text or a response to the input is visible on screen"

    if step.action_type in ("click", "double_click"):
        target = step.target or step.description
        return f"'{target[:50]}' responded to the click — a visual change, new window, or focus change is visible"

    if step.action_type == "right_click":
        return "a context menu appeared on screen after the right-click"

    if step.action_type == "scroll":
        direction = val or "down"
        return f"the content scrolled {direction} — different items are visible than before"

    return "the action completed and a visible change occurred on screen"


class ReflectionAgent:
    def __init__(
        self,
        ovms_client: InferenceClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.8,
    ):
        self.ovms = ovms_client
        self.capturer = capturer
        self.min_confidence = min_confidence

    def verify(
        self,
        step: ActionStep,
        wait_s: float = 1.5
    ) -> ReflectionResult:
        """
        Verify if an action step succeeded.

        1. Wait for UI animations to settle
        2. Capture after screenshot
        3. Ask VLM: "Given the action taken, does the after screenshot show success?"
        4. Parse and return result
        """
        # Key/type actions: give UI time to render.
        # Enter key to launch apps needs longer — most apps take 2-3s to open.
        key = (step.key or "").lower()
        is_app_launch = step.action_type == "key_press" and key == "enter" and \
            any(w in step.description.lower() for w in ("launch", "open", "start", "run"))
        if is_app_launch:
            actual_wait = max(wait_s, 2.5)
        elif step.action_type in ("key_press", "hotkey", "type"):
            actual_wait = max(wait_s, 1.5)
        else:
            actual_wait = wait_s
        time.sleep(actual_wait)
        after_b64 = self.capturer.capture_as_base64(quality=85)

        verification = step.verification if step.verification else _infer_verification(step)
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            action_type=step.action_type,
            description=step.description,
            verification=verification,
        )

        resp = self.ovms.query_vlm(
            prompt=prompt,
            image_base64=after_b64,
            max_tokens=200,
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
        # Remove reasoning block if </think> is present
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
        # Extract outermost JSON object — handles nested braces (e.g. nested dicts).
        # The [^{}]* pattern fails on nesting; instead find the first '{' and scan
        # forward counting braces to find the matching '}'.
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
            success = any(w in text_lower for w in ["success", "succeeded", "yes", "correct", "appeared"])
            data = {
                "success": success, 
                "confidence": 0.9 if success else 0.5,
                "observation": text[:100],
                "error_description": "" if success else text[:100],
                "should_retry": not success,
                "recovery_hint": ""
            }

        success = bool(data.get("success", False))
        
        # If success is true but no confidence provided, assume it's confident
        conf = float(data.get("confidence", 0.9 if success else 0.0))
        
        return ReflectionResult(
            success=success,
            confidence=conf,
            observation=data.get("observation", ""),
            error_description=data.get("error_description", ""),
            should_retry=bool(data.get("should_retry", not success)),
            recovery_hint=data.get("recovery_hint", "")
        )
