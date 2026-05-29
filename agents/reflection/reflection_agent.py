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


REFLECTION_PROMPT_TEMPLATE = """An automated agent just performed this desktop action:
Action: {description}
Expected result: {verification}

Look at this screenshot (taken AFTER the action).
Did the action succeed? Output ONLY JSON:
{{
  "success": true|false,
  "confidence": <float 0.0-1.0>,
  "observation": "<what you see that confirms or denies success>",
  "error_description": "<what went wrong, or empty string if success>",
  "should_retry": true|false,
  "recovery_hint": "<suggestion for recovery if failed, or empty string>"
}}"""


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
        wait_s: float = 0.5
    ) -> ReflectionResult:
        """
        Verify if an action step succeeded.

        1. Wait for UI animations to settle
        2. Capture after screenshot
        3. Ask VLM: "Given the action taken, does the after screenshot show success?"
        4. Parse and return result
        """
        time.sleep(wait_s)
        after_b64 = self.capturer.capture_as_base64(quality=85)

        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            description=step.description,
            verification=step.verification or "action completed without error"
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
        
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        data = {}
        if json_match:
            try:
                # Fix trailing commas
                json_str = re.sub(r",\s*([\]}])", r"\1", json_match.group())
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
