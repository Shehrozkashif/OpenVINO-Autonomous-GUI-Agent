# agents/grounding/grounding_agent.py
"""
UI Grounding Agent — converts natural language element descriptions to screen coordinates.
Uses Phi-3.5-Vision via OVMS.
"""
import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import imagehash
from loguru import logger
from PIL import Image

from core.capture.screenshot import ScreenCapture
from core.pipeline.ovms_client import OVMSClient


@dataclass
class GroundingResult:
    x: int
    y: int
    confidence: float
    found: bool
    latency_ms: float
    target: str


class ElementCache:
    """Cache UI element coordinates to avoid redundant VLM calls."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple] = {}
        self._ttl = ttl_seconds

    def get(self, target: str, screen_hash: str) -> Optional[Tuple[int, int]]:
        if target not in self._cache:
            return None
        x, y, ts, cached_hash = self._cache[target]
        if screen_hash != cached_hash or (time.time() - ts) > self._ttl:
            del self._cache[target]
            return None
        return x, y

    def put(self, target: str, x: int, y: int, screen_hash: str):
        self._cache[target] = (x, y, time.time(), screen_hash)

    def invalidate(self):
        self._cache.clear()


GROUNDING_PROMPT = """Look at this screenshot. Find the UI element: "{target}"

Output ONLY valid JSON in this exact format, no other text:
{{"x": <integer>, "y": <integer>, "confidence": <float 0.0-1.0>, "found": <true|false>}}

Rules:
- x, y must be the CENTER pixel of the element
- confidence 1.0 = certain, 0.0 = not found
- If element not visible: {{"x": 0, "y": 0, "confidence": 0.0, "found": false}}"""


class UIGroundingAgent:
    """
    The agent's ability to "see" and locate UI elements.

    Workflow per grounding request:
    1. Check element cache (skip VLM if same screen + same target)
    2. Capture screenshot
    3. Send screenshot + prompt to VLM via OVMS
    4. Parse response → GroundingResult
    5. Validate coordinates are within screen bounds
    6. Cache on success
    """

    def __init__(
        self,
        ovms_client: OVMSClient,
        capturer: ScreenCapture,
        min_confidence: float = 0.7
    ):
        self.ovms = ovms_client
        self.capturer = capturer
        self.cache = ElementCache()
        self.min_confidence = min_confidence
        import pyautogui
        self.screen_w, self.screen_h = pyautogui.size()

    def ground(self, target: str, max_retries: int = 2) -> GroundingResult:
        """
        Find a UI element by natural language description.
        target: e.g. "the Save button", "search bar at the top", "X close button"
        """
        start = time.time()

        # Capture + hash for cache lookup
        screenshot_b64 = self.capturer.capture_as_base64(quality=90)
        img_data = base64.b64decode(screenshot_b64)
        img = Image.open(io.BytesIO(img_data))
        current_hash = str(imagehash.phash(img))

        # Check cache
        cached = self.cache.get(target, current_hash)
        if cached:
            logger.info(f"[GROUNDING] Cache hit for '{target}' → {cached}")
            return GroundingResult(
                x=cached[0], y=cached[1], confidence=1.0, found=True,
                latency_ms=(time.time() - start) * 1000, target=target
            )

        prompt = GROUNDING_PROMPT.format(target=target)

        for attempt in range(max_retries + 1):
            try:
                resp = self.ovms.query_vlm(
                    prompt=prompt,
                    image_base64=screenshot_b64,
                    max_tokens=60,
                    temperature=0.1
                )
                result = self._parse_response(resp.content, target, start)

                if result.found:
                    # Clamp to screen bounds
                    result.x = max(0, min(result.x, self.screen_w - 1))
                    result.y = max(0, min(result.y, self.screen_h - 1))
                    self.cache.put(target, result.x, result.y, current_hash)

                logger.info(
                    f"[GROUNDING] '{target}' → ({result.x},{result.y}) "
                    f"conf={result.confidence:.2f} found={result.found}"
                )
                return result

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"[GROUNDING] Parse error attempt {attempt+1}: {e}")
                if attempt == max_retries:
                    return GroundingResult(
                        x=0, y=0, confidence=0.0, found=False,
                        latency_ms=(time.time() - start) * 1000, target=target
                    )

    def _parse_response(self, text: str, target: str, start: float) -> GroundingResult:
        """Extract JSON from VLM response (VLMs sometimes add surrounding text)."""
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # Fallback: try "x,y" format
            coord_match = re.search(r'(\d+)\s*,\s*(\d+)', text)
            if coord_match:
                data = {
                    "x": int(coord_match.group(1)),
                    "y": int(coord_match.group(2)),
                    "confidence": 0.75,
                    "found": True
                }
            else:
                raise ValueError(f"Cannot parse VLM response: {text[:100]}")

        return GroundingResult(
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            confidence=float(data.get("confidence", 0.0)),
            found=bool(data.get("found", False)),
            latency_ms=(time.time() - start) * 1000,
            target=target
        )

    def ground_multiple(self, targets: List[str]) -> List[GroundingResult]:
        """Ground multiple elements from a single screenshot (more efficient than N calls)."""
        start = time.time()
        screenshot_b64 = self.capturer.capture_as_base64(quality=90)

        targets_list = "\n".join(f'{i+1}. "{t}"' for i, t in enumerate(targets))
        prompt = f"""Look at this screenshot. Find ALL of these UI elements:
{targets_list}

Output ONLY a JSON array, one object per element, in order:
[{{"target":"...", "x":<int>, "y":<int>, "confidence":<float>, "found":<bool>}}, ...]"""

        try:
            resp = self.ovms.query_vlm(prompt, screenshot_b64, max_tokens=300)
            arr_match = re.search(r'\[.*?\]', resp.content, re.DOTALL)
            if not arr_match:
                raise ValueError("No JSON array in response")
            data = json.loads(arr_match.group())
            results = []
            for item in data:
                results.append(GroundingResult(
                    x=int(item.get("x", 0)),
                    y=int(item.get("y", 0)),
                    confidence=float(item.get("confidence", 0.0)),
                    found=bool(item.get("found", False)),
                    latency_ms=(time.time() - start) * 1000,
                    target=item.get("target", "")
                ))
            return results
        except Exception as e:
            logger.error(f"[GROUNDING] ground_multiple failed: {e} — falling back to individual calls")
            return [self.ground(t) for t in targets]
