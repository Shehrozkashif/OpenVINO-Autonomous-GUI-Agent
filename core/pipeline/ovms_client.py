# core/pipeline/ovms_client.py
"""
OpenVINO Model Server client.
OVMS exposes OpenAI-compatible /v1/chat/completions endpoints.
All agents query models through this class.
"""
import time
from typing import List

import httpx
from loguru import logger
from pydantic import BaseModel


class OVMSResponse(BaseModel):
    content: str
    model: str
    latency_ms: float
    tokens_generated: int


class OVMSClient:
    def __init__(
        self,
        vlm_url: str = "http://localhost:8001",
        llm_url: str = "http://localhost:8002",
        timeout: float = 60.0
    ):
        self.vlm_url = vlm_url
        self.llm_url = llm_url
        self.client = httpx.Client(timeout=timeout)

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 60,
        temperature: float = 0.1
    ) -> OVMSResponse:
        """Send screenshot + question to Phi-3.5-Vision. Used by Grounding + Reflection."""
        start = time.time()
        payload = {
            "model": "phi3_vision",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = self.client.post(f"{self.vlm_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[VLM] {prompt[:50]}… → {content[:80]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content, model="phi3_vision",
            latency_ms=latency_ms,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0)
        )

    def query_llm(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> OVMSResponse:
        """Send messages to DeepSeek-R1-Qwen-7B. Used by Router, Planner, Reflection."""
        start = time.time()
        payload = {
            "model": "deepseek_llm",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = self.client.post(f"{self.llm_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM] → {content[:100]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content, model="deepseek_llm",
            latency_ms=latency_ms,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0)
        )

    def check_health(self) -> dict:
        """Verify both model servers are reachable before starting."""
        results = {}
        for name, url in [("VLM (phi3)", self.vlm_url), ("LLM (deepseek)", self.llm_url)]:
            try:
                r = self.client.get(f"{url}/v1/config", timeout=5.0)
                results[name] = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                results[name] = f"UNREACHABLE: {e}"
        return results
