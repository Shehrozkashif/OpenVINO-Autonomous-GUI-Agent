# core/pipeline/ollama_client.py
"""
Ollama backend — dual-model setup for best performance on current hardware.

LLM (text reasoning): llama3.1:8b  — runs on GPU (~4.7GB VRAM), fast planning
VLM (vision/grounding): qwen2.5vl:3b — runs on CPU (visual encoder needs 7.3GB CUDA)

Ollama automatically swaps between them. They are never loaded simultaneously,
so total VRAM never exceeds 4.7GB.

Setup:
    ollama pull llama3.1:8b    # for router / planner / reflector
    ollama pull qwen2.5vl:3b   # for visual grounding only
"""
import time
from typing import List

import httpx
from loguru import logger

from core.pipeline.ovms_client import OVMSResponse

# Default models — tuned for RTX 2060 (6GB VRAM) + AMD CPU
_DEFAULT_LLM = "llama3.1:8b"    # GPU: fast text reasoning, JSON output
_DEFAULT_VLM = "qwen2.5vl:3b"   # CPU: visual grounding (needs large CUDA buffer)


class OllamaClient:
    """
    Dual-model Ollama backend.
    - query_llm() uses the LLM (llama3.1:8b) — text planning, routing, reflection
    - query_vlm() uses the VLM (qwen2.5vl:3b) — vision grounding only
    """

    def __init__(
        self,
        vlm_model: str = _DEFAULT_VLM,
        llm_model: str = _DEFAULT_LLM,
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,
    ):
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.base_url = base_url
        self.client = httpx.Client(timeout=timeout)

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> OVMSResponse:
        """Send screenshot + prompt to qwen2.5vl for visual grounding."""
        start = time.time()
        payload = {
            "model": self.vlm_model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[VLM] {prompt[:60]}… → {content[:100]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content,
            model=self.vlm_model,
            latency_ms=latency_ms,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0),
        )

    def query_llm(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> OVMSResponse:
        """Send text messages to llama3.1:8b for planning and reasoning."""
        start = time.time()
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM] → {content[:100]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content,
            model=self.llm_model,
            latency_ms=latency_ms,
            tokens_generated=data.get("usage", {}).get("completion_tokens", 0),
        )

    def check_health(self) -> dict:
        results = {}
        try:
            r = self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            if r.status_code != 200:
                results["Ollama"] = f"HTTP {r.status_code}"
                return results

            pulled = [m["name"] for m in r.json().get("models", [])]

            def _present(name: str) -> bool:
                base = name.split(":")[0]
                return any(base in m for m in pulled)

            results[f"LLM ({self.llm_model})"] = (
                "OK" if _present(self.llm_model)
                else f"NOT PULLED — run: ollama pull {self.llm_model}"
            )
            results[f"VLM ({self.vlm_model})"] = (
                "OK" if _present(self.vlm_model)
                else f"NOT PULLED — run: ollama pull {self.vlm_model}"
            )
        except Exception as e:
            results["Ollama"] = f"UNREACHABLE — is 'ollama serve' running? ({e})"
        return results
