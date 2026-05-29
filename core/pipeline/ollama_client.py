# core/pipeline/ollama_client.py
"""
Dual-backend setup for AMD GPU server (48GB VRAM total).

LLM (text reasoning):   llama3.1:8b via Ollama  — ~4.7GB VRAM, fast planning/routing
VLM (visual grounding): UI-TARS-1.5-7B via vLLM — ~14GB VRAM, 94.2% ScreenSpot-V2

Why two backends:
  UI-TARS-1.5-7B is purpose-built for GUI grounding (beats OpenAI CUA on benchmarks)
  but is not available on Ollama. vLLM exposes the same OpenAI-compatible /v1 API.

Setup:
    # LLM — Ollama (port 11434)
    ollama pull llama3.1:8b
    ollama serve

    # VLM — vLLM (port 8000)
    pip install vllm
    vllm serve ByteDance-Seed/UI-TARS-1.5-7B --port 8000
"""
import time
from typing import List

import httpx
from loguru import logger

from core.pipeline.ovms_client import OVMSResponse

_DEFAULT_LLM = "qwen3:14b"                          # best JSON reasoning, 9GB VRAM
_DEFAULT_VLM = "ByteDance-Seed/UI-TARS-1.5-7B"     # 94.2% GUI grounding, 14GB VRAM

_DEFAULT_LLM_BASE_URL = "http://localhost:11434"   # Ollama
_DEFAULT_VLM_BASE_URL = "http://localhost:8000"    # vLLM


class OllamaClient:
    """
    Dual-backend client.
    - query_llm() → Ollama (llama3.1:8b) for planning, routing, reflection
    - query_vlm() → vLLM  (UI-TARS-1.5-7B) for visual grounding
    Both use the OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        vlm_model: str = _DEFAULT_VLM,
        llm_model: str = _DEFAULT_LLM,
        llm_base_url: str = _DEFAULT_LLM_BASE_URL,
        vlm_base_url: str = _DEFAULT_VLM_BASE_URL,
        # legacy single-URL param kept for backward compat
        base_url: str = None,
        timeout: float = 600.0,
    ):
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.llm_base_url = base_url or llm_base_url
        self.vlm_base_url = base_url or vlm_base_url
        self.client = httpx.Client(timeout=timeout)

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        system_prompt: str = None,
    ) -> OVMSResponse:
        """Send screenshot + prompt to UI-TARS-1.5-7B (vLLM) for visual grounding."""
        start = time.time()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        })
        payload = {
            "model": self.vlm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.client.post(f"{self.vlm_base_url}/v1/chat/completions", json=payload)
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
        """Send text messages to qwen3:14b (Ollama) for planning and reasoning."""
        start = time.time()
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = self.client.post(f"{self.llm_base_url}/v1/chat/completions", json=payload)
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
        # Check Ollama (LLM)
        try:
            r = self.client.get(f"{self.llm_base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                pulled = [m["name"] for m in r.json().get("models", [])]
                base = self.llm_model.split(":")[0]
                ok = any(base in m for m in pulled)
                results[f"LLM ({self.llm_model})"] = (
                    "OK" if ok else f"NOT PULLED — run: ollama pull {self.llm_model}"
                )
            else:
                results["Ollama"] = f"HTTP {r.status_code}"
        except Exception as e:
            results["Ollama (LLM)"] = f"UNREACHABLE — is 'ollama serve' running? ({e})"

        # Check vLLM (VLM)
        try:
            r = self.client.get(f"{self.vlm_base_url}/v1/models", timeout=5.0)
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                ok = any(self.vlm_model in m for m in models)
                results[f"VLM ({self.vlm_model})"] = (
                    "OK" if ok else f"NOT LOADED — run: vllm serve {self.vlm_model} --port 8000"
                )
            else:
                results["vLLM (VLM)"] = f"HTTP {r.status_code}"
        except Exception as e:
            results["vLLM (VLM)"] = f"UNREACHABLE — is 'vllm serve' running? ({e})"

        return results
