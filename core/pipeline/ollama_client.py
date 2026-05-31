# core/pipeline/ollama_client.py
"""
Ollama inference client.

LLM (text reasoning):   qwen3:14b via Ollama       — planning, routing, reflection
VLM (visual grounding): UI-TARS-1.5-7B via vLLM    — primary (port 8000, if running)
                        qwen2.5vl-gui via Ollama    — auto-fallback when vLLM absent

Setup:
    ollama pull qwen3:14b
    ollama pull qwen2.5vl:7b
    printf 'FROM qwen2.5vl:7b\nPARAMETER num_ctx 4096\n' | ollama create qwen2.5vl-gui -f -
    ollama serve
"""
import time
from typing import List

import httpx
from loguru import logger
from pydantic import BaseModel


class OVMSResponse(BaseModel):
    """Unified response type for all inference calls."""
    content: str
    model: str
    latency_ms: float
    tokens_generated: int

_DEFAULT_LLM = "qwen3:14b"
_DEFAULT_VLM_VLLM = "ByteDance-Seed/UI-TARS-1.5-7B"   # vLLM primary
_DEFAULT_VLM_OLLAMA = "qwen2.5vl-gui"                  # Ollama fallback (4096-ctx, GPU-friendly)

_DEFAULT_LLM_BASE_URL = "http://localhost:11434"   # Ollama
_DEFAULT_VLM_BASE_URL = "http://localhost:8000"    # vLLM


def _pick_ollama_vlm(ollama_base_url: str) -> str:
    """Return the best available vision model from Ollama.

    Prefers qwen2.5vl-gui (4096-ctx build) so the VLM loads on GPU.
    Falls back to full qwen2.5vl:7b or llava variants if not available.
    """
    # Base prefixes in priority order — matched with or without tag suffix
    preferred_bases = ["qwen2.5vl-gui", "qwen2.5vl", "llava"]
    try:
        r = httpx.get(f"{ollama_base_url}/api/tags", timeout=3.0)
        if r.status_code == 200:
            pulled = [m["name"] for m in r.json().get("models", [])]
            for base in preferred_bases:
                match = next((m for m in pulled if m == base or m.startswith(base + ":") or m.startswith(base + "-")), None)
                if match:
                    return match
    except Exception:
        pass
    return _DEFAULT_VLM_OLLAMA


class OllamaClient:
    """
    Dual-backend client.
    - query_llm() → Ollama (qwen3:14b) for planning, routing, reflection
    - query_vlm() → vLLM (UI-TARS) if running, else Ollama (qwen2.5vl:7b)
    Both backends use the OpenAI-compatible /v1/chat/completions endpoint.
    """

    def __init__(
        self,
        vlm_model: str = None,
        llm_model: str = _DEFAULT_LLM,
        llm_base_url: str = _DEFAULT_LLM_BASE_URL,
        vlm_base_url: str = _DEFAULT_VLM_BASE_URL,
        # legacy single-URL param kept for backward compat
        base_url: str = None,
        timeout: float = 600.0,
    ):
        self.llm_model = llm_model
        self.llm_base_url = base_url or llm_base_url
        self.client = httpx.Client(timeout=timeout)

        # Auto-detect VLM backend: prefer vLLM (UI-TARS), fall back to Ollama.
        # Always probe the dedicated vLLM port — do NOT inherit the legacy base_url
        # here, because Ollama also exposes /v1/models and would be falsely detected.
        _vlm_url = vlm_base_url
        _vllm_available = False
        try:
            r = self.client.get(f"{_vlm_url}/v1/models", timeout=2.0)
            if r.status_code == 200:
                _vllm_available = True
        except Exception:
            pass

        if _vllm_available:
            self.vlm_base_url = _vlm_url
            self.vlm_model = vlm_model or _DEFAULT_VLM_VLLM
            logger.info(f"[OllamaClient] VLM backend: vLLM ({self.vlm_model}) at {self.vlm_base_url}")
        else:
            # Fall back to Ollama for VLM — same endpoint, different model
            self.vlm_base_url = self.llm_base_url
            self.vlm_model = vlm_model or _pick_ollama_vlm(self.llm_base_url)
            logger.info(
                f"[OllamaClient] vLLM not detected — VLM via Ollama: {self.vlm_model} "
                f"at {self.vlm_base_url}"
            )

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
        response_schema: dict = None,
    ) -> OVMSResponse:
        """Send text messages to the LLM via Ollama's native /api/chat.

        Uses Ollama native API (not OpenAI-compat) so we can pass:
          think=False  — prevents qwen3 from filling content with chain-of-thought
          format=schema — structured output via JSON Schema (guarantees correct structure)

        response_schema: optional JSON Schema dict; if provided enforces output structure.
        """
        start = time.time()
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if response_schema is not None:
            payload["format"] = response_schema

        resp = self.client.post(f"{self.llm_base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content") or ""
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM] → {content[:100]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content,
            model=self.llm_model,
            latency_ms=latency_ms,
            tokens_generated=data.get("eval_count", 0),
        )

    def check_health(self) -> dict:
        results = {}
        _vlm_via_ollama = self.vlm_base_url == self.llm_base_url

        # Check Ollama (always used for LLM; also VLM when vLLM is absent)
        try:
            r = self.client.get(f"{self.llm_base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                pulled = [m["name"] for m in r.json().get("models", [])]

                # LLM check
                llm_base = self.llm_model.split(":")[0]
                llm_ok = any(llm_base in m for m in pulled)
                results[f"LLM ({self.llm_model})"] = (
                    "OK" if llm_ok else f"NOT PULLED — run: ollama pull {self.llm_model}"
                )

                # VLM via Ollama check
                if _vlm_via_ollama:
                    vlm_base = self.vlm_model.split(":")[0]
                    vlm_ok = any(vlm_base in m for m in pulled)
                    results[f"VLM ({self.vlm_model}) via Ollama"] = (
                        "OK" if vlm_ok else f"NOT PULLED — run: ollama pull {self.vlm_model}"
                    )
            else:
                results["Ollama"] = f"HTTP {r.status_code}"
        except Exception as e:
            results["Ollama"] = f"UNREACHABLE — is 'ollama serve' running? ({e})"

        # Check vLLM only when it's the active VLM backend
        if not _vlm_via_ollama:
            try:
                r = self.client.get(f"{self.vlm_base_url}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    models = [m["id"] for m in r.json().get("data", [])]
                    ok = any(self.vlm_model in m for m in models)
                    results[f"VLM ({self.vlm_model}) via vLLM"] = (
                        "OK" if ok else f"NOT LOADED — run: vllm serve {self.vlm_model} --port 8000"
                    )
                else:
                    results["vLLM"] = f"HTTP {r.status_code}"
            except Exception as e:
                results["vLLM"] = f"UNREACHABLE — is 'vllm serve' running? ({e})"

        return results
