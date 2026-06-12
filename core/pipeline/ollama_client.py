# core/pipeline/ollama_client.py
"""
Ollama inference client.

LLM (text reasoning):   qwen3:8b via Ollama        — planning, routing, reflection
VLM (visual grounding): UI-TARS-1.5-7B via vLLM    — primary (port 8000, if running)
                        UI-TARS-1.5-7B via Ollama  — auto-fallback when vLLM absent

Setup:
    ollama pull qwen3:8b
    ollama pull hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q4_K_S
    ollama serve
"""
import time
from typing import List

import httpx
from loguru import logger
from pydantic import BaseModel

from config import LLM_MODEL, VLM_OLLAMA, VLM_VLLM, LLM_BASE_URL, VLM_BASE_URL


class InferenceResponse(BaseModel):
    """Unified response type for all inference calls."""
    content: str
    model: str
    latency_ms: float
    tokens_generated: int


# Backward-compat alias — the class predates the OVMS backend being dropped.
OVMSResponse = InferenceResponse

_DEFAULT_LLM        = LLM_MODEL
_DEFAULT_VLM_VLLM   = VLM_VLLM
_DEFAULT_VLM_OLLAMA = VLM_OLLAMA

_DEFAULT_LLM_BASE_URL = LLM_BASE_URL
_DEFAULT_VLM_BASE_URL = VLM_BASE_URL


def _pick_ollama_vlm(ollama_base_url: str) -> str:
    """Return the best available vision model from Ollama.

    Preference order: UI-TARS (purpose-built GUI grounding) → qwen2.5vl → llava.
    """
    preferred_bases = ["ui-tars", "UI-TARS", "qwen2.5vl-gui", "qwen2.5vl", "llava"]
    try:
        r = httpx.get(f"{ollama_base_url}/api/tags", timeout=3.0)
        if r.status_code == 200:
            pulled = [m["name"] for m in r.json().get("models", [])]
            for base in preferred_bases:
                match = next(
                    (m for m in pulled
                     if m == base
                     or m.startswith(base + ":")
                     or m.startswith(base + "-")
                     or base.lower() in m.lower()),
                    None,
                )
                if match:
                    return match
    except Exception as e:
        logger.debug(f"[OllamaClient] VLM detection failed: {e}")
    return _DEFAULT_VLM_OLLAMA


def _post_with_retry(
    client: httpx.Client,
    url: str,
    json: dict,
    max_attempts: int = 3,
) -> httpx.Response:
    """POST with exponential back-off retry on transient failures.

    Retries on:  network errors, HTTP 429 (rate-limit), HTTP 503 (unavailable).
    Gives up on: HTTP 4xx client errors (except 429).
    """
    delay = 1.0
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_attempts):
        try:
            resp = client.post(url, json=json)
            if resp.status_code in (429, 503):
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_exc = e
            logger.warning(f"[Client] Network error (attempt {attempt + 1}/{max_attempts}): {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in (429, 503):
                raise  # non-retryable
            last_exc = e
            logger.warning(f"[Client] HTTP {e.response.status_code} — retrying (attempt {attempt + 1}/{max_attempts})")
        if attempt < max_attempts - 1:
            time.sleep(delay)
            delay *= 2.0
    raise last_exc


class OllamaClient:
    """
    Dual-backend client (models configured in config.py).
    - query_llm() → Ollama (LLM_MODEL) for planning, routing, reflection
    - query_vlm() → vLLM (VLM_VLLM) if running, else Ollama (VLM_OLLAMA)
    """

    def __init__(
        self,
        vlm_model: str = None,
        llm_model: str = _DEFAULT_LLM,
        llm_base_url: str = _DEFAULT_LLM_BASE_URL,
        vlm_base_url: str = _DEFAULT_VLM_BASE_URL,
        base_url: str = None,
        timeout: float = 600.0,
    ):
        self.llm_model = llm_model
        self.llm_base_url = base_url or llm_base_url
        self.client = httpx.Client(timeout=timeout)

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
            self.vlm_base_url = self.llm_base_url
            # Honour explicit config value; fall back to auto-detection only when unset.
            if _DEFAULT_VLM_OLLAMA:
                self.vlm_model = vlm_model or _DEFAULT_VLM_OLLAMA
            else:
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
    ) -> InferenceResponse:
        """Send screenshot + prompt to the VLM for visual grounding.

        Uses vLLM's OpenAI endpoint when vLLM is running, otherwise falls back
        to Ollama's native /api/chat endpoint (more reliable than /v1/chat/completions
        for vision models served via Ollama).
        """
        start = time.time()
        _via_vllm = self.vlm_base_url != self.llm_base_url

        if _via_vllm:
            # vLLM uses OpenAI-compatible multimodal format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
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
            resp = _post_with_retry(self.client, f"{self.vlm_base_url}/v1/chat/completions", payload)
            data = resp.json()
            try:
                content = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"[VLM] Unexpected vLLM response shape ({e}): {str(data)[:200]}")
                content = ""
            tokens = data.get("usage", {}).get("completion_tokens", 0)
        else:
            # Ollama native /api/chat — more reliable for vision models via Ollama
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            })
            payload = {
                "model": self.vlm_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            resp = _post_with_retry(self.client, f"{self.vlm_base_url}/api/chat", payload)
            data = resp.json()
            content = (data.get("message") or {}).get("content") or ""
            tokens = data.get("eval_count", 0)

        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[VLM] {prompt[:60]}… → {content[:100]} ({latency_ms:.0f}ms)")
        return InferenceResponse(
            content=content,
            model=self.vlm_model,
            latency_ms=latency_ms,
            tokens_generated=tokens,
        )

    def query_llm(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict = None,
    ) -> InferenceResponse:
        """Send text messages to the configured LLM via Ollama's native /api/chat.

        Uses Ollama native API so we can pass:
          think=False        — suppresses chain-of-thought filling
          format=schema      — structured JSON output
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

        resp = _post_with_retry(self.client, f"{self.llm_base_url}/api/chat", payload)
        data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM] → {content[:100]} ({latency_ms:.0f}ms)")
        return InferenceResponse(
            content=content,
            model=self.llm_model,
            latency_ms=latency_ms,
            tokens_generated=data.get("eval_count", 0),
        )

    def check_health(self) -> dict:
        results = {}
        _vlm_via_ollama = self.vlm_base_url == self.llm_base_url

        try:
            r = self.client.get(f"{self.llm_base_url}/api/tags", timeout=5.0)
            if r.status_code == 200:
                pulled = [m["name"] for m in r.json().get("models", [])]
                llm_base = self.llm_model.split(":")[0]
                llm_ok = any(llm_base in m for m in pulled)
                results[f"LLM ({self.llm_model})"] = (
                    "OK" if llm_ok else f"NOT PULLED — run: ollama pull {self.llm_model}"
                )
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
