# core/ovms_client.py
"""OpenVINO™ Model Server (OVMS) inference client.

A single OVMS instance serves BOTH models on one OpenAI-compatible endpoint:
    LLM (text reasoning):   qwen3-8b-int4-ov        — planning, routing, reflection
    VLM (visual grounding): ui-tars-1.5-7b-int4-ov  — GUI grounding, visual verify

Both are reached at  POST {OVMS_BASE_URL}/v3/chat/completions  and selected by the
"model" field in the request body. start.py prepares the models and launches OVMS.
"""
import time

import httpx
from loguru import logger
from pydantic import BaseModel

from config import LLM_MODEL, OVMS_BASE_URL, VLM_MODEL


class InferenceResponse(BaseModel):
    """Unified response type for all inference calls."""

    content: str
    model: str
    latency_ms: float
    tokens_generated: int


# Backward-compat alias kept for any external importers.
OVMSResponse = InferenceResponse

_DEFAULT_LLM      = LLM_MODEL
_DEFAULT_VLM      = VLM_MODEL
_DEFAULT_BASE_URL = OVMS_BASE_URL

_CHAT_PATH = "/v3/chat/completions"


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


class OVMSClient:
    """OpenVINO Model Server client (models configured in config.py).

    Both models are served by the same OVMS instance on OVMS_BASE_URL via the
    OpenAI-compatible /v3/chat/completions endpoint:
      - query_llm() → LLM_MODEL  for planning, routing, reflection
      - query_vlm() → VLM_MODEL  for grounding / visual verification
    """

    def __init__(
        self,
        vlm_model: str = _DEFAULT_VLM,
        llm_model: str = _DEFAULT_LLM,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 600.0,
    ):
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        # Kept for backward compatibility with callers / tests that inspected the
        # old dual-backend attributes (single endpoint now, so they're equal).
        self.llm_base_url = self.base_url
        self.vlm_base_url = self.base_url
        logger.info(
            f"[OVMSClient] LLM={self.llm_model}  VLM={self.vlm_model}  "
            f"endpoint={self.base_url}{_CHAT_PATH}"
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

        Uses the OpenAI-compatible multimodal chat format — the image is passed
        inline as a base64 data URL.
        """
        start = time.time()
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
        resp = _post_with_retry(self.client, f"{self.base_url}{_CHAT_PATH}", payload)
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"[VLM] Unexpected OVMS response shape ({e}): {str(data)[:200]}")
            content = ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)

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
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict = None,
    ) -> InferenceResponse:
        """Send text messages to the LLM via OVMS' OpenAI-compatible endpoint.

        - chat_template_kwargs.enable_thinking=False suppresses Qwen3 chain-of-thought.
        - response_schema (when given) requests structured JSON output via the
          OpenAI response_format / json_schema interface (guided generation).
        """
        start = time.time()
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        resp = _post_with_retry(self.client, f"{self.base_url}{_CHAT_PATH}", payload)
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"[LLM] Unexpected OVMS response shape ({e}): {str(data)[:200]}")
            content = ""
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM] → {content[:100]} ({latency_ms:.0f}ms)")
        return InferenceResponse(
            content=content,
            model=self.llm_model,
            latency_ms=latency_ms,
            tokens_generated=tokens,
        )

    def check_health(self) -> dict:
        """Report whether each configured servable is loaded and AVAILABLE.

        Queries OVMS' /v1/config endpoint, which lists every servable and its
        version states.
        """
        results = {}
        try:
            r = self.client.get(f"{self.base_url}/v1/config", timeout=5.0)
        except Exception as e:
            results["OVMS"] = f"UNREACHABLE — is the model server running? run start.py ({e})"
            return results

        if r.status_code != 200:
            results["OVMS"] = f"HTTP {r.status_code}"
            return results

        try:
            config = r.json()
        except Exception as e:
            results["OVMS"] = f"BAD RESPONSE — {e}"
            return results

        for label, model in (("LLM", self.llm_model), ("VLM", self.vlm_model)):
            results[f"{label} ({model})"] = (
                "OK" if _servable_available(config, model)
                else f"NOT LOADED — run start.py to prepare/serve {model}"
            )
        return results


def _servable_available(config: dict, model: str) -> bool:
    """True if `model` appears in an OVMS /v1/config response in AVAILABLE state.

    OVMS returns a mapping of servable-name → {"model_version_status": [...]} (or a
    list of such entries depending on version), so handle both shapes defensively.
    """
    def _entry_ok(entry: dict) -> bool:
        for st in entry.get("model_version_status", []):
            if str(st.get("state", "")).upper() == "AVAILABLE":
                return True
        return False

    if isinstance(config, dict):
        if model in config and isinstance(config[model], dict):
            return _entry_ok(config[model])
        # Some versions nest under a list or use other keys — fall back to a scan.
        for key, val in config.items():
            if key == model and isinstance(val, dict) and _entry_ok(val):
                return True
            if isinstance(val, dict) and val.get("name") == model and _entry_ok(val):
                return True
    if isinstance(config, list):
        for entry in config:
            if isinstance(entry, dict) and entry.get("name") == model and _entry_ok(entry):
                return True
    return False
