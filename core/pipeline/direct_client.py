# core/pipeline/direct_client.py
"""
Direct OpenVINO inference client — runs models in-process via openvino-genai.
Intended for Intel AI PCs (Arc GPU / NPU). Drop-in replacement for OVMSClient.

Expected model layout under models/OpenVINO/:
  DeepSeek-R1-Distill-Qwen-7B-int4-ov/     (LLM: router, planner)
  Qwen2.5-VL-7B-Instruct-int4-ov/           (VLM: visual grounding)

Download with:  python scripts/setup/pull_models.py
"""
import base64
import io
import os
import time
from typing import List

from loguru import logger
from PIL import Image

from core.pipeline.optimized_pipeline import OptimizedLLMPipeline, OptimizedVLMPipeline
from core.pipeline.ovms_client import OVMSResponse


class DirectOpenVINOClient:
    """
    Native OpenVINO inference backend.
    Loads LLM and VLM once at startup; all agents call through this instance.
    """

    def __init__(self, device: str = "AUTO"):
        model_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "OpenVINO")
        )
        llm_path = self._resolve_model_path(model_dir, "DeepSeek-R1-Distill-Qwen-7B-int4-ov")
        vlm_path = self._resolve_model_path(model_dir, "Qwen2.5-VL-7B-Instruct-int4-ov")

        if llm_path is None:
            raise FileNotFoundError(
                f"LLM model not found under {model_dir}/DeepSeek-R1-Distill-Qwen-7B-int4-ov\n"
                "Run: python scripts/setup/pull_models.py\n"
                "Or switch to 'Ollama' in Settings."
            )
        if vlm_path is None:
            raise FileNotFoundError(
                f"VLM model not found under {model_dir}/Qwen2.5-VL-7B-Instruct-int4-ov\n"
                "Run: python scripts/setup/pull_models.py\n"
                "Or switch to 'Ollama' in Settings."
            )

        logger.info(f"[DirectOV] Loading LLM from {llm_path} on device={device}")
        self.llm = OptimizedLLMPipeline(llm_path, device=device)

        logger.info(f"[DirectOV] Loading VLM from {vlm_path} on device={device}")
        self.vlm_pipeline = OptimizedVLMPipeline(vlm_path, device=device)

    @staticmethod
    def _resolve_model_path(model_dir: str, model_name: str) -> str | None:
        """
        Locate the directory that contains openvino_model.xml.

        Handles two layouts HuggingFace repos may use:
          Layout A (OVMS style):  model_dir/model_name/1/openvino_model.xml
          Layout B (flat):        model_dir/model_name/openvino_model.xml
        """
        base = os.path.join(model_dir, model_name)
        versioned = os.path.join(base, "1")
        if os.path.isfile(os.path.join(versioned, "openvino_model.xml")):
            return versioned
        if os.path.isfile(os.path.join(base, "openvino_model.xml")):
            return base
        return None

    def query_llm(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> OVMSResponse:
        start = time.time()
        content = self.llm.generate(messages, max_tokens=max_tokens, temperature=temperature)
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[LLM-Direct] → {content[:100]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content,
            model="deepseek-r1-direct",
            latency_ms=latency_ms,
            tokens_generated=0,
        )

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ) -> OVMSResponse:
        start = time.time()
        img_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(img_data))
        content = self.vlm_pipeline.ground(prompt, image)
        latency_ms = (time.time() - start) * 1000
        logger.debug(f"[VLM-Direct] {prompt[:50]}… → {content[:80]} ({latency_ms:.0f}ms)")
        return OVMSResponse(
            content=content,
            model="qwen2.5-vl-direct",
            latency_ms=latency_ms,
            tokens_generated=0,
        )

    def check_health(self) -> dict:
        return {
            "LLM (DeepSeek-R1-Direct)": "OK",
            "VLM (Qwen2.5-VL-Direct)": "OK",
        }
