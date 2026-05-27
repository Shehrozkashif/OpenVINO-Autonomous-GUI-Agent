# core/pipeline/optimized_pipeline.py
"""
Direct OpenVINO GenAI pipelines — bypasses OVMS for lower latency.
Use this after Phase 9, replacing OVMSClient calls.
"""
import openvino_genai as ov_genai
from PIL import Image


class OptimizedVLMPipeline:
    """Direct VLMPipeline with KV caching enabled."""

    def __init__(self, model_path: str, device: str = "AUTO"):
        self.pipe = ov_genai.VLMPipeline(
            model_path,
            device=device,
            config={
                "PERFORMANCE_HINT": "LATENCY",
                "KV_CACHE_PRECISION": "f16",
            }
        )
        self.pipe.start_chat()

    def ground(self, prompt: str, image: Image.Image) -> str:
        return self.pipe.generate(
            prompt, images=image, max_new_tokens=60, temperature=0.1
        )

    def reset_context(self):
        """Reset KV cache between tasks to avoid context bleed."""
        self.pipe.finish_chat()
        self.pipe.start_chat()

    def __del__(self):
        try:
            self.pipe.finish_chat()
        except Exception:
            pass


class OptimizedLLMPipeline:
    """Direct LLMPipeline for Router/Planner/Reflection."""

    # DeepSeek-R1 chat template tokens
    _TEMPLATE = "<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>"

    def __init__(self, model_path: str, device: str = "AUTO"):
        self.pipe = ov_genai.LLMPipeline(
            model_path,
            device=device,
            config={"PERFORMANCE_HINT": "LATENCY"}
        )
        self.pipe.start_chat()

    def generate(self, messages: list, max_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        prompt = self._format(messages)
        return self.pipe.generate(
            prompt, max_new_tokens=max_tokens, temperature=temperature
        )

    def _format(self, messages: list) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return self._TEMPLATE.format(system=system, user=user)

    def reset(self):
        self.pipe.finish_chat()
        self.pipe.start_chat()

    def __del__(self):
        try:
            self.pipe.finish_chat()
        except Exception:
            pass