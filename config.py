# config.py — single source of truth for all model names.
# Change here; everything else picks it up automatically.

LLM_MODEL = "qwen3:8b"               # Ollama LLM: routing, planning, reflection
VLM_OLLAMA = "qwen2.5vl-gui"         # Ollama VLM fallback (used when vLLM is absent)
VLM_VLLM   = "ByteDance-Seed/UI-TARS-1.5-7B"  # vLLM primary VLM (optional, port 8000)

LLM_BASE_URL = "http://localhost:11434"  # Ollama server
VLM_BASE_URL = "http://localhost:8000"   # vLLM server (optional)
