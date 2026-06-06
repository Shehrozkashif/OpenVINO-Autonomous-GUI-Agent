# config.py — single source of truth for all model and server settings.
# Change values here; everything else picks them up automatically.

# ── Models ────────────────────────────────────────────────────────────────────

LLM_MODEL  = "qwen3:8b"                         # Ollama LLM — routing, planning, reflection
VLM_OLLAMA = "qwen2.5vl:3b"                     # Ollama VLM fallback — 3.2GB fits alongside qwen3:8b in 6GB VRAM
VLM_VLLM   = "ByteDance-Seed/UI-TARS-1.5-7B"   # vLLM primary VLM (port 8000, if running)

# ── Endpoints ─────────────────────────────────────────────────────────────────

LLM_BASE_URL = "http://localhost:11434"   # Ollama server
VLM_BASE_URL = "http://localhost:8000"    # vLLM server (optional)

# ── GPU / Server settings ─────────────────────────────────────────────────────
# These are used by start.py to generate the correct vLLM launch command.
# They do NOT affect the HTTP client — vLLM handles GPU assignment internally.

# Which GPU index(es) Ollama uses (set via ROCR_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES)
# "0"   → GPU 0 only
# "0,1" → both GPUs (for Ollama with tensor parallelism)
# ""    → let Ollama choose (uses all available GPUs)
OLLAMA_GPU_DEVICES = "0"

# Which GPU index(es) vLLM uses (set via HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES)
# Recommended for 2×24GB setup: "1" so Ollama and vLLM each own one GPU
# For maximum vLLM throughput: "0,1" with TENSOR_PARALLEL_SIZE = 2 (Ollama falls to CPU)
VLLM_GPU_DEVICES = "1"

# Number of GPUs vLLM splits the VLM across (tensor parallelism)
# 1 → single GPU   2 → both GPUs (set VLLM_GPU_DEVICES = "0,1")
TENSOR_PARALLEL_SIZE = 1

# Fraction of GPU VRAM vLLM may use (0.85 leaves headroom for other processes)
VLM_GPU_MEMORY_UTIL = 0.85

# VLM precision — float16 is safe for AMD ROCm; bfloat16 requires CDNA2+ (MI200+)
VLM_DTYPE = "float16"

# VLM context window (tokens) — 4096 is enough for grounding + reflection
VLM_MAX_MODEL_LEN = 4096
