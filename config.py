# config.py — single source of truth for all model and server settings.
# Change values here; everything else picks them up automatically.
#
# Inference runs entirely through OpenVINO™ Model Server (OVMS). A single OVMS
# instance serves BOTH models on one OpenAI-compatible endpoint (port 8000):
#   • the LLM  (planning / routing / reflection)
#   • the VLM  (visual grounding / verification)
# Requests are routed to the right model by the "model" field in the request body.

# ── Models (OVMS servable names) ────────────────────────────────────────────────
# These names must match the servable names registered in the OVMS config.json
# that start.py generates (model_repository_path below).

LLM_MODEL = "qwen3-8b-int4-ov"          # text reasoning — routing, planning, reflection
VLM_MODEL = "ui-tars-1.5-7b-int4-ov"    # GUI grounding & visual verification (UI-TARS)

# ── Model sources (where start.py fetches / converts them from) ─────────────────
# LLM_SOURCE is a pre-converted OpenVINO IR repo on Hugging Face — OVMS pulls it
# directly. VLM_SOURCE is the upstream UI-TARS checkpoint; start.py converts it to
# OpenVINO INT4 IR with optimum-cli on first run (no pre-built OV build exists).

LLM_SOURCE = "OpenVINO/Qwen3-8B-int4-ov"
VLM_SOURCE = "ByteDance-Seed/UI-TARS-1.5-7B"

# ── Endpoint ────────────────────────────────────────────────────────────────────

OVMS_BASE_URL = "http://localhost:8000"   # OpenVINO Model Server (OpenAI-compatible)
OVMS_REST_PORT = 8000

# ── Server settings ─────────────────────────────────────────────────────────────
# Used by start.py to launch OVMS. The HTTP client only needs OVMS_BASE_URL.

# Inference device passed to OVMS as --target_device.
#   "GPU"  → Intel iGPU / Arc discrete GPU
#   "CPU"  → portable fallback (slower for 7–8 B models)
#   "NPU"  → Intel Core Ultra NPU (limited model support)
#   "AUTO" → let OpenVINO pick the best available device
TARGET_DEVICE = "GPU"

# Local directory OVMS uses as its model repository (holds the IR models and the
# generated config.json). Relative to the project root.
MODEL_REPOSITORY_PATH = "models"
