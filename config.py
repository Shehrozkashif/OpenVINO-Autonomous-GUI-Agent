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

# KV-cache budget PER MODEL (GB).  Both models share the same GPU memory, so the
# total KV-cache allocation is 2 × this value.  On a 16 GB GPU with two INT4 7–8 B
# models (~5 GB weights each), 2 GB per model is a safe default.  Increase on GPUs
# with more VRAM (e.g. 4–6 on a 24 GB card) for longer context windows.
KV_CACHE_SIZE_GB = 2

# Local directory OVMS uses as its model repository (holds the IR models and the
# generated config.json). Relative to the project root.
MODEL_REPOSITORY_PATH = "models"

# ── Grounding ───────────────────────────────────────────────────────────────────
# Coordinate convention of the served UI-TARS build. The prompt asks for the
# native 0-1000 scale, but INT4 conversions sometimes emit raw pixels of the
# input image instead — and values ≤ 1000 fit both readings, so "auto" has to
# guess (heuristic in grounding._parse_coords). To make parsing deterministic:
# run tests/live/test_vlm_coordinates.py on the target machine once, see which
# convention the model actually uses, and pin this to "norm1000" or "pixels".
VLM_COORD_SPACE = "auto"   # "auto" | "norm1000" | "pixels"
