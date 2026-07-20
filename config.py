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

LLM_MODEL = "qwen3-14b-int4-ov"         # text reasoning — routing, planning, reflection
VLM_MODEL = "ui-tars-1.5-7b-int8-ov"    # GUI grounding & visual verification (UI-TARS)

# The 14B LLM (INT4) is the default: live OCR-path runs showed the 8B's
# reasoning is the bottleneck — hallucinated verifier verdicts ("Calendar
# panel is active" while the click landed on Meet), planners returning []
# ("goal achieved") in rejection loops, and dropped decomposition steps. The
# 14B trades per-call speed (~1.5-2x slower generation on the iGPU) for
# fewer of those, which live cost far more time than the extra tokens.
# Budget on a 27 GB GPU: 14B-int4 (~9.7 GB) + 7B-int8 VLM (~7.5 GB) + KV
# caches (4 + 2 GB) + runtime overhead ≈ 24.7 GB.
# FP16 UI-TARS (~15 GB weights) does NOT fit alongside the LLM + KV on 27 GB.
# For faster (but less reliable) reasoning the 8B remains available — its
# exported weights stay on disk after a swap, so switching back is instant:
#   LLM_MODEL  = "qwen3-8b-int4-ov"
#   LLM_SOURCE = "OpenVINO/Qwen3-8B-int4-ov"

# ── Model sources (where start.py fetches / converts them from) ─────────────────
# LLM_SOURCE is a pre-converted OpenVINO IR repo on Hugging Face — OVMS pulls it
# directly (already INT4, so LLM_WEIGHT_FORMAT is a no-op for it). VLM_SOURCE is
# the upstream UI-TARS checkpoint; start.py converts it to OpenVINO IR at
# VLM_WEIGHT_FORMAT with optimum-cli on first run (no pre-built OV build exists).

LLM_SOURCE = "OpenVINO/Qwen3-14B-int4-ov"
VLM_SOURCE = "ByteDance-Seed/UI-TARS-1.5-7B"

# Quantization each model is exported at. The VLM is converted locally, so this
# is where its precision is chosen: "int8" trades ~2.5 GB more VRAM for more
# accurate coordinate grounding than "int4". "fp16" is too large to co-reside
# with the LLM on 27 GB. Changing VLM_WEIGHT_FORMAT requires the servable name
# (VLM_MODEL above) to change too, so start.py re-exports instead of reusing the
# old precision's directory.
LLM_WEIGHT_FORMAT = "int4"
VLM_WEIGHT_FORMAT = "int8"

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

# KV-cache budget per model (GB). Both models share the same GPU memory, so the
# total allocation is the SUM of the two values plus the model weights.
# The LLM gets the bigger cache: router decomposition, task replanning, and
# batch planning carry the longest prompts. The VLM only ever sees one
# screenshot + a short instruction per request, so 2 GB is plenty.
# start.py applies changes here to already-exported models automatically
# (it patches the baked cache_size in each servable's graph.pbtxt).
LLM_KV_CACHE_GB = 4
VLM_KV_CACHE_GB = 2

# Local directory OVMS uses as its model repository (holds the IR models and the
# generated config.json). Relative to the project root.
MODEL_REPOSITORY_PATH = "models"

# ── Grounding ───────────────────────────────────────────────────────────────────
# Coordinate convention of the served UI-TARS build. The prompt asks for the
# native 0-1000 scale, but quantized conversions sometimes emit raw pixels of
# the input image instead — and values ≤ 1000 fit both readings, so "auto" has
# to guess (heuristic in grounding._parse_coords). To make parsing deterministic:
# run tests/live/test_vlm_coordinates.py on the target machine once, see which
# convention the model actually uses, and pin this to "norm1000" or "pixels".
# NOTE: re-run that test after changing VLM_WEIGHT_FORMAT — a different
# quantization can change which convention the model emits.
VLM_COORD_SPACE = "auto"   # "auto" | "norm1000" | "pixels"
