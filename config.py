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
VLM_MODEL = "ui-tars-1.5-7b-int8-ov"    # GUI grounding & visual verification (UI-TARS)

# The 8B LLM (INT4) is the default. A two-run 14B trial (2026-07-19) lost on
# this hardware: planning calls went from 7-15 s to 45-101 s, runs took 17
# then 36 minutes to hit the same perception-level wall the 8B hits in 6-8,
# its verbose JSON truncated the planner twice, and its plans were no better
# (it clicked the window TITLE as a target and typed into a Teams chat). The
# blocking failures are OCR-blind WebView2 state, which model scale does not
# fix. Budget on a 27 GB GPU: 8B-int4 (~5.5 GB) + 7B-int8 VLM (~7.5 GB) +
# KV (4 + 2 GB) + overhead ≈ 20.5 GB.
# FP16 UI-TARS (~15 GB weights) does NOT fit alongside the LLM + KV on 27 GB.
# The 14B export stays on disk — re-swapping is instant if reasoning ever
# becomes the bottleneck (14B-int4 ~9.7 + int8 VLM ~7.5 + KV 6 ≈ 24.7 GB):
#   LLM_MODEL  = "qwen3-14b-int4-ov"
#   LLM_SOURCE = "OpenVINO/Qwen3-14B-int4-ov"

# ── Model sources (where start.py fetches / converts them from) ─────────────────
# LLM_SOURCE is a pre-converted OpenVINO IR repo on Hugging Face — OVMS pulls it
# directly (already INT4, so LLM_WEIGHT_FORMAT is a no-op for it). VLM_SOURCE is
# the upstream UI-TARS checkpoint; start.py converts it to OpenVINO IR at
# VLM_WEIGHT_FORMAT with optimum-cli on first run (no pre-built OV build exists).

LLM_SOURCE = "OpenVINO/Qwen3-8B-int4-ov"
VLM_SOURCE = "ByteDance-Seed/UI-TARS-1.5-7B"

# Quantization each model is exported at. The VLM is converted locally, so this
# is where its precision is chosen: "int8" trades ~2.5 GB more VRAM for more
# accurate coordinate grounding than "int4".
# fp16 was tested live (2026-07-20, 27.5 GB budget): the 15.5 GB fp16 VLM DOES
# co-reside with the int4 LLM if KV is trimmed to 3+2 GB — but its grounding
# accuracy was IDENTICAL to int8 (Chat 3px, New meeting 36px, Calendar ~250px
# both). The model's small/stacked-icon miss is BEHAVIORAL, not precision-bound,
# so fp16 buys nothing while costing ~8 GB and KV headroom. int8 stays the pick.
# Changing VLM_WEIGHT_FORMAT requires the servable name (VLM_MODEL above) to
# change too, so start.py re-exports instead of reusing the old precision's dir.
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
# Coordinate convention of the served UI-TARS build. UI-TARS-1.5 is qwen2.5-VL
# based, which emits ABSOLUTE pixel coordinates in the smart-resized image space
# (each side rounded to a multiple of 28; no budget downscale for our screen
# sizes). Pinned to "pixels" — a live calibration against UIA ground truth
# (tests/live/test_vlm_coordinates.py) confirmed pixel/resized mapping beats the
# 0-1000 reading for distinct targets (e.g. a top-right button: /1000 lands
# off-screen, /resized lands within ~30px). grounding._CoordSpace divides by the
# /28-rounded (resized) dimension. Re-calibrate after changing VLM_WEIGHT_FORMAT.
VLM_COORD_SPACE = "pixels"   # "auto" | "norm1000" | "pixels"

# ── Interaction ─────────────────────────────────────────────────────────────────
# Force every grounded click to be delivered with the REAL MOUSE — the cursor
# glides to the target and presses — instead of the faster UIA pattern-invoke
# shortcut (which actuates a control through the accessibility API with NO cursor
# motion). UIA still supplies the exact coordinates, so click accuracy is
# unchanged; this only changes HOW the click is delivered. Default True: a
# desktop-automation agent should visibly drive mouse + keyboard (demos, and
# operators who need to trust what they see). Set False to prefer UIA invoke,
# which is more robust on the few WebView2 buttons that swallow synthesized pixel
# clicks. Env AGENT_FORCE_MOUSE=1/0 overrides this per-run.
FORCE_MOUSE = True
