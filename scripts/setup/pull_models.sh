#!/usr/bin/env bash
# scripts/setup/pull_models.sh
# Downloads OpenVINO-optimized models from HuggingFace Hub.
# These models are loaded directly by the agent via openvino-genai natively.

set -e

MODELS_DIR="$(pwd)/models/OpenVINO"
mkdir -p "$MODELS_DIR"

echo "==> Installing/Updating huggingface_hub..."
pip install -q -U huggingface_hub

echo ""
echo "==> Downloading models using huggingface_hub.snapshot_download"
python3 - <<'PY'
from huggingface_hub import snapshot_download
import os

MODELS_DIR = os.path.join(os.getcwd(), "models", "OpenVINO")
models = [
    # VLM: Qwen2.5-VL — trained for UI grounding, gives correct screen coordinates
    ("OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov", "Qwen2.5-VL-7B-Instruct-int4-ov"),
    # LLM: DeepSeek-R1 — used by router and planner for task decomposition
    ("OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov", "DeepSeek-R1-Distill-Qwen-7B-int4-ov"),
]
for repo_id, local_name in models:
    dest = os.path.join(MODELS_DIR, local_name)
    print(f"Downloading {repo_id} to {dest} ...")
    snapshot_download(repo_id, local_dir=dest, repo_type="model")

print("==> All models downloaded")
PY

ls -lh "$MODELS_DIR"
