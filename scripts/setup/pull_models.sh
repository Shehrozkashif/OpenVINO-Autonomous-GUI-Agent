#!/bin/bash
# scripts/setup/pull_models.sh
# Downloads OpenVINO-optimized models from HuggingFace Hub.
# Must be run from project root with venv active.

set -e

MODELS_DIR="$(pwd)/models/OpenVINO"
mkdir -p "$MODELS_DIR"

echo "==> Installing/Updating HuggingFace Hub CLI..."
pip install -q -U huggingface_hub

echo ""
echo "==> Downloading Phi-3.5-vision-instruct-int4-ov (~2.3 GB)..."
echo "    This is the VLM used by the UI Grounding Agent."
hf download OpenVINO/Phi-3.5-vision-instruct-int4-ov \
  --local-dir "$MODELS_DIR/Phi-3.5-vision-instruct-int4-ov"

echo ""
echo "==> Downloading DeepSeek-R1-Distill-Qwen-7B-int4-cw-ov (~4.1 GB)..."
echo "    This is the LLM used by the Router, Planning, and Reflection agents."
hf download OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov \
  --local-dir "$MODELS_DIR/DeepSeek-R1-Distill-Qwen-7B-int4-ov"

echo ""
echo "==> All models downloaded to: $MODELS_DIR"
ls -lh "$MODELS_DIR"
