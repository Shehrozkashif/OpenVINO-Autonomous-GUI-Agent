#!/usr/bin/env python3
"""
scripts/setup/pull_models.py
Cross-platform model downloader (works on Windows, Linux, macOS).
Downloads OpenVINO-optimized models from HuggingFace Hub.

Run once before first use:
    python scripts/setup/pull_models.py
"""
import os
import subprocess
import sys

MODELS = [
    # VLM: Qwen2.5-VL — trained for UI grounding, gives accurate screen coordinates
    ("OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov", "Qwen2.5-VL-7B-Instruct-int4-ov"),
    # LLM: DeepSeek-R1 — used by router and planner for task decomposition
    ("OpenVINO/DeepSeek-R1-Distill-Qwen-7B-int4-ov", "DeepSeek-R1-Distill-Qwen-7B-int4-ov"),
]

def main():
    # Resolve models directory relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir = os.path.join(repo_root, "models", "OpenVINO")
    os.makedirs(models_dir, exist_ok=True)

    print("==> Installing/updating huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub"])

    from huggingface_hub import snapshot_download

    for repo_id, local_name in MODELS:
        dest = os.path.join(models_dir, local_name)
        print(f"\n==> Downloading {repo_id}")
        print(f"    → {dest}")
        snapshot_download(repo_id, local_dir=dest, repo_type="model")
        print(f"    ✓ Done")

    print("\n==> All models downloaded.")
    print(f"    Location: {models_dir}")
    print("\nNext step: python main.py  (select 'Direct OpenVINO' in the UI)")

if __name__ == "__main__":
    main()
