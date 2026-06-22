#!/usr/bin/env python3
"""
Desktop GUI Agent — single entry point.

    python start.py

Does everything automatically:
  1. Platform setup (Linux Qt libs / Windows UIA)
  2. Detect GPU (Intel / AMD / NVIDIA)
  3. Prepare both models in the OpenVINO Model Server (OVMS) repository:
       • LLM  qwen3-8b-int4-ov        (pulled pre-converted from Hugging Face)
       • VLM  ui-tars-1.5-7b-int4-ov  (converted from UI-TARS on first run)
  4. Start OVMS serving both models on one OpenAI-compatible endpoint (port 8000)
       — native ovms/ovms.exe if available, otherwise the Docker image
  5. Wait for the server to be ready
  6. Launch the agent UI
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from config import (LLM_MODEL, LLM_SOURCE, VLM_MODEL, VLM_SOURCE,
                    OVMS_BASE_URL, OVMS_REST_PORT, TARGET_DEVICE,
                    MODEL_REPOSITORY_PATH)

_OS = platform.system()
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, MODEL_REPOSITORY_PATH)
_CONFIG_JSON = os.path.join(_REPO, "config.json")

# Pinned OVMS ref used to fetch the model-export helper when it isn't already present.
_OVMS_REF = "releases/2025/3"
_EXPORT_TOOL_URL = (
    f"https://raw.githubusercontent.com/openvinotoolkit/model_server/"
    f"{_OVMS_REF}/demos/common/export_models/export_model.py"
)
_DOCKER_IMAGE = "openvino/model_server:latest-gpu"


# ── Colour helpers ────────────────────────────────────────────────────────────
def _green(s):  return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _cyan(s):   return f"\033[96m{s}\033[0m"


def banner():
    print(_bold("\n╔══════════════════════════════════════════════╗"))
    print(_bold("║       Desktop GUI Agent — Startup Check      ║"))
    print(_bold("╚══════════════════════════════════════════════╝\n"))


# ── 1. Fix libxcb-cursor on Linux (no sudo needed) ───────────────────────────
def setup_linux_libs():
    lib_path = os.path.expanduser("~/.local_xcb/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0")
    if os.path.exists(lib_path):
        return
    print(_yellow("  [SETUP] libxcb-cursor not found — extracting (one-time)..."))
    try:
        import glob
        subprocess.run(["apt-get", "download", "libxcb-cursor0"],
                       cwd="/tmp", check=True, capture_output=True)
        debs = glob.glob("/tmp/libxcb-cursor0*.deb")
        if not debs:
            raise FileNotFoundError("libxcb-cursor0 .deb not downloaded")
        dest = os.path.expanduser("~/.local_xcb")
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["dpkg-deb", "-x", debs[0], dest], check=True, capture_output=True)
        print(_green("  [OK] libxcb-cursor extracted"))
    except Exception as e:
        print(_yellow(f"  [WARN] Could not extract libxcb-cursor: {e}"))


def inject_linux_env():
    lib_dir = os.path.expanduser("~/.local_xcb/usr/lib/x86_64-linux-gnu")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in existing:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir


# ── 2. GPU detection ─────────────────────────────────────────────────────────

def check_gpus():
    """Detect GPUs, print a summary, return (gpu_type, gpus)."""
    from utils.platform_utils import detect_gpus

    print(_bold("\nGPU Detection:"))
    gpus = detect_gpus()
    if not gpus:
        print(_yellow("  No GPU detected — OVMS will fall back to CPU (slow)"))
        return "cpu", []

    backend = gpus[0].backend
    total = sum(g.vram_gb for g in gpus)
    for g in gpus:
        vram = f"  {g.vram_gb}GB VRAM" if g.vram_mb else ""
        print(_green(f"  [{backend.upper()}] GPU{g.index}: {g.name}{vram}"))
    if total:
        print(_green(f"  Total: {len(gpus)} {backend.upper()} GPU(s), {total:.1f}GB VRAM"))
    return backend, gpus


# ── 3. Locate OVMS (native binary or Docker) ─────────────────────────────────

def find_ovms_binary() -> str:
    """Return the path to a native ovms executable, or '' if none is found.

    Honours the OVMS_DIR / OVMS_PATH env vars (the OVMS Windows package extracts
    to a folder containing ovms.exe), then falls back to PATH.
    """
    exe = "ovms.exe" if _OS == "Windows" else "ovms"
    for env in ("OVMS_PATH", "OVMS_DIR"):
        base = os.environ.get(env)
        if base:
            cand = base if os.path.isfile(base) else os.path.join(base, exe)
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(base, "ovms", exe)  # common extracted layout
            if os.path.isfile(cand):
                return cand
    found = shutil.which("ovms") or shutil.which(exe)
    return found or ""


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return False


# ── 4. Model preparation (export into the OVMS repository) ────────────────────

def _ensure_export_tool() -> str:
    """Return a path to export_model.py, downloading it once if necessary."""
    tools_dir = os.path.join(_HERE, "tools", "ovms")
    os.makedirs(tools_dir, exist_ok=True)
    dest = os.path.join(tools_dir, "export_model.py")
    if os.path.isfile(dest):
        return dest
    print(_yellow("  [SETUP] Fetching OVMS export_model.py (one-time)..."))
    try:
        import urllib.request
        urllib.request.urlretrieve(_EXPORT_TOOL_URL, dest)
        print(_green("  [OK] export_model.py downloaded"))
        return dest
    except Exception as e:
        print(_red(f"  [FAIL] Could not download export_model.py: {e}"))
        print(_yellow(f"        Download it manually from {_EXPORT_TOOL_URL}"))
        print(_yellow(f"        and place it at {dest}"))
        return ""


def _ensure_hf_cli():
    """Make the legacy `huggingface-cli` command resolve.

    export_model.py downloads pre-converted OpenVINO models with
    `huggingface-cli download ...`, but huggingface_hub >= 1.0 removed that
    command in favour of `hf`. If only `hf` is present, drop a thin shim so the
    legacy invocation keeps working.
    """
    if shutil.which("huggingface-cli"):
        return
    if not shutil.which("hf"):
        return  # nothing to shim onto — optimum-intel install provides one of them
    scripts_dir = os.path.dirname(sys.executable)
    try:
        if _OS == "Windows":
            shim = os.path.join(scripts_dir, "huggingface-cli.bat")
            with open(shim, "w") as f:
                f.write("@echo off\r\nhf %*\r\n")
        else:
            shim = os.path.join(scripts_dir, "huggingface-cli")
            with open(shim, "w") as f:
                f.write('#!/bin/sh\nexec hf "$@"\n')
            os.chmod(shim, 0o755)
        print(_green("  [OK] huggingface-cli → hf shim created"))
    except Exception as e:
        print(_yellow(f"  [WARN] Could not create huggingface-cli shim: {e}"))


def _model_already_exported(model_name: str) -> bool:
    """True if model_name is present in the OVMS repo config.json."""
    if not os.path.isfile(_CONFIG_JSON):
        return False
    try:
        with open(_CONFIG_JSON) as f:
            cfg = json.load(f)
    except Exception:
        return False
    names = [e.get("name") for e in cfg.get("mediapipe_config_list", [])]
    names += [e.get("config", {}).get("name") for e in cfg.get("model_config_list", [])]
    return model_name in names


def _export_model(export_tool: str, source_model: str, model_name: str,
                  device: str) -> bool:
    """Run export_model.py to convert/pull a model into the OVMS repository.

    The `text_generation` subcommand handles both plain LLMs and vision-language
    models — it runs optimum-cli for non-prebuilt sources (e.g. UI-TARS), writes
    a graph.pbtxt, and appends the servable to config.json.
    """
    if _model_already_exported(model_name):
        print(_green(f"  [OK] {model_name:<24} already in repository"))
        return True

    print(_yellow(f"  [..] {model_name:<24} preparing from {source_model} (first run is slow)..."))
    cmd = [
        sys.executable, export_tool, "text_generation",
        "--source_model", source_model,
        "--model_name", model_name,
        "--weight-format", "int4",
        "--config_file_path", _CONFIG_JSON,
        "--model_repository_path", _REPO,
        "--target_device", device,
    ]
    ret = subprocess.run(cmd, cwd=_HERE).returncode
    if ret == 0 and _model_already_exported(model_name):
        print(_green(f"  [OK] {model_name} ready"))
        return True
    print(_red(f"  [FAIL] Could not export {model_name}"))
    return False


def ensure_models(device: str) -> bool:
    """Make sure both servables exist in the OVMS repository / config.json."""
    os.makedirs(_REPO, exist_ok=True)
    _ensure_hf_cli()
    export_tool = _ensure_export_tool()
    if not export_tool:
        return False

    ok = _export_model(export_tool, LLM_SOURCE, LLM_MODEL, device)
    ok = _export_model(export_tool, VLM_SOURCE, VLM_MODEL, device) and ok
    if not ok:
        print(_yellow("  Model export needs the OVMS export toolchain. Install with:"))
        print(_yellow('    pip install "optimum-intel[openvino]" nncf'))
        print(_yellow(f"  (also: pip install -r requirements from {os.path.dirname(export_tool)})"))
    return ok


# ── 5. Start / check OVMS ─────────────────────────────────────────────────────

def check_ovms() -> bool:
    try:
        import httpx
        r = httpx.get(f"{OVMS_BASE_URL}/v1/config", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def _both_servables_ready() -> bool:
    try:
        import httpx
        r = httpx.get(f"{OVMS_BASE_URL}/v1/config", timeout=3.0)
        if r.status_code != 200:
            return False
        body = r.text
        return LLM_MODEL in body and VLM_MODEL in body
    except Exception:
        return False


def start_ovms_native(binary: str, device: str) -> bool:
    print(_yellow(f"  [SETUP] Starting OVMS (native: {binary})..."))
    cmd = [
        binary,
        "--config_path", _CONFIG_JSON,
        "--rest_port", str(OVMS_REST_PORT),
        "--target_device", device,
    ]
    log_path = os.path.join(_HERE, "ovms.log")
    log_file = open(log_path, "w")
    print(_yellow(f"  Log: {log_path}"))
    try:
        subprocess.Popen(cmd, cwd=_HERE, stdout=log_file, stderr=log_file)
    except FileNotFoundError:
        print(_red(f"  [FAIL] Could not launch {binary}"))
        return False
    return _wait_for_ovms(log_path)


def start_ovms_docker(device: str) -> bool:
    print(_yellow(f"  [SETUP] Starting OVMS (Docker: {_DOCKER_IMAGE})..."))
    repo_mount = _REPO.replace("\\", "/")
    cmd = ["docker", "run", "-d", "--rm",
           "-p", f"{OVMS_REST_PORT}:{OVMS_REST_PORT}",
           "-v", f"{repo_mount}:/models:rw"]
    if _OS != "Windows":
        # GPU passthrough for Linux Docker (Intel/AMD render nodes)
        if os.path.exists("/dev/dri"):
            cmd += ["--device", "/dev/dri"]
    cmd += [_DOCKER_IMAGE,
            "--config_path", "/models/config.json",
            "--rest_port", str(OVMS_REST_PORT),
            "--target_device", device]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(_red("  [FAIL] docker not found"))
        return False
    if r.returncode != 0:
        print(_red(f"  [FAIL] docker run failed: {r.stderr.strip()[:300]}"))
        return False
    return _wait_for_ovms()


def _wait_for_ovms(log_path: str = "") -> bool:
    """Poll until both servables report ready (model load can take minutes)."""
    max_wait, poll = 600, 5
    for elapsed in range(poll, max_wait + poll, poll):
        time.sleep(poll)
        if _both_servables_ready():
            print(_green(f"\n  [OK] OVMS ready — both models loaded ({elapsed}s)"))
            return True
        sys.stdout.write(f"\r  Waiting for OVMS... {elapsed}s / {max_wait}s "
                         "(first run loads models into device memory)")
        sys.stdout.flush()
    print()
    print(_red(f"  [FAIL] OVMS not ready after {max_wait}s"))
    if log_path:
        print(_yellow(f"  Check {log_path} for errors."))
    return False


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    banner()

    # ── Platform setup ────────────────────────────────────────────
    print(_bold("Platform:"), _OS)
    if _OS == "Linux":
        setup_linux_libs()
        inject_linux_env()
        print(_green("  [OK] Linux environment configured"))
    elif _OS == "Windows":
        print(_green("  [OK] Windows"))
        try:
            import uiautomation  # noqa: F401
            print(_green("  [OK] uiautomation — Stage 0 UIA grounding active"))
        except ImportError:
            print(_yellow("  [..] uiautomation not installed — installing for Stage 0 grounding..."))
            ret = subprocess.run([sys.executable, "-m", "pip", "install", "uiautomation"],
                                 capture_output=True)
            if ret.returncode == 0:
                print(_green("  [OK] uiautomation installed"))
            else:
                print(_yellow("  [WARN] Could not install uiautomation — Stage 0 disabled"))
                print(_yellow("         Run manually: pip install uiautomation"))
    elif _OS == "Darwin":
        print(_green("  [OK] macOS — no extra setup needed"))

    # ── GPU detection ─────────────────────────────────────────────
    check_gpus()
    device = TARGET_DEVICE
    print(_cyan(f"  OVMS target device: {device}"))

    # ── OVMS already running? ─────────────────────────────────────
    print(_bold("\nOpenVINO Model Server:"))
    if check_ovms():
        print(_green(f"  [OK] OVMS already running on {OVMS_BASE_URL}"))
    else:
        # ── Prepare models ────────────────────────────────────────
        print(_bold("\nModels:"))
        if not ensure_models(device):
            print(_red("\n  Could not prepare models. Check the messages above."))
            sys.exit(1)

        # ── Launch OVMS (native preferred, Docker fallback) ───────
        print(_bold("\nStarting server:"))
        binary = find_ovms_binary()
        started = False
        if binary:
            started = start_ovms_native(binary, device)
        elif docker_available():
            print(_yellow("  No native ovms found — using Docker."))
            started = start_ovms_docker(device)
        else:
            print(_red("  [FAIL] Neither a native OVMS binary nor Docker is available."))
            print(_yellow("  Install one of:"))
            print(_yellow("    • OVMS binary — https://docs.openvino.ai/latest/model-server/ovms_docs_deploying_server.html"))
            print(_yellow("      then set OVMS_DIR to its folder (containing ovms.exe), or add it to PATH"))
            print(_yellow("    • Docker     — https://docs.docker.com/get-docker/"))
            sys.exit(1)

        if not started:
            sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    print(_bold("\nStatus:"))
    print(_green("  [OK] OVMS") + f"  → {OVMS_BASE_URL}/v3/chat/completions  (device {device})")
    print(_green("  [OK] LLM ") + f"  → {LLM_MODEL}")
    print(_green("  [OK] VLM ") + f"  → {VLM_MODEL}")

    # ── Launch agent ──────────────────────────────────────────────
    print(_bold("\nStarting Desktop GUI Agent...\n"))
    time.sleep(0.3)
    ret = subprocess.run([sys.executable, os.path.join(_HERE, "main.py")] + sys.argv[1:])
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
