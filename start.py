#!/usr/bin/env python3
"""Desktop GUI Agent — single entry point (Windows).

    python start.py

Does everything automatically:
  1. Check for Windows UIA (Stage 0 grounding)
  2. Detect GPU (Intel / AMD / NVIDIA)
  3. Prepare both models in the OpenVINO Model Server (OVMS) repository:
       • LLM  config.LLM_MODEL  (pulled pre-converted from Hugging Face)
       • VLM  config.VLM_MODEL  (converted from UI-TARS on first run)
  4. Start OVMS serving both models on one OpenAI-compatible endpoint (port 8000)
       using the native ovms.exe binary
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

from config import (
    LLM_KV_CACHE_GB,
    LLM_MODEL,
    LLM_SOURCE,
    LLM_WEIGHT_FORMAT,
    MODEL_REPOSITORY_PATH,
    OVMS_BASE_URL,
    OVMS_REST_PORT,
    TARGET_DEVICE,
    VLM_KV_CACHE_GB,
    VLM_MODEL,
    VLM_SOURCE,
    VLM_WEIGHT_FORMAT,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, MODEL_REPOSITORY_PATH)
_CONFIG_JSON = os.path.join(_REPO, "config.json")

# Pinned OVMS ref used to fetch the model-export helper when it isn't already present.
_OVMS_REF = "releases/2025/3"
_EXPORT_TOOL_URL = (
    f"https://raw.githubusercontent.com/openvinotoolkit/model_server/"
    f"{_OVMS_REF}/demos/common/export_models/export_model.py"
)


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


# ── 1. GPU detection ──────────────────────────────────────────────────────────

def check_gpus():
    """Detect GPUs, print a summary, return (gpu_type, gpus)."""
    from desktop.system import detect_gpus

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


# ── 2. Locate OVMS (native binary) ────────────────────────────────────────────

def find_ovms_binary() -> str:
    """Return the path to a native ovms executable, or '' if none is found.

    Honours the OVMS_DIR / OVMS_PATH env vars (the OVMS Windows package extracts
    to a folder containing ovms.exe), then falls back to PATH.
    """
    exe = "ovms.exe"
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


# ── 3. Model preparation (export into the OVMS repository) ────────────────────

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


# CLIs export_model.py shells out to, as (shim name, distribution, entry point).
# Windows console-script .exe launchers embed the venv's python.exe path at
# install time — copy or move the venv and every one of them dies with
# "The system cannot find the file specified" ('hf.exe' and 'optimum-cli.exe'
# both hit this in live runs). huggingface-cli additionally maps to the `hf`
# entry point because huggingface_hub >= 1.0 ships a huggingface-cli stub that
# just errors out.
_CLI_SHIMS = (
    ("huggingface-cli", "huggingface_hub", "hf"),
    ("optimum-cli", "optimum", "optimum-cli"),
)


def _console_entrypoint(dist: str, name: str) -> str:
    """Return a console_scripts entry point as "module:function", or "".

    Read from package metadata rather than hardcoding module paths so the
    shims keep working across package versions.
    """
    try:
        import importlib.metadata as md
        for ep in md.distribution(dist).entry_points:
            if ep.group == "console_scripts" and ep.name == name:
                return ep.value  # e.g. "huggingface_hub.cli.hf:main"
    except Exception:
        pass
    return ""


def _ensure_cli_shims():
    """Guarantee WORKING CLIs for export_model.py, immune to venv moves.

    Each shim is a .bat that invokes the entry point through the CURRENT
    interpreter (sys.executable, resolved fresh each run). The shim dir is
    PREPENDED to PATH so the shims win over the venv's broken .exe launchers.
    """
    shim_dir = os.path.join(_HERE, "tools", "ovms", "_shims")
    os.makedirs(shim_dir, exist_ok=True)
    active = []
    for shim_name, dist, ep_name in _CLI_SHIMS:
        entrypoint = _console_entrypoint(dist, ep_name)
        if not entrypoint:
            continue  # package absent — leave that CLI alone
        module, _, func = entrypoint.partition(":")
        # `python -c "import sys; from <mod> import <fn>; sys.exit(<fn>())" %*`
        runner = f"import sys; from {module} import {func}; sys.exit({func}())"
        try:
            with open(os.path.join(shim_dir, f"{shim_name}.bat"), "w") as f:
                f.write(f'@echo off\r\n"{sys.executable}" -c "{runner}" %*\r\n')
            active.append(shim_name)
        except Exception as e:
            print(_yellow(f"  [WARN] Could not create {shim_name} shim: {e}"))
    if not active:
        return
    if shim_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = shim_dir + os.pathsep + os.environ.get("PATH", "")
    print(_green(f"  [OK] CLI shims active: {', '.join(active)}"))


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


def _prune_stale_servables():
    """Unregister servables that config.py no longer names.

    OVMS loads EVERY entry in the repository config.json into device memory.
    After a model swap (e.g. qwen3-8b → qwen3-14b) the old entry would still
    be loaded alongside the new one and overflow VRAM. Remove stale entries
    from config.json only — the exported weights stay on disk, so switching
    back is instant.
    """
    if not os.path.isfile(_CONFIG_JSON):
        return
    keep = {LLM_MODEL, VLM_MODEL}
    try:
        with open(_CONFIG_JSON) as f:
            cfg = json.load(f)
        removed = []
        for key, name_of in (
            ("mediapipe_config_list", lambda e: e.get("name")),
            ("model_config_list", lambda e: e.get("config", {}).get("name")),
        ):
            entries = cfg.get(key, [])
            kept = [e for e in entries if name_of(e) in keep]
            removed += [name_of(e) for e in entries if name_of(e) not in keep]
            if key in cfg:
                cfg[key] = kept
        if removed:
            with open(_CONFIG_JSON, "w") as f:
                json.dump(cfg, f, indent=4)
            for name in removed:
                print(_yellow(f"  [OK] {name}: unregistered (weights kept on disk)"))
    except Exception as e:
        print(_yellow(f"  [WARN] Could not prune stale servables: {e}"))


def _ensure_cache_size(model_name: str, cache_gb: int):
    """Sync an already-exported servable's KV-cache budget with config.py.

    export_model.py bakes `cache_size: N` into the servable's graph.pbtxt at
    export time, and exports are skipped once the model is in the repository —
    so a later config change would silently never apply. Patch the text
    protobuf in place instead of forcing a multi-GB re-export.
    """
    import re
    graph = os.path.join(_REPO, model_name, "graph.pbtxt")
    if not os.path.isfile(graph):
        return
    try:
        with open(graph) as f:
            text = f.read()
        patched, n = re.subn(r"cache_size:\s*\d+", f"cache_size: {cache_gb}", text)
        if n and patched != text:
            with open(graph, "w") as f:
                f.write(patched)
            print(_green(f"  [OK] {model_name:<24} KV cache updated to {cache_gb} GB"))
    except Exception as e:
        print(_yellow(f"  [WARN] Could not update KV cache for {model_name}: {e}"))


def _export_model(export_tool: str, source_model: str, model_name: str,
                  device: str, cache_gb: int, weight_format: str) -> bool:
    """Run export_model.py to convert/pull a model into the OVMS repository.

    The `text_generation` subcommand handles both plain LLMs and vision-language
    models — it runs optimum-cli for non-prebuilt sources (e.g. UI-TARS), writes
    a graph.pbtxt, and appends the servable to config.json.

    weight_format is the export precision (int4/int8/fp16). For a source that is
    already a converted IR (the pre-quantized LLM repo) it is a consistency no-op;
    for the upstream UI-TARS checkpoint it selects the VLM's quantization.
    """
    if _model_already_exported(model_name):
        print(_green(f"  [OK] {model_name:<24} already in repository"))
        _ensure_cache_size(model_name, cache_gb)
        return True

    print(_yellow(f"  [..] {model_name:<24} preparing from {source_model} "
                  f"({weight_format}, first run is slow)..."))
    cmd = [
        sys.executable, export_tool, "text_generation",
        "--source_model", source_model,
        "--model_name", model_name,
        "--weight-format", weight_format,
        "--config_file_path", _CONFIG_JSON,
        "--model_repository_path", _REPO,
        "--target_device", device,
        "--cache_size", str(cache_gb),
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
    _prune_stale_servables()
    _ensure_cli_shims()
    export_tool = _ensure_export_tool()
    if not export_tool:
        return False

    ok = _export_model(export_tool, LLM_SOURCE, LLM_MODEL, device,
                       LLM_KV_CACHE_GB, LLM_WEIGHT_FORMAT)
    ok = _export_model(export_tool, VLM_SOURCE, VLM_MODEL, device,
                       VLM_KV_CACHE_GB, VLM_WEIGHT_FORMAT) and ok
    if not ok:
        print(_yellow("  Model export failed. Check the output above for the specific error."))
        print(_yellow("  Common causes:"))
        print(_yellow("    - First-run UI-TARS conversion needs ~16 GB RAM and internet access"))
        print(_yellow("    - Missing toolchain: pip install -r requirements-export.txt"))
        print(_yellow("  On Windows: if model files have 'Access denied', delete the model folder"))
        print(_yellow("  from an elevated terminal and re-run this script."))
    return ok


# ── 4. Start / check OVMS ─────────────────────────────────────────────────────

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
    # NOTE: --target_device is a per-model parameter and is mutually exclusive
    # with --config_path ("Model parameters in CLI are exclusive with the config
    # file"). The device was baked into each servable's graph.pbtxt at export
    # time, so it must NOT be repeated here.
    ovms_args = [
        "--config_path", _CONFIG_JSON,
        "--rest_port", str(OVMS_REST_PORT),
    ]
    # ovms.exe needs the env from setupvars (its DLLs + bundled Python on PATH).
    # Running setupvars in OUR shell would hijack the venv's Python (PYTHONHOME),
    # so we run it only inside the ovms subprocess. We write a one-shot launcher
    # .bat (avoids the cmd /c inline-quoting trap around the setupvars path).
    ovms_dir = os.path.dirname(binary)
    setupvars = os.path.join(ovms_dir, "setupvars.bat")
    inner = subprocess.list2cmdline([binary] + ovms_args)
    lines = ["@echo off"]
    if os.path.isfile(setupvars):
        print(_green(f"  [OK] Sourcing {setupvars} for the OVMS process"))
        lines.append(f'call "{setupvars}"')
    else:
        print(_yellow(f"  [WARN] setupvars.bat not found next to {binary}; "
                      "starting ovms.exe directly (may fail to find its DLLs)"))
    lines.append(inner)
    launcher = os.path.join(_HERE, "_run_ovms.bat")
    with open(launcher, "w") as f:
        f.write("\r\n".join(lines) + "\r\n")
    cmd = ["cmd", "/c", launcher]
    log_path = os.path.join(_HERE, "ovms.log")
    log_file = open(log_path, "w")
    print(_yellow(f"  Log: {log_path}"))
    try:
        subprocess.Popen(cmd, cwd=_HERE, stdout=log_file, stderr=log_file)
    except FileNotFoundError:
        print(_red(f"  [FAIL] Could not launch {binary}"))
        return False
    return _wait_for_ovms(log_path)


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


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    banner()

    # ── Platform check ────────────────────────────────────────────
    if platform.system() != "Windows":
        print(_red(f"  [FAIL] This agent supports Windows only (detected: {platform.system()})"))
        sys.exit(1)
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

    # ── GPU detection ─────────────────────────────────────────────
    check_gpus()
    device = TARGET_DEVICE
    print(_cyan(f"  OVMS target device: {device}"))

    # ── OVMS already running? ─────────────────────────────────────
    print(_bold("\nOpenVINO Model Server:"))
    if check_ovms() and _both_servables_ready():
        print(_green(f"  [OK] OVMS already running on {OVMS_BASE_URL}"))
    else:
        if check_ovms():
            # Server is up but serving a stale model set (e.g. after a model
            # swap in config.py) — restart it so it reloads config.json.
            print(_yellow(f"  [..] OVMS is running but not serving "
                          f"{LLM_MODEL} + {VLM_MODEL} — restarting it"))
            subprocess.run(["taskkill", "/F", "/IM", "ovms.exe"],
                           capture_output=True, timeout=10)
            time.sleep(2.0)
        # ── Prepare models ────────────────────────────────────────
        print(_bold("\nModels:"))
        if not ensure_models(device):
            print(_red("\n  Could not prepare models. Check the messages above."))
            sys.exit(1)

        # ── Launch OVMS (native binary) ────────────────────────────
        print(_bold("\nStarting server:"))
        binary = find_ovms_binary()
        started = False
        if binary:
            started = start_ovms_native(binary, device)
        else:
            print(_red("  [FAIL] No native OVMS binary found."))
            print(_yellow("  Install OVMS — https://docs.openvino.ai/latest/model-server/ovms_docs_deploying_server.html"))
            print(_yellow("  then set OVMS_DIR to its folder (containing ovms.exe), or add it to PATH"))
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
