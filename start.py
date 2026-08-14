#!/usr/bin/env python3
"""Desktop GUI Agent — single entry point (Windows).

    python start.py

This is the whole setup — a fresh clone in an activated venv needs nothing else.
Every step is skipped if it is already done, so it is also the normal way to
start the agent day to day:

  1. Install the runtime packages (requirements.txt) if any are missing
  2. Detect GPU (Intel / AMD / NVIDIA)
  3. Download the native OVMS server into ./ovms/ if the machine has none
  4. Prepare both models in the OpenVINO Model Server (OVMS) repository:
       • LLM  config.LLM_MODEL  (pulled pre-converted from Hugging Face)
       • VLM  config.VLM_MODEL  (converted from UI-TARS on first run)
     Installing the conversion toolchain (requirements-export.txt) first, and
     only when something actually has to be converted.
  5. Start OVMS serving both models on one OpenAI-compatible endpoint (port 8000)
  6. Wait for the server to be ready, then launch the agent UI
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

# The OVMS server itself: a native binary, not a pip package. Downloaded into
# ./ovms/ when the machine has none, so setup needs no manual unzip and no
# OVMS_DIR environment variable.
_OVMS_VERSION = "2026.2"
_OVMS_ZIP_URL = (
    f"https://github.com/openvinotoolkit/model_server/releases/download/"
    f"v{_OVMS_VERSION}/ovms_windows_{_OVMS_VERSION}.0_python_on.zip"
)
_OVMS_LOCAL_DIR = os.path.join(_HERE, "ovms")

_REQS = os.path.join(_HERE, "requirements.txt")
_EXPORT_REQS = os.path.join(_HERE, "requirements-export.txt")
# CPU-only torch index. This is an Intel-GPU target and conversion runs on CPU,
# so the default CUDA build would add ~3.4 GB of NVIDIA wheels for nothing.
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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


# ── 0. Python dependencies ────────────────────────────────────────────────────

def _missing_modules(mods) -> list:
    """Return the subset of `mods` this interpreter cannot import."""
    import importlib.util
    missing = []
    for mod in mods:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except Exception:
            missing.append(mod)  # namespace clash / half-removed package
    return missing


def _pip_install(*args) -> bool:
    """Run pip in THIS interpreter, so packages always land in the venv in use."""
    print(_yellow(f"  [..] pip install {' '.join(args)}"))
    cmd = [sys.executable, "-m", "pip", "install", *args]
    if subprocess.run(cmd).returncode != 0:
        return False
    import importlib
    importlib.invalidate_caches()  # let find_spec see what pip just wrote
    return True


# Import names of the runtime packages, which is what requirements.txt installs.
# Checked by import name because that is what actually fails at run time.
_RUNTIME_IMPORTS = (
    "PyQt6", "PIL", "imagehash", "rapidocr_onnxruntime", "uiautomation",
    "httpx", "pydantic", "numpy", "loguru", "pyperclip", "keyring",
)


def ensure_runtime_deps() -> bool:
    """Install requirements.txt if anything the agent imports is missing."""
    missing = _missing_modules(_RUNTIME_IMPORTS)
    if not missing:
        print(_green("  [OK] runtime packages"))
        return True
    print(_yellow(f"  [..] installing runtime packages ({', '.join(missing)})..."))
    if _pip_install("-r", _REQS) and not _missing_modules(_RUNTIME_IMPORTS):
        print(_green("  [OK] runtime packages installed"))
        return True
    still = _missing_modules(_RUNTIME_IMPORTS)
    print(_red(f"  [FAIL] Still missing: {', '.join(still)}"))
    print(_yellow(f"        Run manually: pip install -r {_REQS}"))
    return False


# Modules requirements-export.txt pulls in, directly or as dependencies of
# optimum-intel[openvino]. If everything missing is on this list, one file
# installs the lot — no need to name packages one at a time and have the user
# rediscover the next one on the following run.
_EXPORT_TOOLCHAIN = {
    "huggingface_hub", "jinja2", "nncf", "numpy", "openvino",
    "openvino_genai", "openvino_tokenizers", "optimum", "torch",
    "transformers",
}


def _export_tool_imports(export_tool: str) -> list:
    """Top-level modules export_model.py imports.

    export_model.py is Intel's script, not ours, and it imports things a
    runtime-only venv does not have (jinja2 is the usual one). Read its imports
    rather than hardcoding a list that drifts when Intel edits the script.
    """
    import ast
    try:
        with open(export_tool, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return []  # Not worth failing over; the real import will speak up.
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return sorted(m for m in mods if m not in sys.builtin_module_names)


def ensure_export_deps(export_tool: str) -> bool:
    """Install the conversion toolchain BEFORE the multi-GB model download.

    Without this, a missing package surfaces as a ModuleNotFoundError traceback
    from Intel's script partway through, which reads like the model is
    unsupported when the real problem is one absent pip package.
    """
    missing = _missing_modules(_export_tool_imports(export_tool))
    if not missing:
        print(_green("  [OK] model-conversion toolchain"))
        return True

    print(_yellow(f"  [..] installing model-conversion toolchain "
                  f"({', '.join(missing)})..."))
    if set(missing) <= _EXPORT_TOOLCHAIN and os.path.isfile(_EXPORT_REQS):
        # requirements.txt deliberately omits the ML stack — the agent talks to
        # OVMS over HTTP and imports no framework. Conversion is the one job
        # that needs it, and requirements-export.txt pins that whole set.
        if "torch" in missing:
            print(_yellow("       (first time: ~2 GB of packages, CPU-only torch)"))
            if not _pip_install("torch", "--index-url", _TORCH_CPU_INDEX):
                print(_red("  [FAIL] Could not install torch"))
                return False
        ok = _pip_install("-r", _EXPORT_REQS)
    else:
        ok = _pip_install(*missing)

    still = _missing_modules(_export_tool_imports(export_tool))
    if ok and not still:
        print(_green("  [OK] model-conversion toolchain installed"))
        return True
    print(_red(f"  [FAIL] Could not install: {', '.join(still or missing)}"))
    print(_yellow(f"        Run manually: pip install -r {_EXPORT_REQS}"))
    return False


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
    total = sum(g.usable_gb for g in gpus)
    for g in gpus:
        if g.shared_mb:
            mem = (f"  {g.usable_gb}GB usable "
                   f"({g.vram_gb}GB dedicated + {g.shared_gb}GB shared)")
        elif g.vram_mb:
            mem = f"  {g.vram_gb}GB VRAM"
        else:
            mem = ""
        print(_green(f"  [{backend.upper()}] GPU{g.index}: {g.name}{mem}"))
    if total:
        print(_green(f"  Total: {len(gpus)} {backend.upper()} GPU(s), {total:.1f}GB usable"))
    return backend, gpus


# ── 2. Locate OVMS (native binary) ────────────────────────────────────────────

def find_ovms_binary() -> str:
    """Return the path to a native ovms executable, or '' if none is found.

    Honours the OVMS_DIR / OVMS_PATH env vars (the OVMS Windows package extracts
    to a folder containing ovms.exe), then PATH, then this project's own ./ovms/
    — the last one is what ensure_ovms_binary() populates, and is why setup does
    not need an environment variable at all.
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
    if found:
        return found
    # Where ensure_ovms_binary() puts it — checked last so an existing install
    # always wins, and found without any environment variable being set.
    for cand in (os.path.join(_OVMS_LOCAL_DIR, exe),
                 os.path.join(_OVMS_LOCAL_DIR, "ovms", exe)):
        if os.path.isfile(cand):
            return cand
    return ""


def _download(url: str, dest: str, label: str) -> bool:
    """Download with a one-line progress readout."""
    import urllib.request
    tmp = dest + ".part"

    def progress(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            sys.stdout.write(f"\r  [..] {label}: {pct}% "
                             f"({done // 1048576} / {total // 1048576} MB)")
        else:
            sys.stdout.write(f"\r  [..] {label}: {done // 1048576} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=progress)
        print()
        os.replace(tmp, dest)
        return True
    except Exception as e:
        print()
        print(_red(f"  [FAIL] Download failed: {e}"))
        if os.path.isfile(tmp):
            os.remove(tmp)  # never leave a truncated file to be reused
        return False


def ensure_ovms_binary() -> str:
    """Return a path to ovms.exe, downloading the server once if needed.

    OVMS ships as a native binary, not a pip package, so nothing in
    requirements.txt can bring it in. Fetching it here is what removes the
    manual download / unzip / `setx OVMS_DIR` steps from setup.
    """
    binary = find_ovms_binary()
    if binary:
        print(_green(f"  [OK] ovms.exe found ({binary})"))
        return binary

    zip_path = os.path.join(_HERE, f"ovms_windows_{_OVMS_VERSION}.zip")
    print(_yellow(f"  [SETUP] OpenVINO Model Server {_OVMS_VERSION} not found "
                  f"— downloading (one-time)..."))
    if not os.path.isfile(zip_path) and not _download(_OVMS_ZIP_URL, zip_path,
                                                      "ovms.zip"):
        print(_yellow(f"        Download it manually from {_OVMS_ZIP_URL},"))
        print(_yellow(f"        extract it so ovms.exe sits in {_OVMS_LOCAL_DIR}"))
        return ""

    print(_yellow("  [..] extracting..."))
    try:
        import zipfile
        # The archive already contains a top-level ovms/ directory, so extract
        # into the project root rather than into _OVMS_LOCAL_DIR itself.
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_HERE)
    except Exception as e:
        print(_red(f"  [FAIL] Could not extract {zip_path}: {e}"))
        return ""

    binary = find_ovms_binary()
    if binary:
        print(_green(f"  [OK] OpenVINO Model Server ready ({binary})"))
        os.remove(zip_path)  # ~2 GB of archive no longer needed
    else:
        print(_red(f"  [FAIL] ovms.exe not found under {_OVMS_LOCAL_DIR} "
                   "after extracting"))
    return binary


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
    if _model_already_exported(LLM_MODEL) and _model_already_exported(VLM_MODEL):
        # Nothing to convert — skip the export tool and its heavy toolchain
        # entirely. A machine that received converted models never needs them.
        for name, cache in ((LLM_MODEL, LLM_KV_CACHE_GB), (VLM_MODEL, VLM_KV_CACHE_GB)):
            print(_green(f"  [OK] {name:<24} already in repository"))
            _ensure_cache_size(name, cache)
        return True

    export_tool = _ensure_export_tool()
    if not export_tool:
        return False
    # Both before the first byte of a multi-GB model is fetched: the toolchain
    # export_model.py imports, then the CLIs it shells out to (which only exist
    # once that toolchain is installed).
    if not ensure_export_deps(export_tool):
        return False
    _ensure_cli_shims()

    ok = _export_model(export_tool, LLM_SOURCE, LLM_MODEL, device,
                       LLM_KV_CACHE_GB, LLM_WEIGHT_FORMAT)
    ok = _export_model(export_tool, VLM_SOURCE, VLM_MODEL, device,
                       VLM_KV_CACHE_GB, VLM_WEIGHT_FORMAT) and ok
    if not ok:
        print(_yellow("  Model export failed. Check the output above for the specific error."))
        print(_yellow("  Common causes:"))
        print(_yellow("    - First-run UI-TARS conversion needs ~16 GB RAM and internet access"))
        print(_yellow("    - Not enough free disk for the converted weights (~15 GB)"))
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
    if sys.prefix == sys.base_prefix:
        print(_yellow("  [WARN] Not running inside a virtual environment — "
                      "packages will install system-wide"))

    # ── Python packages ───────────────────────────────────────────
    # Installed here rather than left to the README: the agent's own imports
    # first, and the conversion toolchain later, only if a model must be built.
    print(_bold("\nDependencies:"))
    if not ensure_runtime_deps():
        sys.exit(1)

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

        # ── Fetch the server BEFORE the models ────────────────────
        # Model preparation can take an hour on first run. Failing to get the
        # server after that wait would be a miserable way to find out.
        binary = ensure_ovms_binary()
        if not binary:
            sys.exit(1)

        # ── Prepare models ────────────────────────────────────────
        print(_bold("\nModels:"))
        if not ensure_models(device):
            print(_red("\n  Could not prepare models. Check the messages above."))
            sys.exit(1)

        # ── Launch OVMS (native binary) ────────────────────────────
        print(_bold("\nStarting server:"))
        if not start_ovms_native(binary, device):
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
