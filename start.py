#!/usr/bin/env python3
"""
Desktop GUI Agent — single entry point.

    python start.py

Does everything automatically:
  1. Detect GPUs (AMD ROCm / NVIDIA CUDA)
  2. Start Ollama with the correct GPU env var (if not already running)
  3. Start vLLM  with the correct GPU env var (if installed and not running)
  4. Pull any missing Ollama models
  5. Wait for both services to be ready
  6. Launch the agent UI
"""
import os
import sys
import platform
import subprocess
import time

_OS = platform.system()

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

def _detect_amd_gpus():
    gpus = []
    try:
        rn = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        if rn.returncode != 0:
            return []
        name_lines = [l.strip() for l in rn.stdout.splitlines()
                      if l.strip() and not l.lower().startswith("device")]
        for i, line in enumerate(name_lines):
            parts = line.split(",")
            name = parts[-1].strip() if parts else f"AMD GPU {i}"
            gpus.append({"index": i, "name": name, "vram_gb": 0})

        rv = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        if rv.returncode == 0:
            vram_lines = [l.strip() for l in rv.stdout.splitlines()
                          if l.strip() and not l.lower().startswith("device")]
            for i, vl in enumerate(vram_lines):
                if i < len(gpus):
                    try:
                        vram_bytes = int(vl.split(",")[-1].strip())
                        gpus[i]["vram_gb"] = round(vram_bytes / (1024 ** 3), 1)
                    except (ValueError, IndexError):
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _detect_nvidia_gpus():
    gpus = []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_gb": round(int(parts[2]) / 1024, 1),
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def check_gpus():
    """Detect GPUs, print a summary, return (gpu_type, gpus)."""
    print(_bold("\nGPU Detection:"))
    amd = _detect_amd_gpus()
    if amd:
        total = sum(g["vram_gb"] for g in amd)
        for g in amd:
            print(_green(f"  [AMD] GPU{g['index']}: {g['name']}  {g['vram_gb']}GB VRAM"))
        print(_green(f"  Total: {len(amd)} AMD GPU(s), {total:.1f}GB VRAM"))
        return "amd", amd

    nvidia = _detect_nvidia_gpus()
    if nvidia:
        total = sum(g["vram_gb"] for g in nvidia)
        for g in nvidia:
            print(_green(f"  [NVIDIA] GPU{g['index']}: {g['name']}  {g['vram_gb']}GB VRAM"))
        print(_green(f"  Total: {len(nvidia)} NVIDIA GPU(s), {total:.1f}GB VRAM"))
        return "nvidia", nvidia

    print(_yellow("  No GPU detected — running in CPU-only mode (slow)"))
    return "cpu", []


# ── 3. Ollama ─────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama(gpu_type: str, gpu_devices: str):
    """
    Launch `ollama serve` in the background with the correct GPU env var.
    Waits up to 15 s for it to become responsive.
    """
    print(_yellow("  [SETUP] Starting Ollama..."))
    env = os.environ.copy()

    if gpu_type == "amd" and gpu_devices:
        env["ROCR_VISIBLE_DEVICES"] = gpu_devices
        print(_cyan(f"  ROCR_VISIBLE_DEVICES={gpu_devices}"))
    elif gpu_type == "nvidia" and gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        print(_cyan(f"  CUDA_VISIBLE_DEVICES={gpu_devices}"))

    try:
        subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print(_red("  [FAIL] 'ollama' not found — install from https://ollama.com"))
        return False

    for i in range(15):
        time.sleep(1)
        if check_ollama():
            print(_green(f"  [OK] Ollama ready ({i + 1}s)"))
            return True
        sys.stdout.write(f"\r  Waiting for Ollama... {i + 1}s")
        sys.stdout.flush()
    print()
    print(_red("  [FAIL] Ollama did not start in 15 s — run 'ollama serve' manually"))
    return False


# ── 4. vLLM ──────────────────────────────────────────────────────────────────

def _vllm_installed() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("vllm") is not None
    except Exception:
        return False


def check_vllm() -> bool:
    try:
        import httpx
        from config import VLM_BASE_URL
        r = httpx.get(f"{VLM_BASE_URL}/v1/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def start_vllm(gpu_type: str, gpu_devices: str) -> bool:
    """
    Launch vLLM in the background and wait up to 5 minutes for it to be ready.
    First run downloads the model (~16 GB) — that can take longer; we wait up to
    10 minutes in that case.  Returns True if vLLM becomes responsive in time.
    """
    from config import (VLM_VLLM, VLM_BASE_URL, VLM_DTYPE,
                        VLM_GPU_MEMORY_UTIL, VLM_MAX_MODEL_LEN,
                        TENSOR_PARALLEL_SIZE)

    port = VLM_BASE_URL.split(":")[-1]
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", VLM_VLLM,
        "--port", port,
        "--dtype", VLM_DTYPE,
        "--max-model-len", str(VLM_MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(VLM_GPU_MEMORY_UTIL),
    ]
    if TENSOR_PARALLEL_SIZE > 1:
        cmd += ["--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE)]

    env = os.environ.copy()
    if gpu_type == "amd" and gpu_devices:
        env["HIP_VISIBLE_DEVICES"] = gpu_devices
        print(_cyan(f"  HIP_VISIBLE_DEVICES={gpu_devices}"))
    elif gpu_type == "nvidia" and gpu_devices:
        env["CUDA_VISIBLE_DEVICES"] = gpu_devices
        print(_cyan(f"  CUDA_VISIBLE_DEVICES={gpu_devices}"))

    # Write a log file so the user can tail it if needed
    here = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(here, "vllm.log")
    log_file = open(log_path, "w")

    print(_yellow(f"  [SETUP] Starting vLLM ({VLM_VLLM})..."))
    print(_yellow(f"  Log: {log_path}  (tail -f vllm.log to watch progress)"))

    try:
        subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    except FileNotFoundError:
        print(_red("  [FAIL] vllm not found — pip install vllm"))
        return False

    # Poll until ready — allow up to 10 min (model may download on first run)
    max_wait = 600
    poll = 5
    for i in range(0, max_wait, poll):
        time.sleep(poll)
        if check_vllm():
            print(_green(f"\n  [OK] vLLM ready ({i + poll}s)"))
            return True
        elapsed = i + poll
        sys.stdout.write(
            f"\r  Waiting for vLLM... {elapsed}s / {max_wait}s  "
            f"(first run downloads ~16 GB)"
        )
        sys.stdout.flush()

    print()
    print(_yellow(f"  [WARN] vLLM not ready after {max_wait}s"))
    print(_yellow("  The agent will start using the Ollama VLM fallback."))
    print(_yellow(f"  Check {log_path} for errors."))
    return False


# ── 5. Model checks ───────────────────────────────────────────────────────────
from config import LLM_MODEL, VLM_OLLAMA

UITARS_GGUF_NAME = "ui-tars-1.5-7b-q4_k_m.gguf"


def get_pulled_models() -> list:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ensure_vlm_model(pulled: list) -> bool:
    """Register ui-tars-1.5-7b-gui with Ollama from a local GGUF file."""
    if any("ui-tars-1.5-7b-gui" in m for m in pulled):
        return True
    here = os.path.dirname(os.path.abspath(__file__))
    gguf_path = os.path.join(here, UITARS_GGUF_NAME)
    if not os.path.exists(gguf_path):
        print(_yellow(f"  [INFO] UI-TARS GGUF not found — to use it via Ollama:"))
        print(_yellow(f"    huggingface-cli download Mungert/UI-TARS-1.5-7B \\"))
        print(_yellow(f"        --include '*.gguf' --local-dir . --local-dir-use-symlinks False"))
        print(_yellow(f"    mv *Q4_K_M*.gguf {UITARS_GGUF_NAME}"))
        return False
    print(_yellow("  [SETUP] Registering ui-tars-1.5-7b-gui with Ollama..."))
    mf_path = "/tmp/ui-tars-1.5-7b-gui.Modelfile" if _OS != "Windows" else \
              os.path.join(os.environ.get("TEMP", "C:\\Temp"), "ui-tars-1.5-7b-gui.Modelfile")
    with open(mf_path, "w") as f:
        f.write(f"FROM {gguf_path}\nPARAMETER num_ctx 4096\nPARAMETER num_predict 256\n")
    ret = subprocess.run(["ollama", "create", "ui-tars-1.5-7b-gui", "-f", mf_path]).returncode
    if ret == 0:
        print(_green("  [OK] ui-tars-1.5-7b-gui registered"))
    else:
        print(_red("  [FAIL] ollama create failed"))
    return ret == 0


def check_models() -> bool:
    pulled = get_pulled_models()
    all_ok = True

    llm_base = LLM_MODEL.split(":")[0]
    if any(llm_base in m for m in pulled):
        print(_green(f"  [OK] {LLM_MODEL:<30} LLM — planning, routing, reflection"))
    else:
        print(_yellow(f"  [..] {LLM_MODEL:<30} not found — downloading (~9 GB)..."))
        ret = subprocess.run(["ollama", "pull", LLM_MODEL]).returncode
        if ret == 0:
            print(_green(f"  [OK] {LLM_MODEL} downloaded"))
        else:
            print(_red(f"  [FAIL] Could not pull {LLM_MODEL}"))
            all_ok = False

    # vLLM is already handling UI-TARS if it's running — skip Ollama VLM in that case
    if check_vllm():
        print(_green(f"  [OK] VLM served by vLLM (UI-TARS)"))
        return all_ok

    uitars_ok = any("ui-tars-1.5-7b-gui" in m for m in pulled)
    vlm_fb_base = VLM_OLLAMA.split(":")[0]
    fallback_ok = any(vlm_fb_base in m for m in pulled)

    if uitars_ok:
        print(_green(f"  [OK] ui-tars-1.5-7b-gui               VLM primary (Ollama/GGUF)"))
    else:
        uitars_ok = ensure_vlm_model(pulled)

    if not uitars_ok:
        if fallback_ok:
            print(_green(f"  [OK] {VLM_OLLAMA:<30} VLM fallback (Ollama)"))
        else:
            print(_yellow(f"  [..] {VLM_OLLAMA:<30} not found — downloading..."))
            ret = subprocess.run(["ollama", "pull", VLM_OLLAMA]).returncode
            if ret == 0:
                print(_green(f"  [OK] {VLM_OLLAMA} downloaded"))
                fallback_ok = True
            else:
                print(_red(f"  [FAIL] Could not pull {VLM_OLLAMA}"))

        if not fallback_ok:
            print(_red("  [FAIL] No VLM available — grounding will not work."))
            all_ok = False

    return all_ok


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
        # Check for Windows UIAutomation (Stage 0 grounding — big accuracy boost)
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
    gpu_type, gpus = check_gpus()

    from config import OLLAMA_GPU_DEVICES, VLLM_GPU_DEVICES

    # ── Ollama ────────────────────────────────────────────────────
    print(_bold("\nOllama (LLM):"))
    if check_ollama():
        print(_green("  [OK] Ollama already running on localhost:11434"))
    else:
        if not start_ollama(gpu_type, OLLAMA_GPU_DEVICES):
            sys.exit(1)

    # ── vLLM ──────────────────────────────────────────────────────
    print(_bold("\nvLLM (VLM — UI-TARS):"))
    if check_vllm():
        print(_green("  [OK] vLLM already running — UI-TARS active"))
    elif not gpus:
        print(_yellow("  No GPU found — skipping vLLM, using Ollama VLM fallback"))
    elif not _vllm_installed():
        print(_yellow("  vLLM not installed — using Ollama VLM fallback"))
        print(_yellow("  To enable: pip install vllm"))
    else:
        start_vllm(gpu_type, VLLM_GPU_DEVICES)

    # ── Models ────────────────────────────────────────────────────
    print(_bold("\nModels:"))
    if not check_models():
        print(_red("\n  Some required models are missing. Check the messages above."))
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────
    print(_bold("\nStatus:"))
    print(_green("  [OK] Ollama") + f"  → localhost:11434  ({LLM_MODEL})")
    vlm_status = "vLLM (UI-TARS)" if check_vllm() else f"Ollama fallback ({VLM_OLLAMA})"
    print(_green("  [OK] VLM   ") + f"  → {vlm_status}")

    # ── Launch agent ──────────────────────────────────────────────
    print(_bold("\nStarting Desktop GUI Agent...\n"))
    time.sleep(0.3)
    here = os.path.dirname(os.path.abspath(__file__))
    ret = subprocess.run([sys.executable, os.path.join(here, "main.py")] + sys.argv[1:])
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
