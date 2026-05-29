#!/usr/bin/env python3
"""
Desktop GUI Agent — single entry point.

Run this file to start the agent:
    python start.py

It will:
  1. Check Ollama is running
  2. Check required models are pulled (pulls them if missing)
  3. Set up platform-specific environment (libxcb on Linux, etc.)
  4. Launch the agent UI
"""
import os
import sys
import platform
import subprocess
import time

_OS = platform.system()   # "Linux", "Windows", "Darwin"

# ── Colour helpers (no dependencies) ─────────────────────────────────────────
def _green(s):  return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"

def banner():
    print(_bold("\n╔══════════════════════════════════════════════╗"))
    print(_bold("║       Desktop GUI Agent — Startup Check      ║"))
    print(_bold("╚══════════════════════════════════════════════╝\n"))

# ── 1. Fix libxcb-cursor on Linux (no sudo needed) ───────────────────────────
def setup_linux_libs():
    """Extract libxcb-cursor to ~/.local_xcb if not already done."""
    lib_path = os.path.expanduser("~/.local_xcb/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0")
    if os.path.exists(lib_path):
        return   # already extracted

    print(_yellow("  [SETUP] libxcb-cursor not found — extracting (one-time setup)..."))
    deb_path = "/tmp/libxcb-cursor0.deb"
    try:
        subprocess.run(["apt-get", "download", "libxcb-cursor0"],
                       cwd="/tmp", check=True, capture_output=True)
        dest = os.path.expanduser("~/.local_xcb")
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["dpkg-deb", "-x", deb_path, dest], check=True, capture_output=True)
        print(_green("  [OK] libxcb-cursor extracted"))
    except Exception as e:
        print(_yellow(f"  [WARN] Could not extract libxcb-cursor: {e}"))


def inject_linux_env():
    """Prepend ~/.local_xcb to LD_LIBRARY_PATH so Qt finds libxcb-cursor."""
    lib_dir = os.path.expanduser("~/.local_xcb/usr/lib/x86_64-linux-gnu")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in existing:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir


# ── 2. Check Ollama ───────────────────────────────────────────────────────────
def check_ollama() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama():
    """Try to start Ollama in the background."""
    print(_yellow("  [SETUP] Ollama not running — attempting to start it..."))
    try:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        for _ in range(10):
            time.sleep(1)
            if check_ollama():
                print(_green("  [OK] Ollama started"))
                return True
        print(_red("  [FAIL] Ollama did not start in time. Run 'ollama serve' manually."))
        return False
    except FileNotFoundError:
        print(_red("  [FAIL] 'ollama' not found in PATH. Install from https://ollama.com"))
        return False


# ── 3. Check / pull models ────────────────────────────────────────────────────
REQUIRED_MODELS = {
    "qwen3:14b":       "LLM  — planning, routing, reflection",
    "qwen2.5vl-gui":   "VLM  — visual grounding & verification",
}

def get_pulled_models() -> list:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def ensure_vlm_model(pulled: list) -> bool:
    """Create qwen2.5vl-gui (4096-ctx GPU build) from qwen2.5vl:7b if needed."""
    has_gui = any("qwen2.5vl-gui" in m for m in pulled)
    if has_gui:
        return True

    # Need base model first
    has_base = any("qwen2.5vl" in m for m in pulled)
    if not has_base:
        print(_yellow("  [PULL] Downloading qwen2.5vl:7b (~6 GB) — first run only..."))
        ret = subprocess.run(["ollama", "pull", "qwen2.5vl:7b"]).returncode
        if ret != 0:
            return False

    # Create the 4096-ctx variant
    print(_yellow("  [SETUP] Creating qwen2.5vl-gui model (4096-ctx, fits on GPU)..."))
    modelfile = "FROM qwen2.5vl:7b\nPARAMETER num_ctx 4096\nPARAMETER num_predict 256\n"
    mf_path = "/tmp/qwen2.5vl-gui.Modelfile"
    with open(mf_path, "w") as f:
        f.write(modelfile)
    ret = subprocess.run(["ollama", "create", "qwen2.5vl-gui", "-f", mf_path]).returncode
    return ret == 0


def check_models() -> bool:
    pulled = get_pulled_models()
    all_ok = True

    for model, desc in REQUIRED_MODELS.items():
        base = model.split(":")[0]
        present = any(base in m for m in pulled)

        if present:
            print(_green(f"  [OK] {model:<22} {desc}"))
        else:
            print(_yellow(f"  [..] {model:<22} {desc}  — not found"))
            if model == "qwen3:14b":
                print(_yellow(f"       Downloading {model} (~9 GB) — first run only..."))
                ret = subprocess.run(["ollama", "pull", model]).returncode
                if ret == 0:
                    print(_green(f"  [OK] {model} downloaded"))
                else:
                    print(_red(f"  [FAIL] Could not pull {model}"))
                    all_ok = False
            elif "qwen2.5vl-gui" in model:
                ok = ensure_vlm_model(pulled)
                if ok:
                    print(_green(f"  [OK] qwen2.5vl-gui ready"))
                else:
                    print(_red("  [FAIL] Could not set up VLM model"))
                    all_ok = False

    return all_ok


# ── 4. Main ───────────────────────────────────────────────────────────────────
def main():
    banner()

    # ── Platform setup ────────────────────────────────────────────
    print(_bold("Platform:"), _OS)
    if _OS == "Linux":
        setup_linux_libs()
        inject_linux_env()
        print(_green("  [OK] Linux environment configured"))
    elif _OS == "Windows":
        print(_green("  [OK] Windows — no extra setup needed"))
    elif _OS == "Darwin":
        print(_green("  [OK] macOS — no extra setup needed"))

    # ── Ollama ────────────────────────────────────────────────────
    print(_bold("\nOllama:"))
    if check_ollama():
        print(_green("  [OK] Ollama is running on localhost:11434"))
    else:
        if not start_ollama():
            print(_red("\n  Start Ollama first: ollama serve"))
            sys.exit(1)

    # ── Models ────────────────────────────────────────────────────
    print(_bold("\nModels:"))
    if not check_models():
        print(_red("\n  Some models are missing. Check your connection and retry."))
        sys.exit(1)

    # ── Launch ────────────────────────────────────────────────────
    print(_bold("\nStarting Desktop GUI Agent...\n"))
    time.sleep(0.5)

    here = os.path.dirname(os.path.abspath(__file__))
    ret = subprocess.run([sys.executable, os.path.join(here, "main.py")] + sys.argv[1:])
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
