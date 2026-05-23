# scripts/setup/verify_setup.py
"""Run this to verify all dependencies are correctly installed."""
import sys
import subprocess

REQUIRED = [
    ("openvino", "OpenVINO"),
    ("PyQt6", "PyQt6"),
    ("pyautogui", "pyautogui"),
    ("mss", "mss"),
    ("imagehash", "imagehash"),
    ("fastapi", "FastAPI"),
    ("httpx", "httpx"),
    ("pydantic", "Pydantic"),
    ("loguru", "loguru"),
    ("PIL", "Pillow"),
    ("sentence_transformers", "sentence-transformers"),
    ("psutil", "psutil"),
    ("yaml", "pyyaml"),
]

print("=== Desktop GUI Agent — Setup Verification ===\n")
all_ok = True

for module, name in REQUIRED:
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        all_ok = False

print()

# Check Docker
try:
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    print(f"  [OK] Docker: {result.stdout.strip()}")
except FileNotFoundError:
    print("  [FAIL] Docker not found — install docker.io")
    all_ok = False

# Check X11
import os
session = os.environ.get("XDG_SESSION_TYPE", "unknown")
display = os.environ.get("DISPLAY", "not set")
if session == "wayland":
    print(f"\n  [WARN] Wayland detected! pyautogui will NOT work.")
    print(f"         Add 'export XDG_SESSION_TYPE=x11' to ~/.bashrc")
    all_ok = False
else:
    print(f"  [OK] Session: {session}, DISPLAY={display}")

# Test screen capture
try:
    import mss
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = sct.grab(monitor)
    print(f"  [OK] Screen capture: {img.width}x{img.height}")
except Exception as e:
    print(f"  [FAIL] Screen capture: {e}")
    all_ok = False

# Test pyautogui
try:
    import pyautogui
    w, h = pyautogui.size()
    print(f"  [OK] pyautogui screen size: {w}x{h}")
except Exception as e:
    print(f"  [FAIL] pyautogui: {e}")
    all_ok = False

print()
if all_ok:
    print("ALL CHECKS PASSED — ready to start building.")
else:
    print("SOME CHECKS FAILED — fix the above before continuing.")
    sys.exit(1)