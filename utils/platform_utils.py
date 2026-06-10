# utils/platform_utils.py
"""Shared platform detection utilities — imported by router, planner, and start."""
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List

_OS = platform.system()


# ── Firefox ───────────────────────────────────────────────────────────────────

def detect_firefox() -> str:
    """Return the best available Firefox launch command for this OS."""
    if _OS == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\firefox.exe",
            ) as k:
                path = winreg.QueryValue(k, None)
                if path and os.path.exists(path):
                    return f'"{path}"'
        except Exception:
            pass
        for path in [
            os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Mozilla Firefox\firefox.exe"),
        ]:
            if os.path.exists(path):
                return f'"{path}"'
        return "firefox"

    which = shutil.which("firefox")
    if which:
        return which
    for path in [
        os.path.expanduser("~/apps/firefox/firefox/firefox"),
        os.path.expanduser("~/firefox/firefox"),
        "/snap/bin/firefox",
        "/usr/bin/firefox",
        "/usr/local/bin/firefox",
        "/opt/firefox/firefox",
    ]:
        if os.path.exists(path):
            return path
    return "firefox"


# ── Desktop path ──────────────────────────────────────────────────────────────

def get_desktop_path() -> str:
    """Return the REAL Desktop folder path for this machine.

    On Windows the Desktop is frequently redirected by OneDrive
    (C:\\Users\\<u>\\OneDrive\\Desktop), so '%USERPROFILE%\\Desktop' and
    '$env:USERPROFILE\\Desktop' point at a directory that does not exist.
    Ask the shell for the actual known-folder location instead, and bake the
    resolved LITERAL path into prompts — it works in any shell, no expansion.
    """
    if _OS == "Windows":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            # CSIDL_DESKTOPDIRECTORY = 0x0010 — follows OneDrive redirection
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x0010, None, 0, buf) == 0:
                if buf.value and os.path.isdir(buf.value):
                    return buf.value
        except Exception:
            pass
        fallback = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
        if os.path.isdir(fallback):
            return fallback
        onedrive = os.path.join(os.environ.get("OneDrive", ""), "Desktop")
        if os.path.isdir(onedrive):
            return onedrive
        return fallback or "~\\Desktop"
    return os.path.expanduser("~/Desktop")


# ── GPU detection ─────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_mb: int
    backend: str   # "amd" | "nvidia"

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024, 1)


def detect_gpus() -> List[GPUInfo]:
    """Detect all available GPUs. Tries AMD ROCm first, then NVIDIA CUDA."""
    gpus: List[GPUInfo] = []

    # ── AMD ROCm ──────────────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname", "--csv"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.splitlines() if l.strip() and not l.startswith("device")]
            for i, line in enumerate(lines):
                parts = line.split(",")
                name = parts[-1].strip() if parts else f"AMD GPU {i}"
                gpus.append(GPUInfo(index=i, name=name, vram_mb=0, backend="amd"))

            # Fill VRAM from a separate call
            rv = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=5,
            )
            if rv.returncode == 0:
                vram_lines = [l.strip() for l in rv.stdout.splitlines()
                              if l.strip() and not l.startswith("device")]
                for i, vl in enumerate(vram_lines):
                    if i < len(gpus):
                        parts = vl.split(",")
                        try:
                            # rocm-smi reports VRAM in bytes
                            gpus[i].vram_mb = int(parts[-1].strip()) // (1024 * 1024)
                        except (ValueError, IndexError):
                            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if gpus:
        return gpus

    # ── NVIDIA CUDA ───────────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append(GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_mb=int(parts[2]),
                        backend="nvidia",
                    ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return gpus


def gpu_summary(gpus: List[GPUInfo]) -> str:
    """One-line summary of detected GPUs."""
    if not gpus:
        return "No GPUs detected (CPU-only mode)"
    total_vram = sum(g.vram_mb for g in gpus)
    names = ", ".join(f"GPU{g.index} {g.name} ({g.vram_gb}GB)" for g in gpus)
    return f"{len(gpus)}× {gpus[0].backend.upper()} — {names} — total {round(total_vram/1024,1)}GB VRAM"
