# utils/platform_utils.py
"""Windows platform detection utilities — imported by router, planner, and start."""
import os
import subprocess
from dataclasses import dataclass

# ── DPI awareness ─────────────────────────────────────────────────────────────

def enable_dpi_awareness() -> None:
    """Make this process DPI-aware so every coordinate space agrees.

    Without this, on displays scaled above 100% Windows virtualizes coordinates
    for the process: UIA reports physical pixels while SetCursorPos and GDI
    screenshots use scaled logical pixels — every grounded click lands off by
    the scale factor. Declaring Per-Monitor-V2 awareness (with graceful
    fallbacks for older Windows) puts screenshots, UIA rectangles, and injected
    mouse input in one physical-pixel space at ANY display scale.
    Must run before the first capture or Qt initialisation. No-op off Windows.
    """
    try:
        import ctypes
        try:
            # Per-Monitor V2 (Windows 10 1703+): -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor (Win 8.1+)
            return
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()            # system-aware (Vista+)
    except Exception:
        pass   # non-Windows or restricted environment — nothing to do


# ── Installed apps ────────────────────────────────────────────────────────────

_installed_apps_cache: list[str] | None = None


def installed_apps() -> list[str]:
    """Names of every launchable app on this machine, from the OS itself.

    Get-StartApps lists the Start-menu catalogue — classic Win32 apps AND
    packaged/Store apps (e.g. the new Outlook), which have no .lnk on disk.
    This is planning ground truth: it lets the router prefer an app the user
    actually has over a web fallback, without hardcoding a single app name.
    Cached for the process lifetime (~1-3 s first call). Returns [] off
    Windows or on failure — callers must treat the list as best-effort.
    """
    global _installed_apps_cache
    if _installed_apps_cache is not None:
        return _installed_apps_cache
    apps: list[str] = []
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-StartApps | ForEach-Object Name"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            apps = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    _installed_apps_cache = apps
    return apps


# ── Desktop path ──────────────────────────────────────────────────────────────

def get_desktop_path() -> str:
    """Return the REAL Desktop folder path for this machine.

    The Desktop is frequently redirected by OneDrive
    (C:\\Users\\<u>\\OneDrive\\Desktop), so '%USERPROFILE%\\Desktop' and
    '$env:USERPROFILE\\Desktop' point at a directory that does not exist.
    Ask the shell for the actual known-folder location instead, and bake the
    resolved LITERAL path into prompts — it works in any shell, no expansion.
    """
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


# ── GPU detection ─────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_mb: int
    backend: str   # "nvidia" | "intel"

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024, 1)


def detect_gpus() -> list[GPUInfo]:
    """Detect available GPUs for the startup banner: NVIDIA first, then Intel."""
    gpus: list[GPUInfo] = []

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

    if gpus:
        return gpus

    # ── Intel (iGPU / Arc) — used by OpenVINO Model Server with --target_device GPU
    gpus.extend(_detect_intel_gpus())

    return gpus


def _detect_intel_gpus() -> list[GPUInfo]:
    """Best-effort Intel GPU detection for the startup banner.

    OVMS selects the device itself via --target_device GPU, so this is purely
    informational and never required for inference to work.
    """
    gpus: list[GPUInfo] = []
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Where-Object { $_.Name -match 'Intel' } | "
             "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0:
            for i, line in enumerate(ln.strip() for ln in r.stdout.splitlines() if ln.strip()):
                name, _, ram = line.partition("|")
                try:
                    # AdapterRAM is bytes (and unreliable/0 for shared-memory iGPUs)
                    vram_mb = max(int(ram), 0) // (1024 * 1024)
                except ValueError:
                    vram_mb = 0
                gpus.append(GPUInfo(index=i, name=name.strip() or f"Intel GPU {i}",
                                    vram_mb=vram_mb, backend="intel"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus

