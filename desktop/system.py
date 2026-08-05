# desktop/system.py
"""Windows facts: DPI, windows, processes, installed apps, GPUs.

Everything here answers a question about the machine — never what to do about
the answer. Policy built on these facts lives in core/ (groundtruth.py decides
whether a launch counted; anchor.py decides which window the task owns).
Every function is best-effort and returns a safe empty value off Windows or on
failure, so callers never have to guard the platform themselves.
"""
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


# ── Windows and processes ─────────────────────────────────────────────────────
# Ground truth for "what is on screen and who owns it". The orchestrator uses
# these to prove a launch happened, to keep the task inside its own app, and to
# refuse clicks that would land in another process.

_MIN_REAL_WINDOW = (200, 120)   # smaller than this is a tray/tooltip, not a window


def _exe_of_pid(pid: int) -> str:
    """Executable file name (no path) of a process id, "" when unavailable."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(260)
            size = ctypes.c_ulong(260)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1]
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        pass
    return ""


def foreground_app() -> tuple[int, int, str]:
    """(hwnd, pid, exe_name) of the current foreground window; zeros on failure."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        if not hwnd:
            return 0, 0, ""
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return hwnd, pid.value, _exe_of_pid(pid.value)
    except Exception:
        return 0, 0, ""


def window_title(hwnd: int) -> str:
    """Title text of a window handle, "" when it cannot be read."""
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value or ""
    except Exception:
        return ""


def window_owner_at_point(x: int, y: int) -> tuple[int, int, str]:
    """(root_hwnd, pid, exe_name) of the window that owns screen pixel (x, y).

    This is what a click there would actually hit — the OS's own answer, read
    before any input fires.
    """
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        user32.WindowFromPoint.argtypes = [ctypes.wintypes.POINT]
        user32.WindowFromPoint.restype = ctypes.wintypes.HWND
        h = user32.WindowFromPoint(ctypes.wintypes.POINT(int(x), int(y)))
        if not h:
            return 0, 0, ""
        root = user32.GetAncestor(h, 2) or h   # 2 = GA_ROOT
        pid = ctypes.wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
        if not pid.value:
            return int(root), 0, ""
        return int(root), pid.value, _exe_of_pid(pid.value)
    except Exception:
        return 0, 0, ""


def is_process_running(exe_name: str) -> bool:
    """True when a process with this executable name is in the task list."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/NH"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        return exe_name.lower() in out.lower()
    except Exception:
        return False


def count_process_windows(exe_name: str) -> int:
    """Count visible, non-trivial top-level windows owned by `exe_name`.

    Used for launch confirmation (explorer.exe is always running, so only a
    window proves anything) and for new-window verification when the app was
    already running: focusing the old window keeps the count flat, a real
    launch raises it.
    """
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        target = exe_name.lower()
        count = [0]
        min_w, min_h = _MIN_REAL_WINDOW

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def _cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if (rect.right - rect.left) < min_w or (rect.bottom - rect.top) < min_h:
                return True
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if _exe_of_pid(pid.value).lower() == target:
                count[0] += 1
            return True

        user32.EnumWindows(_cb, 0)
        return count[0]
    except Exception:
        return 0


def process_has_visible_window(exe_name: str) -> bool:
    """True when `exe_name` owns at least one visible, non-trivial window."""
    return count_process_windows(exe_name) > 0


# Foreground processes where a click or a re-typed command cannot help: command
# errors need corrected text, not mouse input.
TERMINAL_PROCESSES = frozenset({
    "windowsterminal.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
    "conhost.exe", "openconsole.exe",
})


def foreground_is_terminal() -> bool:
    """True when a console window owns the foreground. False under pytest —
    the test process itself runs inside a terminal.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        from desktop.snapshot import (
            _get_foreground_hwnd_and_title,
            _get_foreground_process,
        )
        hwnd, _ = _get_foreground_hwnd_and_title()
        return _get_foreground_process(hwnd).lower() in TERMINAL_PROCESSES
    except Exception:
        return False


def own_console_is_foreground() -> bool:
    """True when the agent's OWN host console owns the foreground.

    Typing then would inject into the terminal session that launched the agent.
    Best-effort: under Windows Terminal the console is a hidden ConPTY handle
    that is never foreground, so this stays inert there.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    try:
        import ctypes
        own = ctypes.windll.kernel32.GetConsoleWindow()
        if not own or not ctypes.windll.user32.IsWindowVisible(own):
            return False
        return bool(ctypes.windll.user32.GetForegroundWindow() == own)
    except Exception:
        return False


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
    vram_mb: int             # dedicated only — see shared_mb for integrated GPUs
    backend: str             # "nvidia" | "intel"
    shared_mb: int = 0       # system RAM an integrated GPU may borrow

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024, 1)

    @property
    def shared_gb(self) -> float:
        return round(self.shared_mb / 1024, 1)

    @property
    def usable_gb(self) -> float:
        """What the models can actually occupy.

        Win32_VideoController.AdapterRAM reports only the DEDICATED carve-out,
        which on an integrated GPU (Arc 140V and friends) is a couple of GB
        while the real budget is system memory. The startup banner printed
        that 2 GB figure next to a 16 GB part and read as a hard limit — a
        model that fits fine looked impossible. Windows lets an iGPU address
        roughly half of system RAM on top of its dedicated slice.
        """
        return round((self.vram_mb + self.shared_mb) / 1024, 1)


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
             # RAM= line first, then one line per Intel adapter. One
             # PowerShell start instead of two keeps the banner snappy.
             "\"RAM=$((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory)\"; "
             "Get-CimInstance Win32_VideoController | "
             "Where-Object { $_.Name -match 'Intel' } | "
             "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }"],
            capture_output=True, text=True, timeout=8,
        )
        if r.returncode == 0:
            shared_mb, i = 0, 0
            for line in (ln.strip() for ln in r.stdout.splitlines() if ln.strip()):
                if line.startswith("RAM="):
                    try:
                        # Windows lets an integrated GPU address about half of
                        # system RAM as "shared GPU memory".
                        shared_mb = int(line[4:]) // (1024 * 1024) // 2
                    except ValueError:
                        shared_mb = 0
                    continue
                name, _, ram = line.partition("|")
                try:
                    # AdapterRAM is bytes, and on an iGPU it is the dedicated
                    # carve-out only — not the budget a model has to fit in.
                    vram_mb = max(int(ram), 0) // (1024 * 1024)
                except ValueError:
                    vram_mb = 0
                gpus.append(GPUInfo(index=i, name=name.strip() or f"Intel GPU {i}",
                                    vram_mb=vram_mb, backend="intel",
                                    shared_mb=shared_mb))
                i += 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus

