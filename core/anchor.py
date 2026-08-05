# core/anchor.py
"""The window a task belongs to, and the rules that keep it there.

After a successful "open X" subtask, the window owning the foreground IS the
launched app — no name-to-process mapping needed, so this works for any app.
Everything afterwards is checked against that anchor:

  * before planning — the anchor must own the foreground, or actions land on
    whatever window happens to be on top (a leftover Edge error page once
    swallowed every action while Outlook sat behind it);
  * before clicking — the point must belong to the anchor's process, tested
    with the OS's own hit test. Grounding happily returns text found in a
    background browser tab, and each such click cost a full act-verify-refocus
    cycle (~30 s) before the mistake was even noticed.
"""
import os
import re
import time

from loguru import logger

from core import apps
from core.types import SubTask
from desktop import system

_LAUNCH_DESC_RX = re.compile(r"^\s*(?:open|launch|start)\b", re.IGNORECASE)

# Shell surfaces a click may legitimately land on while a task is anchored:
# Start menu and search flyouts. explorer.exe is deliberately absent — during
# an anchored subtask a click resolving to explorer is the desktop or taskbar,
# a stray point, never the task (a blind click at (32,95) once launched a
# desktop icon). Launch subtasks turn the whole gate off, so taskbar and
# desktop-icon launches are unaffected.
_SHELL_PROCS = frozenset({
    "searchhost.exe", "searchapp.exe", "searchui.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe",
})


class AppAnchor:
    """Tracks the task's app window and gates actions against it."""

    def __init__(self, log=None, own_hwnd=None):
        self.log = log or (lambda _msg: None)
        # Callable returning the agent's own window handle — a click there is
        # the VLM aiming at our log panel, never the task.
        self._own_hwnd = own_hwnd or (lambda: None)
        self.window: tuple[int, int, str] | None = None   # (hwnd, pid, exe)
        # True while a launch subtask runs: those legitimately click OUTSIDE
        # the anchor (Start menu results, desktop icons).
        self.gate_off = False

    # ── State ─────────────────────────────────────────────────────────────────

    def clear(self) -> None:
        self.window = None
        self.gate_off = False

    @property
    def is_set(self) -> bool:
        return self.window is not None

    @property
    def hwnd(self) -> int:
        return self.window[0] if self.window else 0

    @property
    def pid(self) -> int:
        return self.window[1] if self.window else 0

    @property
    def exe(self) -> str:
        return self.window[2] if self.window else ""

    # ── Setting the anchor ────────────────────────────────────────────────────

    def adopt_foreground(self, description: str) -> None:
        """Record the foreground window as the anchor after a successful launch."""
        desc = description.lower()
        if not _LAUNCH_DESC_RX.match(desc) or "already" in desc:
            return
        hwnd, pid, name = system.foreground_app()
        if not hwnd or hwnd == self._own_hwnd():
            return
        self.window = (hwnd, pid, name)
        self.log(f"  [ANCHOR] Task app window = {name} (hwnd={hwnd})")
        logger.info(f"[ORCHESTRATOR] App anchor set: {name} hwnd={hwnd}")

    def launch_already_satisfied(self, subtask: SubTask) -> bool:
        """True when "open X" is already done because X owns the foreground.

        Launching blindly when the app is already in front is actively harmful:
        a Start-menu Enter once landed on the WEB result and opened a browser
        page over the real Outlook. Terminals are excluded — reusing a
        pre-existing console can hand the task a window that is busy running
        something else, so they keep the stricter new-window rule.
        """
        desc = subtask.description
        if not _LAUNCH_DESC_RX.match(desc.lower()):
            return False
        if any(k in desc.lower() for k in apps.TERMINAL_LAUNCH_KEYS):
            return False
        signals = apps.launch_signals(desc)
        if not signals:
            return False
        hwnd, pid, name = system.foreground_app()
        if not hwnd or hwnd == self._own_hwnd():
            return False
        title = system.window_title(hwnd)
        haystack = f"{title} {name}".lower()
        if not any(s.lower() in haystack for s in signals if len(s) >= 3):
            return False
        self.window = (hwnd, pid, name)
        self.log(
            f"  [LAUNCH-SKIP] '{signals[0]}' already owns the foreground "
            f"('{title[:60]}') — anchored it, nothing to launch"
        )
        logger.info(
            f"[ORCHESTRATOR] Launch skipped: '{signals[0]}' foreground "
            f"(title='{title[:60]}', proc={name}) — anchored hwnd={hwnd}"
        )
        return True

    # ── Enforcing the anchor ──────────────────────────────────────────────────

    def ensure_foreground(self) -> None:
        """Bring the task's window back to the front if something covered it.

        Same-process windows (dialogs, pickers) count as the anchor being
        active. A window that no longer exists drops the anchor.
        """
        if not self.window:
            return
        hwnd, pid, name = self.window
        try:
            import ctypes
            user32 = ctypes.windll.user32
            if not user32.IsWindow(hwnd):
                logger.info(f"[ORCHESTRATOR] App anchor window gone ({name}) — cleared")
                self.window = None
                return
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                return
            fg_pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))
            if fg_pid.value == pid:
                return      # a dialog/popup of the same app — still our context
            self.log(f"  [FOCUS] Foreground is not {name} — refocusing the task app")
            logger.info(f"[ORCHESTRATOR] Refocusing task app {name} (hwnd={hwnd})")
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)      # SW_RESTORE
            if not user32.SetForegroundWindow(hwnd):
                # Refused when we lack foreground rights; SwitchToThisWindow
                # (alt-tab semantics) still works.
                user32.SwitchToThisWindow(hwnd, True)
            time.sleep(0.6)     # let the window paint before capture/grounding
        except Exception:
            pass

    def foreign_app_at(self, x: int, y: int) -> str | None:
        """Name of the OTHER app whose window owns pixel (x, y), else None.

        None (allow) when: no anchor, a launch subtask is running, the point
        belongs to the anchor's process (same pid or same exe — a WebView2
        child or same-app popup counts), the shell, or the hit test fails. A
        failed lookup never blocks: the agent must not refuse a click on a
        guess.
        """
        if not self.window or self.gate_off:
            return None
        root, pid, name = system.window_owner_at_point(x, y)
        if not root or root == self.hwnd:
            return None
        if not pid or pid == self.pid:
            return None
        if pid == os.getpid():
            return "the agent's own GUI window"
        if not name:
            return None
        low = name.lower()
        if low == (self.exe or "").lower() or low in _SHELL_PROCS:
            return None
        return name
