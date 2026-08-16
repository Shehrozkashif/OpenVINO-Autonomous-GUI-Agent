# desktop/input.py
"""Windows desktop controller — raw Win32 SendInput, no third-party input library.

Mouse/keyboard events are injected via ctypes bindings to user32.SendInput
(the same OS-level API pynput and every other automation tool wrap). Typed
text uses KEYEVENTF_UNICODE so arbitrary Unicode characters work regardless
of the active keyboard layout; named keys and hotkeys resolve to real virtual
key codes so applications recognise them as actual shortcuts.

Nothing here touches ctypes.windll at import time — only inside functions —
so the module can still be imported (and its higher-level methods mocked)
on a non-Windows host for unit testing.
"""
import ctypes
import os
import threading
import time
from ctypes import wintypes

from loguru import logger

# ── Win32 SendInput structures ─────────────────────────────────────────────────

_ULONG_PTR = wintypes.WPARAM  # pointer-sized unsigned int on both 32/64-bit


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1

_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_MOUSEEVENTF_ABSOLUTE = 0x8000

_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004

_MOUSE_DOWN_FLAG = {
    "left": _MOUSEEVENTF_LEFTDOWN, "right": _MOUSEEVENTF_RIGHTDOWN, "middle": _MOUSEEVENTF_MIDDLEDOWN,
}
_MOUSE_UP_FLAG = {
    "left": _MOUSEEVENTF_LEFTUP, "right": _MOUSEEVENTF_RIGHTUP, "middle": _MOUSEEVENTF_MIDDLEUP,
}


def _send_input(inp: "_INPUT") -> None:
    """Inject one input event into the OS event queue via user32.SendInput."""
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _mouse_event(flags: int, data: int = 0) -> None:
    """Send one mouse event (a button down/up flag, or a wheel step in `data`)."""
    inp = _INPUT(type=_INPUT_MOUSE)
    inp.mi = _MOUSEINPUT(0, 0, data, flags, 0, 0)
    _send_input(inp)


def _key_event(vk: int, keyup: bool = False, extended: bool = False) -> None:
    """Press or release one virtual-key code (extended=True for nav/right-side keys)."""
    flags = _KEYEVENTF_KEYUP if keyup else 0
    if extended:
        flags |= _KEYEVENTF_EXTENDEDKEY
    inp = _INPUT(type=_INPUT_KEYBOARD)
    inp.ki = _KEYBDINPUT(vk, 0, flags, 0, 0)
    _send_input(inp)


def _unicode_key_event(ch: str, keyup: bool = False) -> None:
    """Type one Unicode character directly, bypassing the keyboard layout.

    KEYEVENTF_UNICODE carries the codepoint itself (not a key code), so any
    character types correctly regardless of the active layout.
    """
    flags = _KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if keyup else 0)
    inp = _INPUT(type=_INPUT_KEYBOARD)
    inp.ki = _KEYBDINPUT(0, ord(ch), flags, 0, 0)
    _send_input(inp)


def _set_cursor_pos(x: int, y: int) -> None:
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def _get_cursor_pos() -> tuple:
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _mouse_move_abs(x: int, y: int) -> None:
    """Move the cursor via a real SendInput MOVE event (not SetCursorPos).

    WebView2/Electron apps hit-test and fire hover (pointerover) from events
    that arrive through the input pipeline; a SetCursorPos teleport does not
    always register. Coordinates are normalized to the 0-65535 virtual-desktop
    space SendInput expects for absolute moves.
    """
    u = ctypes.windll.user32
    vx, vy = u.GetSystemMetrics(76), u.GetSystemMetrics(77)      # SM_X/YVIRTUALSCREEN
    vw, vh = u.GetSystemMetrics(78), u.GetSystemMetrics(79)      # SM_C X/Y VIRTUALSCREEN
    if vw <= 0 or vh <= 0:
        _set_cursor_pos(x, y)
        return
    nx = int((x - vx) * 65535 / (vw - 1))
    ny = int((y - vy) * 65535 / (vh - 1))
    inp = _INPUT(type=_INPUT_MOUSE)
    inp.mi = _MOUSEINPUT(nx, ny, 0,
                         _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK,
                         0, 0)
    _send_input(inp)


def _focus_window_at(x: int, y: int) -> None:
    """Bring the top-level window under (x, y) to the foreground before clicking.

    A click into a background window is delivered but web engines frequently
    use the first click only to take focus, swallowing the action (live: Teams'
    'New meeting' ignored repeated pixel clicks). Best-effort — foreground
    rules can deny the request, and the click proceeds regardless.
    """
    try:
        u = ctypes.windll.user32
        pt = wintypes.POINT(int(x), int(y))
        hwnd = u.WindowFromPoint(pt)
        if not hwnd:
            return
        root = u.GetAncestor(hwnd, 2)   # GA_ROOT
        # Own-window trap: the agent's GUI is capture-EXCLUDED, so screenshots
        # (and therefore grounding) see the app BEHIND it — but a physical
        # click at that point lands on the GUI and silently does nothing
        # (live: 16 minutes of delta=0 clicks with Teams "visible" in OCR).
        # If the window that would receive this click belongs to our own
        # process, minimize it so the click reaches the app underneath.
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(root, ctypes.byref(pid))
        if pid.value == os.getpid():
            logger.warning(
                f"[Controller] Own GUI window is covering the click point "
                f"({x},{y}) — minimizing it so the click reaches the app behind"
            )
            u.ShowWindow(root, 6)   # SW_MINIMIZE
            time.sleep(0.4)
            hwnd = u.WindowFromPoint(pt)
            root = u.GetAncestor(hwnd, 2) if hwnd else 0
        if root and root != u.GetForegroundWindow():
            u.SetForegroundWindow(root)
            time.sleep(0.15)
    except Exception:
        pass


# ── Cursor glide ──────────────────────────────────────────────────────────────
# Purely cosmetic: the physical cursor moves along a smooth eased path instead
# of teleporting, so a human watching (or a demo recording) can follow the
# agent's actions. The agent's own perception is unaffected — GDI screenshots
# do not include the cursor — and coordinates/action policy are unchanged.

_GLIDE_MIN_S = 0.08        # short hops still read as motion
_GLIDE_MAX_S = 0.30        # cap so long moves never slow the pipeline
_GLIDE_SPEED_PX_S = 3500   # duration = distance / speed, clamped to the above


def _glide_max_s() -> float:
    """Longest a single cursor move may take, in seconds.

    AGENT_GLIDE_MAX_S overrides the cap so the cursor can be slowed down to
    watch. The 0.30 s default keeps the pipeline quick, but a jump across a
    1852 px screen is then over in a third of a second, which is easy to miss
    when you are trying to see what the agent is doing. Setting it to, say,
    1.5 stretches every move; nothing else changes, since the glide is purely
    cosmetic (screenshots do not capture the cursor).
    """
    raw = os.environ.get("AGENT_GLIDE_MAX_S", "").strip()
    if not raw:
        return _GLIDE_MAX_S
    try:
        return max(_GLIDE_MIN_S, min(10.0, float(raw)))
    except ValueError:
        return _GLIDE_MAX_S


def _glide_cursor(x: int, y: int) -> None:
    """Move the cursor to (x, y) along a smoothstep-eased path."""
    try:
        sx, sy = _get_cursor_pos()
    except Exception:
        _set_cursor_pos(x, y)   # cursor state unreadable — jump as before
        return
    dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
    if dist < 4:
        _set_cursor_pos(x, y)
        return
    dur = max(_GLIDE_MIN_S, min(_glide_max_s(), dist / _GLIDE_SPEED_PX_S))
    steps = max(2, int(dur / 0.01))
    for i in range(1, steps):
        t = i / steps
        e = t * t * (3.0 - 2.0 * t)   # smoothstep ease-in/out
        wx = round(sx + (x - sx) * e)
        wy = round(sy + (y - sy) * e)
        # A WAYPOINT must never enter the kill-switch corner (x<=2, y<=2);
        # only a deliberate final target may land there.
        if wx <= 2 and wy <= 2:
            wx = 3
        _set_cursor_pos(wx, wy)
        time.sleep(dur / steps)
    _set_cursor_pos(x, y)


# ── Key name table (Windows virtual-key codes) ────────────────────────────────

_VK_MAP: dict = {
    "enter": 0x0D, "return": 0x0D,
    "escape": 0x1B, "esc": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "delete": 0x2E,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "home": 0x24, "end": 0x23,
    "pageup": 0x21, "page_up": 0x21, "pgup": 0x21,
    "pagedown": 0x22, "page_down": 0x22, "pgdn": 0x22,
    "space": 0x20,
    "ctrl": 0x11, "ctrl_l": 0xA2, "ctrl_r": 0xA3,
    "alt": 0x12, "alt_l": 0xA4, "alt_r": 0xA5,
    "shift": 0x10, "shift_l": 0xA0, "shift_r": 0xA1,
    "super": 0x5B, "win": 0x5B, "winleft": 0x5B, "cmd": 0x5B,
    "capslock": 0x14,
    "print_screen": 0x2C, "printscreen": 0x2C,
    "print": 0x2C, "prtscn": 0x2C, "prtsc": 0x2C,
    "scroll_lock": 0x91,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    # Numpad / arithmetic. A planner driving a calculator emits key_press
    # "add" between the operands, which is the obvious thing to write and was
    # an "Unknown key — skipping" that failed the step three times over and
    # sent the whole run off to re-plan (live: 'add any two numbers using
    # calculator' never got past the first operator). Numpad virtual keys
    # deliberately: they carry no shift state, so they do not depend on the
    # keyboard layout the way OEM_PLUS and friends do.
    "add": 0x6B, "plus": 0x6B,
    "subtract": 0x6D, "minus": 0x6D,
    "multiply": 0x6A, "asterisk": 0x6A, "star": 0x6A,
    "divide": 0x6F, "slash": 0x6F,
    "decimal": 0x6E,
    # A calculator commits with Enter; "=" as a plain character still works
    # through the single-character path for text fields.
    "equals": 0x0D, "equal": 0x0D,
    "numpad0": 0x60, "numpad1": 0x61, "numpad2": 0x62, "numpad3": 0x63,
    "numpad4": 0x64, "numpad5": 0x65, "numpad6": 0x66, "numpad7": 0x67,
    "numpad8": 0x68, "numpad9": 0x69,
}

# Keys that must set KEYEVENTF_EXTENDEDKEY to be interpreted correctly
# (distinguishes them from their numpad/left-side counterparts).
_EXTENDED_VK_NAMES = frozenset({
    "delete", "home", "end", "pageup", "page_up", "pgup",
    "pagedown", "page_down", "pgdn", "up", "down", "left", "right",
    "ctrl_r", "alt_r", "super", "win", "winleft", "cmd",
})


def _vk_for_char(ch: str):
    """Return (vk, needs_shift) for a single character on the active keyboard
    layout via VkKeyScanW, or (None, False) if it cannot be typed as a key.
    """
    res = ctypes.windll.user32.VkKeyScanW(ctypes.c_wchar(ch))
    if res == -1 or (res & 0xFF) == 0xFF:
        return None, False
    return res & 0xFF, bool((res >> 8) & 1)


def _resolve_key(name: str):
    """Return (vk, extended, needs_shift) for a key name, or None if unresolvable."""
    lower = name.lower().strip()
    if not lower:
        return None
    if lower in _VK_MAP:
        return _VK_MAP[lower], lower in _EXTENDED_VK_NAMES, False
    if len(name) == 1:
        vk, needs_shift = _vk_for_char(name)
        if vk is not None:
            return vk, False, needs_shift
    return None


# ── Key dispatch ──────────────────────────────────────────────────────────────

def _send_key_name(name: str) -> bool:
    """Press and release a named key (e.g. "enter", "f5", "a"), shift-holding if needed.
    Returns False for unknown key names so the step fails honestly instead of
    reporting a success for a key press that never happened.
    """
    resolved = _resolve_key(name)
    if resolved is None:
        logger.warning(f"[Controller] Unknown key '{name}' — skipping")
        return False
    vk, extended, needs_shift = resolved
    if needs_shift:
        _key_event(_VK_MAP["shift"], keyup=False)
    _key_event(vk, keyup=False, extended=extended)
    time.sleep(0.05)
    _key_event(vk, keyup=True, extended=extended)
    if needs_shift:
        _key_event(_VK_MAP["shift"], keyup=True)
    return True


def _send_hotkey(*key_names: str) -> None:
    """Press a chord like ctrl+shift+s: hold each modifier, tap the last key, release all.

    A try/finally guarantees held modifiers are released even if the main key fails.
    """
    names = [n.strip() for n in key_names if n and n.strip()]
    resolved = [_resolve_key(n) for n in names]
    modifiers, main = resolved[:-1], (resolved[-1] if resolved else None)

    # Use try/finally so modifiers are ALWAYS released even if the main key fails
    pressed: list = []
    try:
        for mod, name in zip(modifiers, names[:-1], strict=True):
            if mod is None:
                logger.warning(f"[Controller] Unknown modifier '{name}' in hotkey — skipping")
                continue
            vk, extended, _ = mod
            _key_event(vk, keyup=False, extended=extended)
            pressed.append((vk, extended))
        time.sleep(0.05)
        if main is not None:
            vk, extended, needs_shift = main
            if needs_shift:
                _key_event(_VK_MAP["shift"], keyup=False)
            _key_event(vk, keyup=False, extended=extended)
            time.sleep(0.05)
            _key_event(vk, keyup=True, extended=extended)
            if needs_shift:
                _key_event(_VK_MAP["shift"], keyup=True)
        else:
            logger.warning(f"[Controller] hotkey main key not resolved: {names}")
    finally:
        for vk, extended in reversed(pressed):
            try:
                _key_event(vk, keyup=True, extended=extended)
            except Exception:
                pass


def _send_text(text: str, interval: float = 0.04) -> None:
    """Type a string one Unicode character at a time, with a small inter-key delay."""
    for ch in text:
        _unicode_key_event(ch, keyup=False)
        time.sleep(interval * 0.4)
        _unicode_key_event(ch, keyup=True)
        time.sleep(interval * 0.6)


# ── DesktopController ─────────────────────────────────────────────────────────

class DesktopController:
    """Windows desktop controller — mouse and keyboard via raw Win32 SendInput."""

    # Optional UI hook fired on every pointer action: on_pointer(x, y, kind).
    # The GUI uses it to pulse a capture-excluded ring where the click lands,
    # so the user can follow the agent's mouse work. Never blocks or raises
    # into the action path.
    on_pointer = None

    def _notify_pointer(self, x: int, y: int, kind: str):
        if self.on_pointer is not None:
            try:
                self.on_pointer(x, y, kind)
            except Exception:
                pass

    def click(self, x: int, y: int, button: str = "left") -> bool:
        x, y = int(x), int(y)
        down = _MOUSE_DOWN_FLAG.get(button, _MOUSEEVENTF_LEFTDOWN)
        up = _MOUSE_UP_FLAG.get(button, _MOUSEEVENTF_LEFTUP)
        self._notify_pointer(x, y, button)
        _glide_cursor(x, y)
        # Real-input hover sequence: focus the target window, then deliver the
        # position through SendInput MOVE events so hover-driven UIs (web
        # engines, custom toolkits) register pointer-over before the press.
        # A 1px settle move guarantees at least one WM_MOUSEMOVE at the target.
        _focus_window_at(x, y)
        _mouse_move_abs(x + 1, y)
        time.sleep(0.03)
        _mouse_move_abs(x, y)
        time.sleep(0.15)
        _mouse_event(down)
        time.sleep(0.08)
        _mouse_event(up)
        time.sleep(0.12)
        logger.info(f"[ACTION] click({button}) @ ({x},{y})")
        return True

    def right_click(self, x: int, y: int) -> bool:
        return self.click(x, y, button="right")

    def double_click(self, x: int, y: int) -> bool:
        x, y = int(x), int(y)
        self._notify_pointer(x, y, "double")
        _glide_cursor(x, y)
        _focus_window_at(x, y)
        _mouse_move_abs(x + 1, y)
        time.sleep(0.03)
        _mouse_move_abs(x, y)
        time.sleep(0.15)
        _mouse_event(_MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.08)
        _mouse_event(_MOUSEEVENTF_LEFTUP)
        time.sleep(0.10)
        _mouse_event(_MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.08)
        _mouse_event(_MOUSEEVENTF_LEFTUP)
        time.sleep(0.12)
        logger.info(f"[ACTION] double_click @ ({x},{y})")
        return True

    def type_text(self, text: str, interval: float = 0.04,
                  use_clipboard: bool = False, sensitive: bool = False) -> bool:
        """Type text via keystrokes (default) or clipboard paste.

        use_clipboard=True: sets clipboard → ctrl+v — instant, handles all Unicode.
        use_clipboard=False: keystroke-by-keystroke — correct for terminal prompts
                             where ctrl+v inserts a literal control character.
        sensitive=True:     the text is a secret (password/token). It is redacted
                            from logs and the clipboard is cleared after paste.
        """
        # NEVER log secret values. Redact to a fixed mask of constant length.
        _shown = "***" if sensitive else f"'{text[:40]}'"
        if use_clipboard:
            from desktop.clipboard import paste_type
            if paste_type(text, _send_hotkey, sensitive=sensitive):
                logger.info(f"[ACTION] type (clipboard) {_shown}")
                return True
            # fall through to keystroke if clipboard unavailable
        _send_text(text, interval)
        logger.info(f"[ACTION] type (keystroke) {_shown}")
        return True

    def press_key(self, key: str) -> bool:
        key = key.strip()
        # Planners sometimes put a chord ("alt+left") in the key field —
        # route it through the hotkey path instead of failing on the name.
        if "+" in key and _resolve_key(key) is None:
            return self.hotkey(*key.split("+"))
        ok = _send_key_name(key)
        logger.info(f"[ACTION] key_press '{key}'")
        return ok

    def hotkey(self, *keys: str) -> bool:
        # Filter empty tokens before dispatch so "ctrl++s".split("+") → ["ctrl","","s"] is safe
        clean = [k.strip() for k in keys if k and k.strip()]
        if not clean:
            logger.warning("[ACTION] hotkey called with no valid keys")
            return False
        _send_hotkey(*clean)
        logger.info(f"[ACTION] hotkey {'+'.join(clean)}")
        return True

    def scroll(self, x: int, y: int, clicks: int = 5, direction: str = "down") -> bool:
        x, y = int(x), int(y)
        _glide_cursor(x, y)
        time.sleep(0.05)
        dy = clicks if direction == "up" else -clicks
        _mouse_event(_MOUSEEVENTF_WHEEL, data=dy * 120)
        logger.info(f"[ACTION] scroll {direction} {clicks}× @ ({x},{y})")
        return True

    def release_all_modifiers(self) -> None:
        """Best-effort release of every modifier key.

        Called by the kill switch so a stop never leaves Ctrl/Alt/Shift/Win
        stuck down (which would otherwise hijack the user's keyboard).
        """
        for name in ("ctrl", "alt", "shift", "win"):
            try:
                _key_event(_VK_MAP[name], keyup=True)
            except Exception:
                pass

# ── Global emergency kill switch ──────────────────────────────────────────────

class KillSwitch:
    """Global emergency-stop poller (Fix C8).

    The agent minimises its own window while it runs, so the on-screen Stop
    button can be obscured by the app being driven. A background thread polls
    real, OS-global input state (not window-focus-dependent) so the user can
    always halt automation:

      • Press Esc three times quickly, OR
      • Slam the mouse into the top-left corner of the screen (failsafe).

    On trigger it calls the supplied `on_trigger` callback (which sets the
    orchestrator stop event) and releases any held modifier keys.
    """

    _POLL_INTERVAL_S = 0.03
    _VK_ESCAPE = 0x1B

    def __init__(self, on_trigger, controller: "DesktopController" = None):
        self._on_trigger = on_trigger
        self._controller = controller
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._esc_times: list = []

    def start(self) -> None:
        try:
            _ = ctypes.windll.user32  # validate Win32 is available before spawning the poller
        except Exception as e:
            logger.warning(f"[KILL-SWITCH] Win32 API unavailable — kill switch disabled ({e})")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("[KILL-SWITCH] Armed - triple-Esc or top-left corner to stop")

    def _poll_loop(self) -> None:
        user32 = ctypes.windll.user32
        while not self._stop_event.is_set():
            try:
                # Low bit of GetAsyncKeyState = "was pressed since the last call" —
                # this reliably catches a brief tap even between polls.
                state = user32.GetAsyncKeyState(self._VK_ESCAPE)
                if state & 0x0001:
                    now = time.time()
                    self._esc_times = [t for t in self._esc_times if now - t < 1.0]
                    self._esc_times.append(now)
                    if len(self._esc_times) >= 3:
                        self._esc_times = []
                        self._fire("triple-Esc")

                x, y = _get_cursor_pos()
                if x <= 2 and y <= 2:
                    self._fire("top-left corner failsafe")
            except Exception:
                pass
            self._stop_event.wait(self._POLL_INTERVAL_S)

    def _fire(self, reason: str) -> None:
        logger.warning(f"[KILL-SWITCH] EMERGENCY STOP ({reason})")
        try:
            self._on_trigger()
        finally:
            if self._controller is not None:
                self._controller.release_all_modifiers()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
