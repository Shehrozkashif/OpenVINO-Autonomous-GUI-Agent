# core/controller.py
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

_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_WHEEL = 0x0800

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
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _mouse_event(flags: int, data: int = 0) -> None:
    inp = _INPUT(type=_INPUT_MOUSE)
    inp.mi = _MOUSEINPUT(0, 0, data, flags, 0, 0)
    _send_input(inp)


def _key_event(vk: int, keyup: bool = False, extended: bool = False) -> None:
    flags = _KEYEVENTF_KEYUP if keyup else 0
    if extended:
        flags |= _KEYEVENTF_EXTENDEDKEY
    inp = _INPUT(type=_INPUT_KEYBOARD)
    inp.ki = _KEYBDINPUT(vk, 0, flags, 0, 0)
    _send_input(inp)


def _unicode_key_event(ch: str, keyup: bool = False) -> None:
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

def _send_key_name(name: str) -> None:
    resolved = _resolve_key(name)
    if resolved is None:
        logger.warning(f"[Controller] Unknown key '{name}' — skipping")
        return
    vk, extended, needs_shift = resolved
    if needs_shift:
        _key_event(_VK_MAP["shift"], keyup=False)
    _key_event(vk, keyup=False, extended=extended)
    time.sleep(0.05)
    _key_event(vk, keyup=True, extended=extended)
    if needs_shift:
        _key_event(_VK_MAP["shift"], keyup=True)


def _send_hotkey(*key_names: str) -> None:
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
    for ch in text:
        _unicode_key_event(ch, keyup=False)
        time.sleep(interval * 0.4)
        _unicode_key_event(ch, keyup=True)
        time.sleep(interval * 0.6)


# ── DesktopController ─────────────────────────────────────────────────────────

class DesktopController:
    """Windows desktop controller — mouse and keyboard via raw Win32 SendInput."""

    def click(self, x: int, y: int, button: str = "left") -> bool:
        x, y = int(x), int(y)
        down = _MOUSE_DOWN_FLAG.get(button, _MOUSEEVENTF_LEFTDOWN)
        up = _MOUSE_UP_FLAG.get(button, _MOUSEEVENTF_LEFTUP)
        _set_cursor_pos(x, y)
        time.sleep(0.12)
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
        _set_cursor_pos(x, y)
        time.sleep(0.12)
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
            from utils.clipboard import paste_type
            if paste_type(text, _send_hotkey, sensitive=sensitive):
                logger.info(f"[ACTION] type (clipboard) {_shown}")
                return True
            # fall through to keystroke if clipboard unavailable
        _send_text(text, interval)
        logger.info(f"[ACTION] type (keystroke) {_shown}")
        return True

    def press_key(self, key: str) -> bool:
        _send_key_name(key.strip())
        logger.info(f"[ACTION] key_press '{key}'")
        return True

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
        _set_cursor_pos(x, y)
        time.sleep(0.05)
        dy = clicks if direction == "up" else -clicks
        _mouse_event(_MOUSEEVENTF_WHEEL, data=dy * 120)
        logger.info(f"[ACTION] scroll {direction} {clicks}× @ ({x},{y})")
        return True

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> bool:
        """Click-and-drag from (x1,y1) to (x2,y2). Used for text selection and drag-drop."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _set_cursor_pos(x1, y1)
        time.sleep(0.1)
        _mouse_event(_MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.05)
        steps = max(10, int(duration / 0.02))
        for i in range(1, steps + 1):
            px = x1 + (x2 - x1) * i / steps
            py = y1 + (y2 - y1) * i / steps
            _set_cursor_pos(int(px), int(py))
            time.sleep(duration / steps)
        _mouse_event(_MOUSEEVENTF_LEFTUP)
        time.sleep(0.1)
        logger.info(f"[ACTION] drag ({x1},{y1}) → ({x2},{y2})")
        return True

    def screenshot_base64(self) -> str:
        import base64
        import io

        from core.capture.screenshot import ScreenCapture
        img = ScreenCapture().capture()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

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
