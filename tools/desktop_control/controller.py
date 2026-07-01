# tools/desktop_control/controller.py
"""Cross-platform desktop controller — auto-detects OS at startup.

  Linux  → XTest (Xlib): injects events at the X11 server level so they pass
            through GNOME Shell's global keyboard capture (needed for Activities).
            Falls back to pynput if XTest/Xlib is not available.

  Windows → pynput: uses win32 SendInput under the hood, works OS-globally.

  macOS   → pynput: uses Accessibility API, no special setup needed.
"""
import os
import platform
import time

from loguru import logger
from pynput.keyboard import Controller as _PKB
from pynput.keyboard import Key as _PKey
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

_mouse = MouseController()
_OS = platform.system()

# ── Backend selection ─────────────────────────────────────────────────────────

# Always define the pynput keyboard singleton symbol so it exists regardless of
# which backend is active (it is only instantiated when XTest is unavailable, but
# tests and tooling reference it unconditionally).
_pynput_kb = None
_XTEST_OK = False

if _OS == "Linux":
    try:
        from Xlib import X as _X
        from Xlib import display as _Xdisplay
        from Xlib.ext import xtest as _xtest
        _xdisplay = _Xdisplay.Display(os.environ.get("DISPLAY", ":0"))
        # Verify it actually works before committing
        _xdisplay.screen()
        _XTEST_OK = True
        logger.info("[Controller] Linux/X11 — keyboard backend: XTest")
    except Exception as _e:
        logger.warning(f"[Controller] XTest unavailable ({_e}) — falling back to pynput")

if not _XTEST_OK:
    _pynput_kb = _PKB()
    _backend_name = "pynput/win32" if _OS == "Windows" else "pynput"
    logger.info(f"[Controller] {_OS} — keyboard backend: {_backend_name}")


# ── Key name tables ───────────────────────────────────────────────────────────

_KEYSYM_MAP: dict = {
    "enter": 0xff0d, "return": 0xff0d,
    "escape": 0xff1b, "esc": 0xff1b,
    "tab": 0xff09,
    "backspace": 0xff08,
    "delete": 0xffff,
    "up": 0xff52, "down": 0xff54, "left": 0xff51, "right": 0xff53,
    "home": 0xff50, "end": 0xff57,
    "pageup": 0xff55, "page_up": 0xff55, "pgup": 0xff55,
    "pagedown": 0xff56, "page_down": 0xff56, "pgdn": 0xff56,
    "space": 0x0020,
    "ctrl": 0xffe3, "ctrl_l": 0xffe3, "ctrl_r": 0xffe4,
    "alt": 0xffe9, "alt_l": 0xffe9, "alt_r": 0xffea,
    "shift": 0xffe1, "shift_l": 0xffe1, "shift_r": 0xffe2,
    "super": 0xffeb, "win": 0xffeb, "winleft": 0xffeb, "cmd": 0xffeb,
    "capslock": 0xffe5,
    "print_screen": 0xff61, "printscreen": 0xff61,
    "print": 0xff61, "prtscn": 0xff61, "prtsc": 0xff61,
    "scroll_lock": 0xff14,
    "f1": 0xffbe, "f2": 0xffbf, "f3": 0xffc0, "f4": 0xffc1,
    "f5": 0xffc2, "f6": 0xffc3, "f7": 0xffc4, "f8": 0xffc5,
    "f9": 0xffc6, "f10": 0xffc7, "f11": 0xffc8, "f12": 0xffc9,
}

_PYNPUT_KEY_MAP: dict = {
    "enter": _PKey.enter, "return": _PKey.enter,
    "escape": _PKey.esc, "esc": _PKey.esc,
    "tab": _PKey.tab,
    "backspace": _PKey.backspace,
    "delete": _PKey.delete,
    "up": _PKey.up, "down": _PKey.down, "left": _PKey.left, "right": _PKey.right,
    "home": _PKey.home, "end": _PKey.end,
    "pageup": _PKey.page_up, "page_up": _PKey.page_up, "pgup": _PKey.page_up,
    "pagedown": _PKey.page_down, "page_down": _PKey.page_down, "pgdn": _PKey.page_down,
    "space": _PKey.space,
    "ctrl": _PKey.ctrl, "ctrl_l": _PKey.ctrl_l, "ctrl_r": _PKey.ctrl_r,
    "alt": _PKey.alt, "alt_l": _PKey.alt_l, "alt_r": _PKey.alt_r,
    "shift": _PKey.shift, "shift_l": _PKey.shift_l, "shift_r": _PKey.shift_r,
    "super": _PKey.cmd, "win": _PKey.cmd, "winleft": _PKey.cmd_l, "cmd": _PKey.cmd,
    "capslock": _PKey.caps_lock,
    "print_screen": _PKey.print_screen, "printscreen": _PKey.print_screen,
    "print": _PKey.print_screen, "prtscn": _PKey.print_screen, "prtsc": _PKey.print_screen,
    "scroll_lock": _PKey.scroll_lock,
    "f1": _PKey.f1, "f2": _PKey.f2, "f3": _PKey.f3, "f4": _PKey.f4,
    "f5": _PKey.f5, "f6": _PKey.f6, "f7": _PKey.f7, "f8": _PKey.f8,
    "f9": _PKey.f9, "f10": _PKey.f10, "f11": _PKey.f11, "f12": _PKey.f12,
}


# ── Linux/XTest implementation ────────────────────────────────────────────────

def _xtest_key(keysym: int, press: bool):
    keycode = _xdisplay.keysym_to_keycode(keysym)
    event_type = _X.KeyPress if press else _X.KeyRelease
    _xtest.fake_input(_xdisplay, event_type, keycode)
    _xdisplay.flush()


def _xtest_tap(keysym: int, delay: float = 0.03):
    _xtest_key(keysym, True)
    time.sleep(delay)
    _xtest_key(keysym, False)
    _xdisplay.flush()


def _xtest_send_key_name(name: str):
    lower = name.lower().strip()
    if not lower:
        return
    ks = _KEYSYM_MAP.get(lower)
    if ks:
        _xtest_tap(ks)
    elif len(name) == 1:
        _xtest_tap(ord(name))
    else:
        logger.warning(f"[Controller] Unknown key '{name}' — skipping")


def _xtest_send_hotkey(*key_names: str):
    # Filter out empty/None tokens that come from split("ctrl++s") etc.
    names = [n.strip() for n in key_names if n and n.strip()]
    syms = [_KEYSYM_MAP.get(n.lower()) or (ord(n) if len(n) == 1 else None) for n in names]
    modifiers, main = syms[:-1], syms[-1] if syms else None

    # Use try/finally so modifiers are ALWAYS released even if main key fails
    pressed = []
    try:
        for m in modifiers:
            if m:
                _xtest_key(m, True)
                pressed.append(m)
        time.sleep(0.05)
        if main:
            _xtest_tap(main)
        else:
            logger.warning(f"[Controller] hotkey main key not resolved: {names}")
    finally:
        for m in reversed(pressed):
            try:
                _xtest_key(m, False)
            except Exception:
                pass
        _xdisplay.flush()


def _xtest_send_text(text: str, interval: float = 0.04):
    for ch in text:
        ks = ord(ch)
        needs_shift = ch.isupper() or ch in '!@#$%^&*()_+{}|:"<>?~'
        try:
            if needs_shift:
                _xtest_key(_KEYSYM_MAP["shift"], True)
                time.sleep(0.01)
            _xtest_tap(ks, delay=0.02)
        finally:
            if needs_shift:
                _xtest_key(_KEYSYM_MAP["shift"], False)
                _xdisplay.flush()
        time.sleep(interval)


# ── Windows/macOS pynput implementation ──────────────────────────────────────

def _pynput_send_key_name(name: str):
    lower = name.lower().strip()
    if not lower:
        return
    k = _PYNPUT_KEY_MAP.get(lower) or (name if len(name) == 1 else None)
    if k:
        _pynput_kb.press(k)
        time.sleep(0.05)
        _pynput_kb.release(k)
    else:
        logger.warning(f"[Controller] Unknown key '{name}' — skipping")


def _pynput_send_hotkey(*key_names: str):
    names = [n.strip() for n in key_names if n and n.strip()]
    keys = [_PYNPUT_KEY_MAP.get(n.lower()) or (n if len(n) == 1 else None) for n in names]
    pressed = []
    try:
        for k in keys[:-1]:
            if k:
                _pynput_kb.press(k)
                pressed.append(k)
        time.sleep(0.05)
        if keys and keys[-1]:
            _pynput_kb.press(keys[-1])
            time.sleep(0.05)
            _pynput_kb.release(keys[-1])
    finally:
        for k in reversed(pressed):
            try:
                _pynput_kb.release(k)
            except Exception:
                pass


def _pynput_send_text(text: str, interval: float = 0.04):
    for ch in text:
        _pynput_kb.type(ch)
        time.sleep(interval)


# ── Unified dispatch ──────────────────────────────────────────────────────────

def _send_key_name(name: str):
    if _XTEST_OK:
        _xtest_send_key_name(name)
    else:
        _pynput_send_key_name(name)


def _send_hotkey(*key_names: str):
    if _XTEST_OK:
        _xtest_send_hotkey(*key_names)
    else:
        _pynput_send_hotkey(*key_names)


def _send_text(text: str, interval: float = 0.04):
    if _XTEST_OK:
        _xtest_send_text(text, interval)
    else:
        _pynput_send_text(text, interval)


# ── DesktopController ─────────────────────────────────────────────────────────

class DesktopController:
    """Cross-platform desktop controller.
    Mouse: pynput (works on Linux/Windows/macOS).
    Keyboard: XTest on Linux/X11, pynput on Windows/macOS.
    """

    def click(self, x: int, y: int, button: str = "left") -> bool:
        x, y = int(x), int(y)
        btn = {"left": Button.left, "right": Button.right, "middle": Button.middle}.get(button, Button.left)
        _mouse.position = (x, y)
        time.sleep(0.12)
        _mouse.press(btn)
        time.sleep(0.08)
        _mouse.release(btn)
        time.sleep(0.12)
        logger.info(f"[ACTION] click({button}) @ ({x},{y})")
        return True

    def right_click(self, x: int, y: int) -> bool:
        return self.click(x, y, button="right")

    def double_click(self, x: int, y: int) -> bool:
        x, y = int(x), int(y)
        _mouse.position = (x, y)
        time.sleep(0.12)
        _mouse.press(Button.left)
        time.sleep(0.08)
        _mouse.release(Button.left)
        time.sleep(0.10)
        _mouse.press(Button.left)
        time.sleep(0.08)
        _mouse.release(Button.left)
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
        _mouse.position = (x, y)
        time.sleep(0.05)
        dy = clicks if direction == "up" else -clicks
        _mouse.scroll(0, dy)
        logger.info(f"[ACTION] scroll {direction} {clicks}× @ ({x},{y})")
        return True

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> bool:
        """Click-and-drag from (x1,y1) to (x2,y2). Used for text selection and drag-drop."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        _mouse.position = (x1, y1)
        time.sleep(0.1)
        _mouse.press(Button.left)
        time.sleep(0.05)
        steps = max(10, int(duration / 0.02))
        for i in range(1, steps + 1):
            px = x1 + (x2 - x1) * i / steps
            py = y1 + (y2 - y1) * i / steps
            _mouse.position = (int(px), int(py))
            time.sleep(duration / steps)
        _mouse.release(Button.left)
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

        Called by the kill switch so a stop never leaves Ctrl/Alt/Shift/Super
        stuck down (which would otherwise hijack the user's keyboard).
        """
        for name in ("ctrl", "alt", "shift", "super"):
            try:
                if _XTEST_OK:
                    ks = _KEYSYM_MAP.get(name)
                    if ks:
                        _xtest_key(ks, False)
                else:
                    k = _PYNPUT_KEY_MAP.get(name)
                    if k:
                        _pynput_kb.release(k)
            except Exception:
                pass
        if _XTEST_OK:
            try:
                _xdisplay.flush()
            except Exception:
                pass

# ── Global emergency kill switch ──────────────────────────────────────────────

class KillSwitch:
    """Global hotkey listener that triggers an emergency stop (Fix C8).

    The agent minimises its own window while it runs, so the on-screen Stop
    button can be obscured by the app being driven. A process-global listener
    guarantees the user can always halt automation:

      • Press Esc three times quickly, OR
      • Slam the mouse into the top-left corner of the screen (failsafe).

    On trigger it calls the supplied `on_trigger` callback (which sets the
    orchestrator stop event) and releases any held modifier keys.
    """

    def __init__(self, on_trigger, controller: "DesktopController" = None):
        self._on_trigger = on_trigger
        self._controller = controller
        self._listener = None
        self._mouse_listener = None
        self._esc_times: list = []

    def start(self) -> None:
        try:
            from pynput import keyboard as _kb
            from pynput import mouse as _ms
        except Exception as e:
            logger.warning(f"[KILL-SWITCH] pynput unavailable — kill switch disabled ({e})")
            return

        def _on_press(key):
            try:
                if key == _kb.Key.esc:
                    now = time.time()
                    self._esc_times = [t for t in self._esc_times if now - t < 1.0]
                    self._esc_times.append(now)
                    if len(self._esc_times) >= 3:
                        self._fire("triple-Esc")
            except Exception:
                pass

        def _on_move(x, y):
            if x <= 2 and y <= 2:
                self._fire("top-left corner failsafe")

        try:
            self._listener = _kb.Listener(on_press=_on_press)
            self._listener.daemon = True
            self._listener.start()
            self._mouse_listener = _ms.Listener(on_move=_on_move)
            self._mouse_listener.daemon = True
            self._mouse_listener.start()
            logger.info("[KILL-SWITCH] Armed — triple-Esc or top-left corner to stop")
        except Exception as e:
            logger.warning(f"[KILL-SWITCH] Could not arm listeners: {e}")

    def _fire(self, reason: str) -> None:
        logger.warning(f"[KILL-SWITCH] EMERGENCY STOP ({reason})")
        try:
            self._on_trigger()
        finally:
            if self._controller is not None:
                self._controller.release_all_modifiers()

    def stop(self) -> None:
        for lst in (self._listener, self._mouse_listener):
            try:
                if lst is not None:
                    lst.stop()
            except Exception:
                pass
        self._listener = self._mouse_listener = None
