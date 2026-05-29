# tools/desktop_control/server.py
"""
Desktop Control Tool Server — FastAPI-based.
Runs on port 8015. Agents call POST /tools/call to execute desktop actions.

Start with: python -m tools.desktop_control.server
"""
import base64
import io
import platform
import time
from typing import Any, Dict

# On Windows, tell the process it is DPI-aware so mss uses physical pixels.
if platform.system() == "Windows":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass

from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
import uvicorn

_mouse = MouseController()
_keyboard = KeyboardController()

# Map common key name strings → pynput Key objects or single-char strings
_KEY_MAP: dict = {
    "enter": Key.enter, "return": Key.enter,
    "escape": Key.esc,  "esc": Key.esc,
    "tab": Key.tab,
    "backspace": Key.backspace,
    "delete": Key.delete,
    "up": Key.up, "down": Key.down, "left": Key.left, "right": Key.right,
    "home": Key.home, "end": Key.end,
    "pageup": Key.page_up,   "page_up": Key.page_up,  "pgup": Key.page_up,
    "pagedown": Key.page_down, "page_down": Key.page_down, "pgdn": Key.page_down,
    "space": Key.space,
    "ctrl": Key.ctrl, "ctrl_l": Key.ctrl_l, "ctrl_r": Key.ctrl_r,
    "alt": Key.alt,   "alt_l": Key.alt_l,   "alt_r": Key.alt_r,
    "shift": Key.shift, "shift_l": Key.shift_l, "shift_r": Key.shift_r,
    "super": Key.cmd,  "win": Key.cmd, "winleft": Key.cmd_l, "cmd": Key.cmd,
    "capslock": Key.caps_lock,
    "f1": Key.f1, "f2": Key.f2, "f3": Key.f3, "f4": Key.f4,
    "f5": Key.f5, "f6": Key.f6, "f7": Key.f7, "f8": Key.f8,
    "f9": Key.f9, "f10": Key.f10, "f11": Key.f11, "f12": Key.f12,
}


def _resolve_key(name: str):
    """Convert a key name string to a pynput Key object or single character."""
    lower = name.lower()
    if lower in _KEY_MAP:
        return _KEY_MAP[lower]
    # Single printable character
    if len(name) == 1:
        return name
    # Unknown key — try as-is (pynput accepts some strings)
    return name


app = FastAPI(title="Desktop Control Tool Server", version="1.0")

TOOLS = {}


def tool(fn):
    TOOLS[fn.__name__] = fn
    return fn


@tool
def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
    """Click at (x, y). button: left/right/middle. clicks: 1=single, 2=double."""
    btn = {"left": Button.left, "right": Button.right, "middle": Button.middle}.get(button, Button.left)
    _mouse.position = (x, y)
    time.sleep(0.05)
    for _ in range(clicks):
        _mouse.click(btn)
        time.sleep(0.08)
    return {"action": f"{button}_click", "x": x, "y": y}


@tool
def mouse_move(x: int, y: int) -> dict:
    """Move mouse without clicking."""
    _mouse.position = (x, y)
    time.sleep(0.1)
    return {"x": x, "y": y}


@tool
def type_text(text: str, interval: float = 0.05) -> dict:
    """Type text at current cursor position."""
    for ch in text:
        _keyboard.type(ch)
        time.sleep(interval)
    return {"typed": text}


@tool
def press_key(key: str) -> dict:
    """Press a single key. Examples: enter, escape, tab, f5, delete."""
    k = _resolve_key(key)
    _keyboard.press(k)
    time.sleep(0.05)
    _keyboard.release(k)
    return {"key": key}


@tool
def hotkey(keys: list) -> dict:
    """Press key combination. Example: keys=[\"ctrl\", \"s\"] for Ctrl+S."""
    resolved = [_resolve_key(k) for k in keys]
    # Hold all but the last, press+release the last
    modifiers = resolved[:-1]
    main = resolved[-1]
    for mod in modifiers:
        _keyboard.press(mod)
    time.sleep(0.05)
    _keyboard.press(main)
    time.sleep(0.05)
    _keyboard.release(main)
    for mod in reversed(modifiers):
        _keyboard.release(mod)
    return {"hotkey": "+".join(keys)}


@tool
def scroll(x: int, y: int, clicks: int = 3, direction: str = "down") -> dict:
    """Scroll at (x, y). direction: up or down."""
    _mouse.position = (x, y)
    time.sleep(0.05)
    dy = clicks if direction == "up" else -clicks
    _mouse.scroll(0, dy)
    return {"direction": direction, "clicks": clicks}


@tool
def screenshot() -> dict:
    """Capture screen and return base64 JPEG."""
    from PIL import ImageGrab
    img = ImageGrab.grab().convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64, "width": img.width, "height": img.height}


@tool
def get_screen_size() -> dict:
    """Return screen dimensions."""
    from PIL import ImageGrab
    img = ImageGrab.grab()
    return {"width": img.width, "height": img.height}


@app.post("/tools/call", response_model=dict)
async def call_tool(request: dict):
    name = request.get("name")
    arguments = request.get("arguments", {})
    if name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    try:
        result = TOOLS[name](**arguments)
        return {"success": True, "result": result, "error": ""}
    except Exception as e:
        return {"success": False, "result": {}, "error": str(e)}


@app.get("/tools/list")
async def list_tools():
    return {"tools": list(TOOLS.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8015)
