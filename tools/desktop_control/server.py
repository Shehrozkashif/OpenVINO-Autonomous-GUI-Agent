# tools/desktop_control/server.py
"""
Desktop Control Tool Server — FastAPI-based.
Runs on port 8015. Agents call POST /tools/call to execute desktop actions.

Start with: python -m tools.desktop_control.server
"""
import base64
import io
import time
from typing import Any, Dict

import pyautogui
import mss
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
import uvicorn

# Safety: moving mouse to top-left corner raises an exception, stopping the program.
# KEEP ENABLED during development.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # 100ms pause between actions — more reliable than 0

app = FastAPI(title="Desktop Control Tool Server", version="1.0")


class ToolRequest(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}


class ToolResponse(BaseModel):
    success: bool
    result: Dict[str, Any] = {}
    error: str = ""


TOOLS = {}


def tool(fn):
    """Register a function as a callable tool."""
    TOOLS[fn.__name__] = fn
    return fn


@tool
def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
    """Click at (x, y). button: left/right/middle. clicks: 1=single, 2=double."""
    pyautogui.click(x, y, button=button, clicks=clicks, duration=0.3)
    return {"action": f"{button}_click", "x": x, "y": y}


@tool
def mouse_move(x: int, y: int) -> dict:
    """Move mouse without clicking. Useful for hover effects."""
    pyautogui.moveTo(x, y, duration=0.3)
    return {"x": x, "y": y}


@tool
def type_text(text: str, interval: float = 0.05) -> dict:
    """Type text at current cursor position. interval: seconds between keystrokes."""
    pyautogui.typewrite(text, interval=interval)
    return {"typed": text}


@tool
def press_key(key: str) -> dict:
    """Press a single key. Examples: enter, escape, tab, f5, delete."""
    pyautogui.press(key)
    return {"key": key}


@tool
def hotkey(keys: list) -> dict:
    """Press key combination. Example: keys=["ctrl", "s"] for Ctrl+S."""
    pyautogui.hotkey(*keys)
    return {"hotkey": "+".join(keys)}


@tool
def scroll(x: int, y: int, clicks: int = 3, direction: str = "down") -> dict:
    """Scroll at (x, y). direction: up or down."""
    amount = clicks if direction == "up" else -clicks
    pyautogui.scroll(amount, x=x, y=y)
    return {"direction": direction, "clicks": clicks}


@tool
def screenshot() -> dict:
    """Capture screen and return base64 JPEG."""
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64, "width": img.width, "height": img.height}


@tool
def get_screen_size() -> dict:
    """Return screen dimensions."""
    w, h = pyautogui.size()
    return {"width": w, "height": h}


@app.post("/tools/call", response_model=ToolResponse)
async def call_tool(request: ToolRequest):
    if request.name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
    try:
        result = TOOLS[request.name](**request.arguments)
        return ToolResponse(success=True, result=result)
    except Exception as e:
        return ToolResponse(success=False, error=str(e))


@app.get("/tools/list")
async def list_tools():
    return {"tools": list(TOOLS.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8015)
