# tools/desktop_control/controller.py
import httpx
from loguru import logger


class DesktopController:
    """
    Sends tool invocation requests to the Tool Server.
    Action Execution Agent uses this — never imports pyautogui directly.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8015"):
        self.server_url = server_url
        self.client = httpx.Client(timeout=15.0)

    def _call(self, tool_name: str, args: dict) -> dict:
        resp = self.client.post(
            f"{self.server_url}/tools/call",
            json={"name": tool_name, "arguments": args}
        )
        resp.raise_for_status()
        data = resp.json()
        if not data["success"]:
            raise RuntimeError(f"Tool '{tool_name}' failed: {data['error']}")
        return data["result"]

    def click(self, x: int, y: int, button: str = "left") -> bool:
        result = self._call("mouse_click", {"x": x, "y": y, "button": button})
        logger.info(f"[ACTION] click ({x},{y}) → {result}")
        return True

    def right_click(self, x: int, y: int) -> bool:
        result = self._call("mouse_click", {"x": x, "y": y, "button": "right"})
        logger.info(f"[ACTION] right_click ({x},{y}) → {result}")
        return True

    def double_click(self, x: int, y: int) -> bool:
        result = self._call("mouse_click", {"x": x, "y": y, "clicks": 2})
        logger.info(f"[ACTION] double_click ({x},{y}) → {result}")
        return True

    def type_text(self, text: str) -> bool:
        result = self._call("type_text", {"text": text})
        logger.info(f"[ACTION] type '{text[:30]}' → {result}")
        return True

    def press_key(self, key: str) -> bool:
        result = self._call("press_key", {"key": key})
        logger.info(f"[ACTION] key '{key}' → {result}")
        return True

    def hotkey(self, *keys: str) -> bool:
        # FIX: pass list, not *args, matching the server's hotkey(keys: list) signature
        result = self._call("hotkey", {"keys": list(keys)})
        logger.info(f"[ACTION] hotkey {'+'.join(keys)} → {result}")
        return True

    def screenshot_base64(self) -> str:
        return self._call("screenshot", {})["image_base64"]

    def scroll(self, x: int, y: int, clicks: int = 3, direction: str = "down") -> bool:
        result = self._call("scroll", {"x": x, "y": y, "clicks": clicks, "direction": direction})
        logger.info(f"[ACTION] scroll {direction} at ({x},{y}) → {result}")
        return True

    def is_server_running(self) -> bool:
        try:
            r = self.client.get(f"{self.server_url}/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False
