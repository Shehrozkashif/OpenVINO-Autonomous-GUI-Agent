# core/error_handling.py
"""
Recovery strategies for common failure modes.
Each recovery function is called by the Orchestrator when a failure is detected.
"""
import time
from loguru import logger
from pynput.keyboard import Key, Controller as KeyboardController

_kb = KeyboardController()


def _press(key):
    _kb.press(key)
    time.sleep(0.05)
    _kb.release(key)


def escape_unexpected_dialogs():
    """Press Escape to dismiss any unexpected modal dialogs."""
    _press(Key.esc)
    time.sleep(0.3)
    _press(Key.esc)  # twice in case first one was consumed
    logger.info("[RECOVERY] Pressed Escape to dismiss dialogs")


def wait_for_app_response(max_wait_s: float = 5.0):
    """Wait for a potentially frozen app to respond."""
    logger.info(f"[RECOVERY] Waiting up to {max_wait_s}s for app to respond")
    time.sleep(max_wait_s)


def get_dpi_corrected_coords(x: int, y: int) -> tuple:
    """
    Correct coordinates for DPI scaling.
    On HiDPI screens, pyautogui reports logical coordinates but
    mss captures physical pixels. The VLM returns physical pixel coordinates,
    so we must divide by the DPI scale factor before calling pyautogui.
    """
    from core.capture.screenshot import ScreenCapture
    scale = ScreenCapture().get_dpi_scale()
    if abs(scale - 1.0) < 0.01:
        return x, y  # no correction needed
    corrected = int(x / scale), int(y / scale)
    logger.debug(f"[DPI] {x},{y} → {corrected[0]},{corrected[1]} (scale={scale:.2f})")
    return corrected