# agents/action/action_agent.py
"""
Action Execution Agent — physically executes action steps on the desktop.
Calls the Tool Server (not pyautogui directly) via DesktopController.
"""
import time

from loguru import logger

from core.protocols.a2a import ActionStep
from tools.desktop_control.controller import DesktopController


_TERMINAL_WORDS = frozenset(
    ("terminal", "command", "shell", "bash", "sh", "prompt", "console", "run")
)


def _should_use_clipboard_for(step, text: str) -> bool:
    """
    Use clipboard paste instead of keystroke-by-keystroke when:
      - text is long (> 20 chars) OR contains non-ASCII / special symbols
      - AND the context is NOT a terminal (ctrl+v is a control char in bash)
    """
    desc = (step.description or "").lower()
    val  = (step.value or "").lower()
    is_terminal = bool(_TERMINAL_WORDS & set(desc.split()) | _TERMINAL_WORDS & set(val.split()))
    if is_terminal:
        return False
    has_complex = any(ord(c) > 127 or c in "@#$%^&*{}[]|<>?~`'\"" for c in text)
    return len(text) > 20 or has_complex


def _screen_center():
    """Return (cx, cy) of the primary screen — used as fallback for scroll/drag."""
    try:
        from core.capture.screenshot import _screen_size
        w, h = _screen_size()
        return w // 2, h // 2
    except Exception:
        return 960, 540


class ActionExecutionAgent:
    """
    Executes atomic ActionStep objects by translating them into tool calls.
    The Orchestrator calls execute(step, x=None, y=None) for each step.

    x, y are provided by the Grounding Agent for click/double_click/drag steps.
    For type/key_press/hotkey steps, x and y are None.
    """

    def __init__(self, controller: DesktopController):
        self.controller = controller
        self._should_use_clipboard = _should_use_clipboard_for

    def execute(
        self,
        step: ActionStep,
        x: int = None,
        y: int = None,
        x2: int = None,
        y2: int = None,
    ) -> bool:
        """
        Execute one ActionStep. Returns True on success, False on failure.

        x, y:      screen coordinates from UIGroundingAgent (source element).
        x2, y2:    destination coordinates for drag steps.
        """
        try:
            if step.action_type == "click":
                if x is None or y is None:
                    logger.error(f"[ACTION] click step {step.id} missing coordinates")
                    return False
                return self.controller.click(x, y)

            elif step.action_type == "right_click":
                if x is None or y is None:
                    logger.error(f"[ACTION] right_click step {step.id} missing coordinates")
                    return False
                return self.controller.right_click(x, y)

            elif step.action_type == "double_click":
                if x is None or y is None:
                    logger.error(f"[ACTION] double_click step {step.id} missing coordinates")
                    return False
                return self.controller.double_click(x, y)

            elif step.action_type == "drag":
                if x is None or y is None:
                    logger.error(f"[ACTION] drag step {step.id} missing source coordinates")
                    return False
                if x2 is None or y2 is None:
                    logger.error(f"[ACTION] drag step {step.id} missing destination coordinates")
                    return False
                return self.controller.drag(x, y, x2, y2)

            elif step.action_type == "type":
                if not step.value:
                    logger.error(f"[ACTION] type step {step.id} has no value")
                    return False
                # Substitute {{cred:site:field}} tokens before typing
                value = step.value
                sensitive = False
                try:
                    from utils.credentials import substitute, has_tokens
                    if has_tokens(value):
                        value = substitute(value)
                        sensitive = True
                        logger.info(f"[ACTION] credential substitution applied")
                except Exception:
                    pass
                use_cb = self._should_use_clipboard(step, value)
                return self.controller.type_text(
                    value, use_clipboard=use_cb, sensitive=sensitive
                )

            elif step.action_type == "key_press":
                # Tolerate models that put key in value instead of key field
                key = (step.key or step.value or "").strip()
                if not key:
                    logger.error(f"[ACTION] key_press step {step.id} has no key")
                    return False
                return self.controller.press_key(key)

            elif step.action_type == "hotkey":
                key = (step.key or step.value or "").strip()
                if not key:
                    logger.error(f"[ACTION] hotkey step {step.id} has no key")
                    return False
                # Filter empty tokens from malformed combos like "ctrl++s"
                keys = [k.strip() for k in key.split("+") if k.strip()]
                if not keys:
                    logger.error(f"[ACTION] hotkey step {step.id} produced no valid keys from '{key}'")
                    return False
                return self.controller.hotkey(*keys)

            elif step.action_type == "scroll":
                # Use screen center if grounding found no specific target
                sx, sy = x or 0, y or 0
                if sx == 0 and sy == 0:
                    sx, sy = _screen_center()
                direction = (step.value or "down").strip().lower()
                # Support "down:5" notation for custom scroll amount
                clicks = 5
                if ":" in direction:
                    direction, amt = direction.split(":", 1)
                    try:
                        clicks = int(amt)
                    except ValueError:
                        pass
                return self.controller.scroll(sx, sy, clicks=clicks, direction=direction)

            elif step.action_type == "wait":
                try:
                    duration = float(step.value or "1.0")
                except (ValueError, TypeError):
                    logger.warning(f"[ACTION] wait step {step.id} invalid duration '{step.value}' — using 1.0s")
                    duration = 1.0
                logger.info(f"[ACTION] Waiting {duration}s")
                time.sleep(duration)
                return True

            elif step.action_type == "screenshot":
                _ = self.controller.screenshot_base64()
                return True

            else:
                logger.error(f"[ACTION] Unknown action_type: '{step.action_type}'")
                return False

        except Exception as e:
            logger.error(f"[ACTION] Step {step.id} raised exception: {e}")
            return False
