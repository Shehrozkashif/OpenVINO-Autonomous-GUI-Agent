# agents/action/action_agent.py
"""
Action Execution Agent — physically executes action steps on the desktop.
Calls the Tool Server (not pyautogui directly) via DesktopController.
"""
from loguru import logger

from core.protocols.a2a import ActionStep
from tools.desktop_control.controller import DesktopController


class ActionExecutionAgent:
    """
    Executes atomic ActionStep objects by translating them into tool calls.
    The Orchestrator calls execute(step, x=None, y=None) for each step.

    x, y are provided by the Grounding Agent for click/double_click steps.
    For type/key_press/hotkey steps, x and y are None.
    """

    def __init__(self, controller: DesktopController):
        self.controller = controller

    def execute(self, step: ActionStep, x: int = None, y: int = None) -> bool:
        """
        Execute one ActionStep. Returns True on success, False on failure.

        x, y: screen coordinates from UIGroundingAgent (required for click steps).
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

            elif step.action_type == "type":
                if not step.value:
                    logger.error(f"[ACTION] type step {step.id} has no value")
                    return False
                return self.controller.type_text(step.value)

            elif step.action_type == "key_press":
                # Tolerate models that accidentally put the key name in value instead of key
                key = step.key or step.value
                if not key:
                    logger.error(f"[ACTION] key_press step {step.id} has no key")
                    return False
                return self.controller.press_key(key)

            elif step.action_type == "hotkey":
                key = step.key or step.value
                if not key:
                    logger.error(f"[ACTION] hotkey step {step.id} has no key")
                    return False
                keys = key.split("+")
                return self.controller.hotkey(*keys)

            elif step.action_type == "scroll":
                if x is None or y is None:
                    logger.error(f"[ACTION] scroll step {step.id} missing coordinates")
                    return False
                direction = step.value or "down"
                return self.controller.scroll(x, y, direction=direction)

            elif step.action_type == "wait":
                import time
                duration = float(step.value or "1.0")
                logger.info(f"[ACTION] Waiting {duration}s")
                time.sleep(duration)
                return True

            elif step.action_type == "screenshot":
                # Side effect: capture and log current screen state
                _ = self.controller.screenshot_base64()
                return True

            else:
                logger.error(f"[ACTION] Unknown action_type: '{step.action_type}'")
                return False

        except Exception as e:
            logger.error(f"[ACTION] Step {step.id} raised exception: {e}")
            return False

    def scroll(self, x: int, y: int, direction: str = "down") -> bool:
        return self.controller.scroll(x, y, direction=direction)
