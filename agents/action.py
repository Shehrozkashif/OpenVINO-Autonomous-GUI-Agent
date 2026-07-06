# agents/action.py
"""Action Execution Agent — physically executes ActionSteps on the desktop.

Dispatches each step to a _do_<action_type> handler. Pointer/keyboard steps
go through DesktopController (raw Win32 SendInput); the structured
set_value / select / invoke steps go through the UIA patterns in
core/windows_uia, with a focus-and-type keyboard fallback for fields that
expose no writable ValuePattern.
"""
import re
import time

from loguru import logger

from core.controller import DesktopController
from core.protocols import ActionStep


def _alnum(s: str) -> str:
    """Lowercase alphanumerics only — read-back comparison that survives the
    app reformatting input ('7/7/2026' shown back as 'Tue 7/7/2026').
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

_TERMINAL_WORDS = frozenset(
    ("terminal", "command", "shell", "bash", "sh", "prompt", "console", "run")
)


def _should_use_clipboard_for(step, text: str) -> bool:
    """Use clipboard paste instead of keystroke-by-keystroke when:
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
    """Executes atomic ActionStep objects by translating them into tool calls.
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
        """Execute one ActionStep. Returns True on success, False on failure.

        x, y:      screen coordinates from UIGroundingAgent (source element).
        x2, y2:    destination coordinates for drag steps.

        Dispatches to the _do_<action_type> method for the step; every handler
        shares the (step, x, y, x2, y2) signature.
        """
        handler = getattr(self, f"_do_{step.action_type}", None)
        if handler is None:
            logger.error(f"[ACTION] Unknown action_type: '{step.action_type}'")
            return False
        try:
            return handler(step, x, y, x2, y2)
        except Exception as e:
            logger.error(f"[ACTION] Step {step.id} raised exception: {e}")
            return False

    # ── Pointer actions (coordinates come from the Grounding Agent) ──────────

    def _do_click(self, step, x, y, x2, y2) -> bool:
        if x is None or y is None:
            logger.error(f"[ACTION] click step {step.id} missing coordinates")
            return False
        return self.controller.click(x, y)

    def _do_right_click(self, step, x, y, x2, y2) -> bool:
        if x is None or y is None:
            logger.error(f"[ACTION] right_click step {step.id} missing coordinates")
            return False
        return self.controller.right_click(x, y)

    def _do_double_click(self, step, x, y, x2, y2) -> bool:
        if x is None or y is None:
            logger.error(f"[ACTION] double_click step {step.id} missing coordinates")
            return False
        return self.controller.double_click(x, y)

    def _do_drag(self, step, x, y, x2, y2) -> bool:
        if x is None or y is None:
            logger.error(f"[ACTION] drag step {step.id} missing source coordinates")
            return False
        if x2 is None or y2 is None:
            logger.error(f"[ACTION] drag step {step.id} missing destination coordinates")
            return False
        return self.controller.drag(x, y, x2, y2)

    def _do_scroll(self, step, x, y, x2, y2) -> bool:
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

    # ── Keyboard actions ──────────────────────────────────────────────────────

    def _do_type(self, step, x, y, x2, y2) -> bool:
        if not step.value:
            logger.error(f"[ACTION] type step {step.id} has no value")
            return False
        value, sensitive = self._substitute_credentials(step.value)
        use_cb = self._should_use_clipboard(step, value)
        return self.controller.type_text(
            value, use_clipboard=use_cb, sensitive=sensitive
        )

    def _do_key_press(self, step, x, y, x2, y2) -> bool:
        # Tolerate models that put key in value instead of key field
        key = (step.key or step.value or "").strip()
        if not key:
            logger.error(f"[ACTION] key_press step {step.id} has no key")
            return False
        return self.controller.press_key(key)

    def _do_hotkey(self, step, x, y, x2, y2) -> bool:
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

    # ── Passive actions ───────────────────────────────────────────────────────

    def _do_wait(self, step, x, y, x2, y2) -> bool:
        try:
            duration = float(step.value or "1.0")
        except (ValueError, TypeError):
            logger.warning(f"[ACTION] wait step {step.id} invalid duration '{step.value}' — using 1.0s")
            duration = 1.0
        logger.info(f"[ACTION] Waiting {duration}s")
        time.sleep(duration)
        return True

    def _do_screenshot(self, step, x, y, x2, y2) -> bool:
        _ = self.controller.screenshot_base64()
        return True

    # ── UIA structured-control actions ────────────────────────────────────────
    # Deterministic form-control manipulation through the accessibility
    # tree (ValuePattern / SelectionItemPattern / InvokePattern) with
    # read-back verification inside windows_uia. Returning False sends
    # the planner down the click/type fallback path.

    def _do_set_value(self, step, x, y, x2, y2) -> bool:
        if not step.target or step.value is None:
            logger.error(f"[ACTION] set_value step {step.id} needs target and value")
            return False
        value, sensitive = self._substitute_credentials(step.value)
        from core import windows_uia
        if windows_uia.set_element_value(step.target, value):
            return True
        # Fields without a writable ValuePattern (date/time segments, custom
        # web widgets) still take keyboard input: focus via the tree, replace
        # the content, verify by reading the focused control back.
        if not windows_uia.focus_element(step.target):
            logger.warning(
                f"[ACTION] set_value '{step.target}': ValuePattern write "
                f"failed and the control could not be focused"
            )
            return False
        self.controller.hotkey("ctrl", "a")
        if not self.controller.type_text(value, sensitive=sensitive):
            return False
        time.sleep(0.3)   # let the app commit the input before read-back
        info = windows_uia.focused_element_info() or {}
        typed = _alnum(value)
        seen = _alnum(str(info.get("value", "")))
        ok = bool(typed) and bool(seen) and (typed in seen or seen in typed)
        if ok:
            logger.info(f"[ACTION] set_value via focus+type '{step.target}' (verified)")
        else:
            logger.warning(
                f"[ACTION] set_value focus+type read-back mismatch on "
                f"'{step.target}': typed '{value[:40]}', field shows "
                f"'{str(info.get('value', ''))[:40]}'"
            )
        return ok

    def _do_select(self, step, x, y, x2, y2) -> bool:
        if not step.target or not step.value:
            logger.error(f"[ACTION] select step {step.id} needs target and value")
            return False
        from core import windows_uia
        return windows_uia.select_option(step.target, step.value)

    def _do_invoke(self, step, x, y, x2, y2) -> bool:
        if not step.target:
            logger.error(f"[ACTION] invoke step {step.id} needs a target")
            return False
        from core import windows_uia
        return windows_uia.invoke_element(step.target)

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _substitute_credentials(value: str) -> tuple[str, bool]:
        """Replace {{cred:site:field}} tokens with values from the OS keyring.

        Returns (text, sensitive): sensitive is True when a substitution
        happened, so the controller can suppress logging of the typed text.
        """
        try:
            from utils.credentials import has_tokens, substitute
            if has_tokens(value):
                value = substitute(value)
                logger.info("[ACTION] credential substitution applied")
                return value, True
        except Exception:
            pass
        return value, False
