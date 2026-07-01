# core/windows_uia.py
"""Windows UIAutomation (UIA) — Stage 0 grounding for Windows.

Queries the Windows UI Automation tree for elements matching a text label.
Returns screen-pixel coordinates directly from the accessibility tree —
much faster (~20-50ms) and more precise than OCR for standard Windows
controls (buttons, menus, text fields, list items, toolbars).

Used as Stage 0 in UIGroundingAgent._locate().
Safe no-op if the `uiautomation` package is not installed.

Why UIA beats OCR for grounding:
  - Bounding boxes are exact — no OCR rendering artifacts
  - Works on controls with no visible text (icons with accessibility names)
  - Sees disabled/hidden controls that OCR misses
  - 10-20x faster than OCR + VLM inference chain
"""
import difflib
import threading

from loguru import logger

_uia = None          # lazy-loaded uiautomation module reference
_available: bool | None = None


# ── Module availability ───────────────────────────────────────────────────────

def _load() -> bool:
    global _uia, _available
    if _available is not None:
        return _available
    try:
        import uiautomation as _mod
        _uia = _mod
        _available = True
        logger.info("[UIA] Windows UIAutomation ready (Stage 0 grounding active)")
    except ImportError:
        logger.info(
            "[UIA] uiautomation not installed — Stage 0 disabled. "
            "To enable: pip install uiautomation"
        )
        _available = False
    return _available


def is_available() -> bool:
    return _load()


def _thread_com_init():
    """Initialize COM for the current worker thread.

    uiautomation requires per-thread COM initialization when called outside the
    import thread ("CoInitialize has not been called" otherwise). Returns the
    initializer object — keep it referenced for the duration of the UIA work;
    its destructor uninitializes COM.
    """
    try:
        return _uia.UIAutomationInitializerInThread()
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def find_element(
    target: str,
    fuzzy_threshold: float = 0.65,
    timeout_s: float = 1.5,
) -> tuple[int, int, float] | None:
    """Search the Windows UIA tree for a UI element matching `target`.
    Returns (screen_x, screen_y, confidence) or None if not found.

    Search order (fastest → broadest):
      1. Foreground window, depth 12 — deep enough for browser/Electron
         accessibility trees, where a page button can sit 10+ levels down
      2. All top-level windows, depth 4 — for taskbar / notification area

    Runs in a daemon thread with `timeout_s` to never stall the pipeline; the
    walk publishes its best match as it goes, so a timeout on a very deep tree
    still returns the best element found so far instead of nothing.
    Confidence is 1.0 for exact name match, ~0.85-0.92 for fuzzy match.
    """
    if not _load():
        return None

    result: list = [None]

    def _publish(candidate):
        result[0] = candidate

    def _search():
        _com = _thread_com_init()
        try:
            query = _strip_roles(target).lower().strip()
            if not query:
                return

            # Fast path — foreground window only
            fg = _uia.GetForegroundControl()
            r = _walk_and_match(fg, query, fuzzy_threshold, max_depth=12,
                                publish=_publish)
            if r:
                result[0] = r
                return

            # Slow path — remaining visible top-level windows
            desktop = _uia.GetRootControl()
            for win in desktop.GetChildren():
                try:
                    if win.Handle == fg.Handle:
                        continue
                    # Skip tiny / invisible windows (taskbar sub-elements, etc.)
                    rect = win.BoundingRectangle
                    if (rect.right - rect.left) < 10 or (rect.bottom - rect.top) < 10:
                        continue
                    r = _walk_and_match(win, query, fuzzy_threshold, max_depth=4,
                                        publish=_publish)
                    if r:
                        result[0] = r
                        return
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"[UIA] Search error for '{target}': {e}")
        finally:
            del _com   # release per-thread COM

    t = threading.Thread(target=_search, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if result[0]:
        x, y, conf = result[0]
        # Guard: "desktop" queries must not return taskbar-zone coordinates.
        # Windows UIA sometimes resolves "desktop" to a taskbar element (y > 92%
        # of screen height). Right-clicking there never opens the desktop context
        # menu. Reject it here so grounding falls through to OCR/VLM.
        if "desktop" in target.lower():
            try:
                import ctypes
                sh = ctypes.windll.user32.GetSystemMetrics(1)  # SM_CYSCREEN
                if sh > 0 and y > int(sh * 0.92):
                    logger.debug(
                        f"[UIA] Rejecting taskbar-zone coordinate ({x},{y}) for "
                        f"'desktop' query (threshold y>{int(sh*0.92)})"
                    )
                    result[0] = None
            except Exception:
                pass

    if result[0]:
        x, y, conf = result[0]
        logger.info(f"[UIA] '{target}' -> screen({x},{y}) conf={conf:.2f}")
    else:
        logger.debug(f"[UIA] '{target}' not found (timeout={timeout_s}s)")

    return result[0]


def get_interactive_elements(
    max_elements: int = 40,
    max_depth: int = 7,
    timeout_s: float = 1.5,
) -> list:
    """Collect interactive controls from the foreground window's UIA subtree.

    Returns [(name, control_type)] — e.g. [("Save", "Button"), ("File", "MenuItem")].
    Gives the planner a ground-truth list of elements it can actually click,
    instead of guessing from OCR tokens. Names returned here are guaranteed to
    be findable by Stage 0 grounding (same UIA tree, same matching).

    Runs in a daemon thread with a hard timeout so it can never stall planning.
    Returns [] on timeout, or when uiautomation is unavailable.
    """
    if not _load():
        return []

    result: list = [[]]

    def _collect():
        _com = _thread_com_init()
        try:
            out: list = []
            seen: set = set()

            def _walk(ctrl, depth: int):
                if depth > max_depth or len(out) >= max_elements:
                    return
                try:
                    ctype = ctrl.ControlTypeName
                    name = (ctrl.Name or "").strip()
                    if (
                        name
                        and len(name) <= 60
                        and ctype in _INTERACTIVE_CONTROL_TYPES
                        # Skip glyph-only names (icon-font codepoints like
                        # '') — meaningless as planner click targets.
                        and any(ch.isalnum() for ch in name)
                    ):
                        key = (name.lower(), ctype)
                        if key not in seen:
                            seen.add(key)
                            out.append((name, ctype.replace("Control", "")))
                    for child in ctrl.GetChildren():
                        if len(out) >= max_elements:
                            return
                        _walk(child, depth + 1)
                except Exception:
                    pass  # controls can vanish mid-walk

            _walk(_uia.GetForegroundControl(), 0)
            result[0] = out
        except Exception as e:
            logger.debug(f"[UIA] Interactive-element collection error: {e}")
        finally:
            del _com   # release per-thread COM

    t = threading.Thread(target=_collect, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


def focused_element_info(timeout_s: float = 1.0) -> dict | None:
    """Describe the control that currently owns keyboard focus.

    Returns {"rect": (l, t, r, b), "control_type": str, "name": str,
    "value": str} or None. "value" is the control's current text (ValuePattern
    or TextPattern), "" when the control exposes neither — e.g. password
    fields, which deliberately hide their content.

    Ground truth from the accessibility tree — used to verify actions whose
    effect is invisible to pixel comparison (a click that only moves the
    caret) or expensive to read back via OCR+LLM (text typed into a field).
    Runs in a daemon thread with a hard timeout, like the other queries, so it
    can never stall the pipeline.
    """
    if not _load():
        return None

    result: list = [None]

    def _get():
        _com = _thread_com_init()
        try:
            ctrl = _uia.GetFocusedControl()
            if ctrl is None:
                return
            rect = ctrl.BoundingRectangle
            if not _rect_valid(rect):
                return
            value = ""
            try:
                value = ctrl.GetValuePattern().Value or ""
            except Exception:
                try:
                    value = ctrl.GetTextPattern().DocumentRange.GetText(-1) or ""
                except Exception:
                    pass
            result[0] = {
                "rect": (rect.left, rect.top, rect.right, rect.bottom),
                "control_type": ctrl.ControlTypeName,
                "name": (ctrl.Name or "").strip(),
                "value": value,
            }
        except Exception as e:
            logger.debug(f"[UIA] Focused-element query error: {e}")
        finally:
            del _com   # release per-thread COM

    t = threading.Thread(target=_get, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


def save_dialog_open(timeout_s: float = 1.2) -> bool | None:
    """Is a native Windows file (Save/Save-As) dialog in the foreground?

    Detected structurally: the foreground window's subtree contains an Edit or
    ComboBox control whose accessible name is the dialog's filename field
    ("File name:"). The name comes from the accessibility tree — the same
    field the agent types the path into — so this cannot misread pixels the
    way OCR on a downscaled screenshot can.

    Returns True/False when the tree answered, or None when UIA is
    unavailable or the scan timed out (caller should fall back to OCR).
    """
    if not _load():
        return None

    result: list = [None]

    def _scan():
        _com = _thread_com_init()
        try:
            found = [False]

            def _walk(ctrl, depth: int):
                if depth > 8 or found[0]:
                    return
                try:
                    if ctrl.ControlTypeName in ("EditControl", "ComboBoxControl"):
                        name = (ctrl.Name or "").strip().lower()
                        if name.startswith("file name"):
                            found[0] = True
                            return
                    for child in ctrl.GetChildren():
                        if found[0]:
                            return
                        _walk(child, depth + 1)
                except Exception:
                    pass  # controls can vanish mid-walk

            _walk(_uia.GetForegroundControl(), 0)
            result[0] = found[0]
        except Exception as e:
            logger.debug(f"[UIA] Save-dialog scan error: {e}")
        finally:
            del _com   # release per-thread COM

    t = threading.Thread(target=_scan, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


# UIA control types a user can act on — used by get_interactive_elements().
_INTERACTIVE_CONTROL_TYPES = frozenset({
    "ButtonControl", "MenuItemControl", "ListItemControl", "TreeItemControl",
    "TabItemControl", "HyperlinkControl", "ComboBoxControl", "CheckBoxControl",
    "RadioButtonControl", "EditControl", "SplitButtonControl",
    "DocumentControl", "SliderControl", "SpinnerControl",
})


# ── Tree walking ──────────────────────────────────────────────────────────────

def _walk_and_match(
    root,
    query: str,
    threshold: float,
    max_depth: int,
    publish=None,
) -> tuple[int, int, float] | None:
    """DFS walk of a UIA subtree.  Returns the best (x, y, confidence) match or None.
    Stops immediately on an exact name match.

    publish: optional callback invoked with each new best match as the walk
    progresses — lets the caller keep the best result found so far even when
    the walk is cut off by a timeout (deep browser/Electron trees).
    """
    best_result: list = [None]
    best_score:  list = [0.0]

    def _walk(ctrl, depth: int):
        if depth > max_depth:
            return
        if best_result[0] and best_score[0] >= 1.0:
            return   # exact match already found — prune remaining tree

        try:
            for text in _control_texts(ctrl):
                score = _match_score(query, text)

                if score >= 1.0:
                    rect = ctrl.BoundingRectangle
                    if _rect_valid(rect):
                        cx = (rect.left + rect.right) // 2
                        cy = (rect.top + rect.bottom) // 2
                        best_result[0] = (cx, cy, 1.0)
                        best_score[0]  = 1.0
                        if publish:
                            publish(best_result[0])
                        return   # stop this branch — propagate up

                elif score >= threshold and score > best_score[0]:
                    rect = ctrl.BoundingRectangle
                    if _rect_valid(rect):
                        cx = (rect.left + rect.right) // 2
                        cy = (rect.top + rect.bottom) // 2
                        best_result[0] = (cx, cy, round(score * 0.90, 3))
                        best_score[0]  = score
                        if publish:
                            publish(best_result[0])

            for child in ctrl.GetChildren():
                _walk(child, depth + 1)
                if best_score[0] >= 1.0:
                    return   # exact match propagation

        except Exception:
            pass   # UIA elements can disappear mid-walk (window closed, etc.)

    _walk(root, 0)
    return best_result[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _control_texts(ctrl) -> list:
    """Return all useful text strings exposed by a UIA control:
      Name      — the accessible name shown visually (button label, menu item, etc.)
      Value     — current value of a text field or combo box
    Both are stripped and lowercased. Empty strings are excluded.
    """
    texts = []
    try:
        name = (ctrl.Name or "").strip().lower()
        if name:
            texts.append(name)
    except Exception:
        pass
    try:
        val = ctrl.GetValuePattern().Value
        if val and val.strip():
            texts.append(val.strip().lower())
    except Exception:
        pass
    return texts


def _match_score(query: str, text: str) -> float:
    """Score how well `query` matches `text` (both already lower-stripped)."""
    if query == text:
        return 1.0
    if len(query) >= 3 and query in text:
        return 0.93
    if len(text) >= 3 and text in query:
        return 0.88
    len_ratio = min(len(query), len(text)) / max(len(query), len(text), 1)
    if len_ratio < 0.35:
        return 0.0
    return difflib.SequenceMatcher(None, query, text).ratio()


def _rect_valid(rect) -> bool:
    """True if the bounding rectangle is non-empty and on-screen."""
    return (
        rect.right > rect.left
        and rect.bottom > rect.top
        and rect.right > 0
        and rect.bottom > 0
    )


# Role/descriptor words that appear in planning targets but are not
# part of the visible UI element name — strip them before matching.
_ROLE_TOKENS = frozenset({
    "the", "a", "an", "button", "btn", "menu", "bar", "tab", "panel",
    "icon", "link", "field", "box", "input", "area", "window", "dialog",
    "click", "open", "close", "toggle", "select", "choose", "find",
    "at", "on", "in", "of", "for", "with", "to", "from", "this",
    "current", "active", "focused",
})


def _strip_roles(text: str) -> str:
    tokens = text.lower().split()
    core = [t for t in tokens if t not in _ROLE_TOKENS]
    return " ".join(core) if core else text
