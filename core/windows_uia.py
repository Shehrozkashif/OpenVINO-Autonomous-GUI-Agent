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
import os
import threading

from loguru import logger

_uia = None          # lazy-loaded uiautomation module reference
_available: bool | None = None
_disabled_logged = False     # AGENT_DISABLE_UIA notice printed once


# ── Module availability ───────────────────────────────────────────────────────

def _load() -> bool:
    global _uia, _available
    # Kill switch: AGENT_DISABLE_UIA=1 turns off every UIA path (Stage 0
    # grounding, planner control hints, set_value, hit-testing) so the
    # OCR/VLM stages can be exercised on machines where UIA would win.
    if os.environ.get("AGENT_DISABLE_UIA", "").strip().lower() in ("1", "true", "yes"):
        global _disabled_logged
        if not _disabled_logged:
            _disabled_logged = True
            logger.info("[UIA] disabled via AGENT_DISABLE_UIA — OCR/VLM stages take over")
        return False
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
    """True if the uiautomation backend loaded — i.e. UIA grounding can be used.

    Set AGENT_DISABLE_UIA=1 to force this off, e.g. to exercise the OCR/VLM
    grounding stages on machines where UIA would otherwise always win.
    """
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


# ── Native batched tree search ────────────────────────────────────────────────
# The recursive GetChildren() walk pays one cross-process COM round trip per
# node. WebView2/Electron accessibility trees (new Outlook, Teams, Edge,
# VS Code…) run to thousands of nodes, so the walk times out while still
# inside the window chrome — the agent then sees only title-bar controls
# ('System', 'Minimize'…) and never the app's real buttons. FindAll executes
# the traversal inside the UIA engine and returns matches in a single batch.

_PROP_BOUNDING_RECT = 30001
_PROP_CONTROL_TYPE = 30003
_PROP_NAME = 30005
_PROP_IS_ENABLED = 30010
_PROP_IS_OFFSCREEN = 30022
_TREESCOPE_DESCENDANTS = 4

_TYPE_IDS: dict = {}    # "ButtonControl" <-> 50000, both directions


def _type_id_maps() -> dict:
    if not _TYPE_IDS:
        ct = getattr(_uia, "ControlType", None)
        for tname in _INTERACTIVE_CONTROL_TYPES:
            tid = getattr(ct, tname, None) if ct else None
            if tid is not None:
                _TYPE_IDS[tname] = tid
                _TYPE_IDS[tid] = tname
    return _TYPE_IDS


def _raw_iuia():
    """The IUIAutomation COM interface behind the uiautomation package."""
    sub = getattr(_uia, "uiautomation", _uia)
    return sub._AutomationClient.instance().IUIAutomation


def _control_from_element(element):
    """Wrap a raw IUIAutomationElement back into a package Control object."""
    try:
        sub = getattr(_uia, "uiautomation", _uia)
        return sub.Control.CreateControlFromElement(element)
    except Exception:
        return None


def _native_interactive_elements(root, max_items: int = 400) -> list | None:
    """All interactive descendants of `root` via ONE native FindAll call.

    Returns [(name, control_type_name, (l, t, r, b), is_offscreen, is_enabled,
    element)], or None when the raw COM plumbing is unavailable — callers then
    fall back to the recursive walk. MUST run on a COM-initialized thread.
    """
    try:
        iuia = _raw_iuia()
        ids = _type_id_maps()
        cond = None
        for tname in sorted(_INTERACTIVE_CONTROL_TYPES):
            tid = ids.get(tname)
            if tid is None:
                continue
            c = iuia.CreatePropertyCondition(_PROP_CONTROL_TYPE, tid)
            cond = c if cond is None else iuia.CreateOrCondition(cond, c)
        element = getattr(root, "Element", None)
        if cond is None or element is None:
            return None
        # Cache request = element properties arrive in the same batch;
        # otherwise every property read below is its own cross-process call.
        try:
            cr = iuia.CreateCacheRequest()
            for pid in (_PROP_BOUNDING_RECT, _PROP_CONTROL_TYPE,
                        _PROP_NAME, _PROP_IS_OFFSCREEN, _PROP_IS_ENABLED):
                cr.AddProperty(pid)
            found = element.FindAllBuildCache(_TREESCOPE_DESCENDANTS, cond, cr)
            cached = True
        except Exception:
            found = element.FindAll(_TREESCOPE_DESCENDANTS, cond)
            cached = False
        out = []
        for i in range(min(found.Length, max_items)):
            try:
                el = found.GetElement(i)
                if cached:
                    name, ctype = el.CachedName, el.CachedControlType
                    r, off = el.CachedBoundingRectangle, el.CachedIsOffscreen
                    enabled = el.CachedIsEnabled
                else:
                    name, ctype = el.CurrentName, el.CurrentControlType
                    r, off = el.CurrentBoundingRectangle, el.CurrentIsOffscreen
                    enabled = el.CurrentIsEnabled
                out.append((
                    (name or "").strip(),
                    ids.get(ctype, ""),
                    (r.left, r.top, r.right, r.bottom),
                    bool(off),
                    bool(enabled),
                    el,
                ))
            except Exception:
                continue   # elements can vanish mid-read
        return out
    except Exception as e:
        logger.debug(f"[UIA] Native FindAll unavailable: {e}")
        return None


def _rect_tuple_valid(rect: tuple) -> bool:
    left, top, right, bottom = rect
    return right > left and bottom > top and right > 0 and bottom > 0


def _element_value_of(el) -> str | None:
    """Current text of a raw element via ValuePattern, or None when the
    element exposes no readable value. "" is a real answer: field is empty.
    """
    try:
        ctrl = _control_from_element(el)
        if ctrl is None:
            return None
        val = ctrl.GetValuePattern().Value
        if not isinstance(val, str):
            return None
        return " ".join(val.split())[:40]
    except Exception:
        return None


def _native_best_match(root, query: str, threshold: float):
    """Best interactive match for `query` from the native element batch.

    Same scoring/confidence rules as the interactive tier of _walk_and_match.
    Returns (x, y, confidence), or None when nothing scores above threshold
    OR the native plumbing is unavailable (caller falls back to the walk).
    """
    elems = _native_interactive_elements(root)
    if elems is None:
        return None
    best, best_score = None, 0.0
    for name, _tname, rect, offscreen, enabled, el in elems:
        # Disabled controls are never click candidates: clicking them is
        # provably inert (live: a grayed 'Save' ate 10 minutes of retries).
        if offscreen or not enabled or not name or not _rect_tuple_valid(rect):
            continue
        score = _match_score(query, name.lower())
        if score < threshold or score <= best_score:
            continue
        best, best_score = (rect, el), score
        if score >= 1.0:
            break
    if best is None:
        return None
    rect, el = best
    x, y = (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
    ctrl = _control_from_element(el)
    if ctrl is not None:
        try:
            x, y = _click_point(ctrl, ctrl.BoundingRectangle)
        except Exception:
            pass
    conf = 1.0 if best_score >= 1.0 else round(best_score * 0.90, 3)
    return x, y, conf


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

            # Fast path — foreground window only. Native batch first (reaches
            # WebView2/Electron content the recursive walk can't); the walk
            # remains as fallback and adds the decorative-text tier.
            fg = _uia.GetForegroundControl()
            r = _native_best_match(fg, query, fuzzy_threshold)
            if r:
                result[0] = r
                return
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

            # All same-process windows, not just the foreground one — a picker
            # popup or dropdown the user can plainly see must be in the
            # planner's list too.
            any_native = False
            for root in _same_process_roots():
                if len(out) >= max_elements:
                    break
                native = _native_interactive_elements(root)
                if native is None:
                    continue
                any_native = True
                for name, tname, rect, offscreen, enabled, _el in native:
                    if (
                        not name
                        or offscreen
                        or len(name) > 60
                        or not any(ch.isalnum() for ch in name)
                        or not _rect_tuple_valid(rect)
                    ):
                        continue
                    key = (name.lower(), tname)
                    if key not in seen:
                        seen.add(key)
                        # "(disabled)" is load-bearing: it tells the planner
                        # WHY a button won't respond (fill required fields
                        # first) instead of letting it hammer a gray button.
                        label = tname.replace("Control", "")
                        # Text fields carry their CURRENT content — without it
                        # the planner/goal-check cannot tell a filled form from
                        # an empty one (seen live: 'Start time' silently left
                        # at 8:00 AM while the plan moved on to Save).
                        if tname in (
                            "EditControl", "ComboBoxControl", "SpinnerControl"
                        ):
                            val = _element_value_of(_el)
                            if val is not None:
                                label += f" = '{val}'"
                        if not enabled:
                            label += " (disabled)"
                        out.append((name, label))
                        if len(out) >= max_elements:
                            break
            if any_native:
                result[0] = out
                return

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


def control_type_at_point(x: int, y: int, timeout_s: float = 1.0) -> str | None:
    """ControlTypeName of the UIA element under screen pixel (x, y).

    Ground truth for "what did the click actually land on" — e.g. the
    caret-placement rescue must only fire when the clicked pixel IS a text
    control, not merely inside some huge focused WebView document whose rect
    covers the whole window. Returns None when UIA is unavailable/timed out.
    """
    if not _load():
        return None

    result: list = [None]

    def _get():
        _com = _thread_com_init()
        try:
            ctrl = _uia.ControlFromPoint(int(x), int(y))
            if ctrl is not None:
                result[0] = ctrl.ControlTypeName
        except Exception as e:
            logger.debug(f"[UIA] ControlFromPoint({x},{y}) error: {e}")
        finally:
            del _com

    t = threading.Thread(target=_get, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


def _chain_matches_target(names: list, target: str) -> bool:
    """Does any name in a hit-test ancestor chain plausibly BE `target`?

    True means the point genuinely belongs to the target control (or a
    named parent/child of it). All-empty chains also return True: web
    canvases expose nameless nodes and "can't tell" must never be treated
    as proof of occlusion.
    """
    t = _norm_text(target)
    if not t:
        return True
    named = [n for n in (_norm_text(n) for n in names) if n]
    if not named:
        return True
    for n in named:
        if t in n or n in t or _match_score(t, n) >= 0.75:
            return True
    return False


def covering_element(x: int, y: int, target: str, timeout_s: float = 1.5) -> str | None:
    """Name of the element that actually owns screen point (x, y) when that
    point does NOT belong to the control named `target` — i.e. an overlay or
    dialog is physically on top and a click there can never reach the target.

    Ground truth via hit-testing: ControlFromPoint returns what a click at
    (x, y) would really land on; the ancestor walk covers targets whose
    accessible name lives on a parent. Returns None when the point belongs
    to the target, when nothing named is readable there, or when UIA is
    unavailable — callers may only act on a positive answer.
    """
    if not _load():
        return None

    result: list = [None]

    def _get():
        _com = _thread_com_init()
        try:
            ctrl = _uia.ControlFromPoint(int(x), int(y))
            names, hops = [], 0
            while ctrl is not None and hops < 8:
                try:
                    names.append((ctrl.Name or "").strip())
                except Exception:
                    names.append("")
                try:
                    ctrl = ctrl.GetParentControl()
                except Exception:
                    break
                hops += 1
            if not _chain_matches_target(names, target):
                result[0] = next(n for n in names if n)[:80]
        except Exception as e:
            logger.debug(f"[UIA] covering_element({x},{y}) error: {e}")
        finally:
            del _com

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


# ── Structured-control actions (UIA patterns) ────────────────────────────────
# Deterministic, app-agnostic manipulation of form controls through the same
# accessibility tree used for grounding. Where a mouse click on a dropdown or
# date picker is a pixel guess, ValuePattern / SelectionItemPattern /
# ExpandCollapsePattern / InvokePattern act on the control object itself and
# can be verified by reading the control state straight back — no OCR, no VLM.


def _same_process_roots():
    """Search roots: the foreground window plus every other visible top-level
    window OWNED BY THE SAME PROCESS.

    Popups (date pickers, attendee-suggestion dropdowns, confirmation dialogs)
    routinely open as sibling top-level windows — a foreground-only search is
    blind to their controls even though the user sees them. General across
    apps: same-process windows are, by construction, part of the app under
    automation.
    """
    fg = _uia.GetForegroundControl()
    roots = [fg]
    try:
        fg_pid = fg.ProcessId
        for win in _uia.GetRootControl().GetChildren():
            try:
                if win.Handle == fg.Handle or win.ProcessId != fg_pid:
                    continue
                rect = win.BoundingRectangle
                if (rect.right - rect.left) < 10 or (rect.bottom - rect.top) < 10:
                    continue
                roots.append(win)
            except Exception:
                continue
    except Exception:
        pass
    return roots


def _find_control(query: str, timeout_s: float, control_types: frozenset | None = None):
    """Locate the best-matching UIA control object for `query`.

    Same matching rules as find_element(), but returns the live control object
    (valid only on the calling thread's COM context) instead of coordinates.
    MUST be called from a thread that has already initialized COM.
    """
    query = _strip_roles(query).lower().strip()
    if not query:
        return None

    # Native batch first — the recursive walk below cannot reach controls
    # buried in WebView2/Electron trees. Offscreen elements are kept: UIA
    # patterns (Invoke/Value) work on them where a pixel click cannot.
    best_el, best_native = None, 0.0
    for root in _same_process_roots():
        native = _native_interactive_elements(root)
        if native is None:
            continue
        for name, tname, _rect, _offscreen, enabled, el in native:
            if not name or not enabled:
                continue
            if control_types is not None and tname not in control_types:
                continue
            score = _match_score(query, name.lower())
            if score >= 0.65 and score > best_native:
                best_el, best_native = el, score
        if best_native >= 1.0:
            break
    if best_el is not None:
        ctrl = _control_from_element(best_el)
        if ctrl is not None:
            return ctrl

    best: list = [None, 0.0]   # control, score

    def _walk(ctrl, depth: int, max_depth: int):
        if depth > max_depth or best[1] >= 1.0:
            return
        try:
            type_ok = control_types is None or ctrl.ControlTypeName in control_types
            if type_ok:
                for text in _control_texts(ctrl):
                    score = _match_score(query, text)
                    if score > best[1] and score >= 0.65:
                        rect = ctrl.BoundingRectangle
                        if _rect_valid(rect):
                            best[0], best[1] = ctrl, score
                            if score >= 1.0:
                                return
            for child in ctrl.GetChildren():
                _walk(child, depth + 1, max_depth)
                if best[1] >= 1.0:
                    return
        except Exception:
            pass   # controls can vanish mid-walk

    _walk(_uia.GetForegroundControl(), 0, 12)
    return best[0]


def _run_uia_action(label: str, fn, timeout_s: float) -> bool:
    """Run `fn` (a COM-touching callable returning bool) on a daemon thread with
    per-thread COM init and a hard timeout, mirroring the query helpers above.
    Returns False on timeout or any exception — callers fall back to the
    click/type path, so a UIA miss can never block a task.
    """
    if not _load():
        return False
    result = [False]

    def _act():
        _com = _thread_com_init()
        try:
            result[0] = bool(fn())
        except Exception as e:
            logger.debug(f"[UIA] {label} failed: {e}")
        finally:
            del _com

    t = threading.Thread(target=_act, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    return result[0]


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def set_element_value(target: str, value: str, timeout_s: float = 3.0) -> bool:
    """Set a text control's content via ValuePattern and verify by read-back.

    Ground truth end to end: the value lands in the control object itself
    (no focus race, no keystroke loss) and success means the control now
    reports that exact content. Returns False when the control is missing,
    read-only, or the read-back does not match — callers then fall back to
    click+type.
    """
    def _do():
        ctrl = _find_control(target, timeout_s, frozenset({
            "EditControl", "ComboBoxControl", "DocumentControl",
            "SpinnerControl",
        }))
        if ctrl is None:
            logger.debug(f"[UIA] set_value: no editable control named '{target}'")
            return False
        vp = ctrl.GetValuePattern()
        if getattr(vp, "IsReadOnly", False):
            logger.debug(f"[UIA] set_value: '{target}' is read-only")
            return False
        vp.SetValue(value)
        readback = vp.Value or ""
        ok = _norm_text(readback) == _norm_text(value)
        if ok:
            logger.info(f"[UIA] set_value '{target}' = '{value[:60]}' (verified)")
        else:
            logger.debug(
                f"[UIA] set_value read-back mismatch on '{target}': "
                f"'{readback[:60]}' != '{value[:60]}'"
            )
        return ok

    return _run_uia_action(f"set_value('{target}')", _do, timeout_s)


def select_option(target: str, option: str, timeout_s: float = 4.0) -> bool:
    """Select `option` inside the dropdown/list/tab control named `target`.

    Mechanism (general — works on any app exposing standard UIA patterns):
      1. Find the container control by name.
      2. If it is collapsed (ComboBox, expander), expand it via
         ExpandCollapsePattern so its items materialize in the tree.
      3. Find the item matching `option` inside it and select it via
         SelectionItemPattern (or InvokePattern as fallback).
      4. Verify: the item reports IsSelected, or the container's value now
         reads as the option.
    Returns False on any miss so the caller can fall back to click-based steps.
    """
    def _do():
        ctrl = _find_control(target, timeout_s, None)
        if ctrl is None:
            return False

        # Expand collapsed containers so items exist in the tree.
        expanded_here = False
        try:
            ecp = ctrl.GetExpandCollapsePattern()
            if getattr(ecp, "ExpandCollapseState", 1) == 0:   # 0 = Collapsed
                ecp.Expand()
                expanded_here = True
                import time as _t
                _t.sleep(0.25)   # let the popup list materialize
        except Exception:
            pass

        want = _norm_text(option)
        item_best: list = [None, 0.0]
        seen_items: list = []   # for the no-match log — the planner's ground truth

        def _walk_items(c, depth: int):
            if depth > 8 or item_best[1] >= 1.0:
                return
            try:
                if c.ControlTypeName in (
                    "ListItemControl", "TreeItemControl", "TabItemControl",
                    "RadioButtonControl", "MenuItemControl",
                ):
                    name = _norm_text(c.Name)
                    if name and len(seen_items) < 15:
                        seen_items.append(name[:40])
                    score = _match_score(want, name)
                    if score > item_best[1] and score >= 0.65:
                        item_best[0], item_best[1] = c, score
                        if score >= 1.0:
                            return
                for ch in c.GetChildren():
                    _walk_items(ch, depth + 1)
                    if item_best[1] >= 1.0:
                        return
            except Exception:
                pass

        _walk_items(ctrl, 0)
        if item_best[0] is None and expanded_here:
            # ComboBox popups can be hosted in a separate top-level window —
            # search the whole foreground tree once more.
            _walk_items(_uia.GetForegroundControl(), 0)
        if item_best[0] is None:
            # Virtualized popups (WebView2 date/time combos) render their
            # items OUTSIDE the accessibility tree — there is nothing to
            # select, ever. But these combos accept a typed value: write it
            # through ValuePattern, the same ground truth set_element_value
            # uses (live: 'Start time' listed no items for 14 minutes while
            # a direct '3:00 PM' write verified instantly).
            try:
                vp = ctrl.GetValuePattern()
                if not getattr(vp, "IsReadOnly", False):
                    vp.SetValue(option)
                    readback = _norm_text(vp.Value or "")
                    if want and (want in readback or readback in want):
                        if expanded_here:
                            try:
                                ctrl.GetExpandCollapsePattern().Collapse()
                            except Exception:
                                pass
                        logger.info(
                            f"[UIA] select '{option}' in '{target}' (via "
                            f"ValuePattern — no list items in the tree)"
                        )
                        return True
            except Exception:
                pass
            # Name what WAS there: 'GST' matches nothing when the app lists
            # '(UTC+04:00) Abu Dhabi, Muscat' — the mismatch must reach the
            # PLANNER (via pop_select_miss), not just the log, so it can pick
            # the app's real label or scroll the list for more.
            global _last_select_miss
            _last_select_miss = {
                "target": target, "option": option, "items": list(seen_items),
            }
            logger.info(
                f"[UIA] select: no option matching '{option}' in '{target}'"
                + (f"; visible items: {', '.join(seen_items)}"
                   if seen_items else " (no items materialized)")
            )
            return False
        item = item_best[0]

        try:
            item.GetSelectionItemPattern().Select()
        except Exception:
            try:
                item.GetInvokePattern().Invoke()
            except Exception:
                return False

        # Verify selection against the tree — item state or container value.
        try:
            if item.GetSelectionItemPattern().IsSelected:
                logger.info(f"[UIA] select '{option}' in '{target}' (verified)")
                return True
        except Exception:
            pass
        try:
            if want in _norm_text(ctrl.GetValuePattern().Value):
                logger.info(f"[UIA] select '{option}' in '{target}' (value verified)")
                return True
        except Exception:
            pass
        # Selection fired without a readable state — count the action as done;
        # the orchestrator's reflection still verifies the visible outcome.
        logger.info(f"[UIA] select '{option}' in '{target}' (state unreadable)")
        return True

    return _run_uia_action(f"select('{option}' in '{target}')", _do, timeout_s)


_last_select_miss: dict | None = None


def pop_select_miss() -> dict | None:
    """The last select_option no-match details ({target, option, items}),
    cleared on read. Ground truth for the planner's failure record: the
    option the plan asked for versus the labels the app actually shows.
    """
    global _last_select_miss
    miss, _last_select_miss = _last_select_miss, None
    return miss


def focus_element(target: str, timeout_s: float = 3.0) -> bool:
    """Give keyboard focus to the control named `target` via UIA SetFocus.

    The general way into fields that expose no ValuePattern (date/time
    segments, custom web widgets): focus them through the tree, then type.
    Returns False when the control cannot be found — callers fall back to
    click-based focusing.
    """
    def _do():
        ctrl = _find_control(target, timeout_s, None)
        if ctrl is None:
            return False
        ctrl.SetFocus()
        logger.info(f"[UIA] focus '{target}' ({ctrl.ControlTypeName})")
        return True

    return _run_uia_action(f"focus('{target}')", _do, timeout_s)


def invoke_element(target: str, timeout_s: float = 3.0) -> bool:
    """Press the button/menu-item/link named `target` through UIA patterns.

    Works even when the element is occluded or off-screen (where a mouse
    click cannot land) and never mis-clicks a lookalike pixel region.

    Each action pattern is tried independently — WebView2/Electron providers
    routinely raise COM errors from Invoke on perfectly real buttons, and the
    Get<Name>Pattern helpers only exist on control classes whose spec lists
    that pattern, so a failed attempt must never abort the chain.
    GetPattern(id) is safe on every control (returns None if unsupported).
    LegacyIAccessible.DoDefaultAction is the last resort: the MSAA bridge
    presses almost anything, including web-content buttons.
    """
    def _do():
        ctrl = _find_control(target, timeout_s, frozenset({
            "ButtonControl", "MenuItemControl", "HyperlinkControl",
            "ListItemControl", "SplitButtonControl", "CheckBoxControl",
            "TabItemControl",
        }))
        if ctrl is None:
            return False
        pid = _uia.PatternId
        attempts = [
            ("Invoke", pid.InvokePattern, "Invoke"),
            ("Select", pid.SelectionItemPattern, "Select"),
            ("Toggle", pid.TogglePattern, "Toggle"),
            ("DoDefaultAction", pid.LegacyIAccessiblePattern, "DoDefaultAction"),
        ]
        if ctrl.ControlTypeName == "CheckBoxControl":
            attempts.insert(0, attempts.pop(2))   # Toggle first for checkboxes
        for name, pattern_id, method in attempts:
            try:
                pattern = ctrl.GetPattern(pattern_id)
                if pattern is None:
                    continue
                getattr(pattern, method)()
                logger.info(
                    f"[UIA] invoke '{target}' via {name} ({ctrl.ControlTypeName})"
                )
                return True
            except Exception as e:
                logger.debug(f"[UIA] {name} on '{target}' failed: {e}")
        return False

    return _run_uia_action(f"invoke('{target}')", _do, timeout_s)


# UIA control types a user can act on — used by get_interactive_elements().
_INTERACTIVE_CONTROL_TYPES = frozenset({
    "ButtonControl", "MenuItemControl", "ListItemControl", "TreeItemControl",
    "TabItemControl", "HyperlinkControl", "ComboBoxControl", "CheckBoxControl",
    "RadioButtonControl", "EditControl", "SplitButtonControl",
    "DocumentControl", "SliderControl", "SpinnerControl",
})


# ── Tree walking ──────────────────────────────────────────────────────────────

def _click_point(ctrl, rect) -> tuple[int, int]:
    """Where a click should land for this control.

    Prefer the OS-reported clickable point: the center of a wide control's
    bounding rect can sit on an overlapping child or padding, while
    GetClickablePoint() is the accessibility tree's own answer to "where do I
    click this". Falls back to rect center when the pattern is unsupported.
    """
    try:
        x, y, got = ctrl.GetClickablePoint()
        if got and x > 0 and y > 0:
            return x, y
    except Exception:
        pass
    return (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2


def _walk_and_match(
    root,
    query: str,
    threshold: float,
    max_depth: int,
    publish=None,
) -> tuple[int, int, float] | None:
    """DFS walk of a UIA subtree.  Returns the best (x, y, confidence) match or None.

    Two-tier matching: INTERACTIVE controls (buttons, menu items, list items…)
    always outrank decorative ones (text labels, images, badges). A decorative
    element that merely displays the query text — e.g. a "NEW" feature badge
    (TextControl) — must never beat the interactive control the planner means
    (the "New meeting" button, substring match). Clicking decoration is a
    no-op, which is exactly the delta=0 dead-click loop seen in live runs.
    Only an exact INTERACTIVE match stops the walk early; decorative matches
    are kept as a confidence-discounted fallback used when nothing interactive
    matches at all (some apps label a clickable region only via a text child).

    publish: optional callback invoked with each new best match as the walk
    progresses — lets the caller keep the best result found so far even when
    the walk is cut off by a timeout (deep browser/Electron trees).
    """
    best_result: list = [None]   # interactive tier
    best_score:  list = [0.0]
    passive_result: list = [None]   # decorative fallback tier
    passive_score:  list = [0.0]

    def _publish_best():
        if publish:
            publish(best_result[0] or passive_result[0])

    def _walk(ctrl, depth: int):
        if depth > max_depth:
            return
        if best_result[0] and best_score[0] >= 1.0:
            return   # exact interactive match already found — prune remaining tree

        try:
            interactive = ctrl.ControlTypeName in _INTERACTIVE_CONTROL_TYPES
            try:
                if interactive and not ctrl.IsEnabled:
                    interactive = False   # disabled → never a click candidate
            except Exception:
                pass
            for text in _control_texts(ctrl):
                score = _match_score(query, text)
                if score < threshold:
                    continue
                rect = ctrl.BoundingRectangle
                if not _rect_valid(rect):
                    continue
                if interactive:
                    if score >= 1.0:
                        best_result[0] = (*_click_point(ctrl, rect), 1.0)
                        best_score[0]  = 1.0
                        _publish_best()
                        return   # stop this branch — propagate up
                    if score > best_score[0]:
                        best_result[0] = (*_click_point(ctrl, rect),
                                          round(score * 0.90, 3))
                        best_score[0]  = score
                        _publish_best()
                elif score > passive_score[0]:
                    passive_result[0] = (*_click_point(ctrl, rect),
                                         round(score * 0.80, 3))
                    passive_score[0]  = score
                    _publish_best()

            for child in ctrl.GetChildren():
                _walk(child, depth + 1)
                if best_score[0] >= 1.0:
                    return   # exact match propagation

        except Exception:
            pass   # UIA elements can disappear mid-walk (window closed, etc.)

    _walk(root, 0)
    return best_result[0] or passive_result[0]


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
