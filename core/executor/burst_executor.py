# core/executor/burst_executor.py
"""
BurstExecutor — executes an ActionBurst: a short sequence of UI actions that
must happen quickly (context menus, rename dialogs, form submissions) without
any intermediate LLM planning or reflection calls.

Execution flow
--------------
1. Pre-ground click targets that are visible BEFORE the burst begins (e.g.,
   permanent on-screen elements).  Steps that carry explicit "x,y" pixel
   coordinates in their value field bypass grounding entirely.

   Transient targets (context-menu items, submenu items) that don't exist on
   screen yet are NOT pre-grounded.  Instead they are marked for late grounding
   and resolved at execution time, just before the step runs — after the
   preceding action (right-click, click) has made them visible.

2. Execute each step in order.  Inter-step delay is action-type-aware:
     right_click  →  350 ms   (wait for context menu to render)
     click        →  300 ms   (wait for submenu / dialog to open)
     key_press    →  300 ms   (wait for submenu animation to settle)
     other        →   80 ms   (default)

3. Optionally verify the FINAL state with a single reflection call
   (verify_at_end=True).  Intermediate steps are not reflected — that is the
   whole point of a burst.

detect_burst()
--------------
Pattern-matches a SubTask description and returns an ActionBurst if one of
the known fast-sequence patterns is detected, else None.  The caller (the
orchestrator) tries the burst first; on failure it falls back to the normal
planning loop unchanged.

Design principle
----------------
Burst steps should NOT embed hardcoded OS shortcuts or key sequences that
are locale- or version-specific.  Click steps targeting menu items are
grounded visually (OCR/UIA) at runtime so the burst works across Windows
versions without brittle shortcut assumptions.
"""
import re
import time
from typing import Optional

from loguru import logger

from core.protocols.a2a import ActionBurst, ActionStep, BurstResult, SubTask

# 80 ms between steps — enough for the UI to render, far below LLM latency
_INTER_STEP_DELAY_S: float = 0.08


# ── BurstExecutor ─────────────────────────────────────────────────────────────

class BurstExecutor:
    """Execute an ActionBurst using the supplied grounder, actor, and reflector."""

    def __init__(self, grounder, actor, reflector=None,
                 inter_step_delay_s: float = _INTER_STEP_DELAY_S):
        self.grounder = grounder
        self.actor    = actor
        self.reflector = reflector
        self.delay    = inter_step_delay_s

    def run(self, burst: ActionBurst) -> BurstResult:
        """
        Execute burst.steps in order.

        Returns BurstResult with success=False as soon as any phase fails so
        the orchestrator can fall back to the planning loop without side effects
        propagating from partial execution.
        """
        # ── Phase 1: pre-ground targets visible before the burst begins ────────
        # Steps with explicit "x,y" in their value field bypass grounding.
        # Targets that are not yet visible (e.g., context-menu items) are flagged
        # for late grounding — they will be resolved during Phase 2, just before
        # their step executes, once the preceding interaction has made them visible.
        coords: dict = {}        # step_index → (x, y)
        late_ground: set = set() # step indices that need grounding at runtime

        for i, step in enumerate(burst.steps):
            if step.action_type not in ("click", "right_click", "double_click"):
                continue

            # Explicit pixel coordinates in value field bypass grounding entirely.
            if step.value and "," in step.value:
                try:
                    parts = step.value.split(",", 1)
                    px, py = int(float(parts[0])), int(float(parts[1]))
                    coords[i] = (px, py)
                    logger.debug(
                        f"[BURST] Explicit coords for step {i}"
                        + (f" '{step.target}'" if step.target else "")
                        + f": ({px},{py})"
                    )
                    continue
                except (ValueError, IndexError):
                    pass

            if not step.target:
                continue  # no target and no explicit coords — actor uses defaults

            # Use fast grounding (UIA + OCR only) during pre-grounding.
            # Context-menu items don't exist yet; falling to Stage 2 (VLM) would
            # block 30-50 s per not-found.  Check the class dict (not instance)
            # so MagicMock auto-created attributes never shadow the real method.
            if "ground_fast" in type(self.grounder).__dict__:
                result = self.grounder.ground_fast(step.target)
            else:
                result = self.grounder.ground(step.target)
            if not result.found or result.confidence < self.grounder.min_confidence:
                # Target not visible yet. If any earlier step will interact with
                # the UI (right_click / click), this step likely targets a transient
                # element that will appear after those interactions.
                earlier_interaction = any(
                    s.action_type in ("right_click", "click")
                    for s in burst.steps[:i]
                )
                if earlier_interaction:
                    late_ground.add(i)
                    logger.debug(
                        f"[BURST] Step {i} '{step.target}' not visible yet — "
                        "will ground after preceding interaction"
                    )
                else:
                    logger.info(
                        f"[BURST] Pre-ground failed for '{step.target}' "
                        f"(conf={result.confidence:.2f}) — aborting burst"
                    )
                    return BurstResult(
                        success=False,
                        failed_at_step=i,
                        reason=(
                            f"grounding failed for '{step.target}' "
                            f"(conf={result.confidence:.2f})"
                        ),
                    )
            else:
                coords[i] = (result.x, result.y)
                logger.debug(
                    f"[BURST] Pre-grounded step {i}: '{step.target}' -> "
                    f"({result.x},{result.y})"
                )

        # ── Phase 2: execute with action-aware inter-step delay ───────────────
        deadline = time.time() + burst.timeout_ms / 1000
        for i, step in enumerate(burst.steps):
            if time.time() > deadline:
                return BurstResult(
                    success=False,
                    failed_at_step=i,
                    reason=f"burst timeout ({burst.timeout_ms} ms) exceeded before step {i}",
                )

            x, y = coords.get(i, (None, None))

            # Late grounding: element becomes visible only after preceding steps.
            if i in late_ground and step.target:
                result = self.grounder.ground(step.target)
                if not result.found or result.confidence < self.grounder.min_confidence:
                    return BurstResult(
                        success=False,
                        failed_at_step=i,
                        reason=(
                            f"late grounding failed for '{step.target}' "
                            f"(conf={result.confidence:.2f})"
                        ),
                    )
                x, y = result.x, result.y
                logger.info(
                    f"[BURST] Late-grounded step {i}: '{step.target}' -> ({x},{y})"
                )

            ok = self.actor.execute(step, x=x, y=y)
            logger.info(
                f"[BURST] Step {i + 1}/{len(burst.steps)}: "
                f"[{step.action_type}] {step.description}"
            )
            if not ok:
                return BurstResult(
                    success=False,
                    failed_at_step=i,
                    reason=(
                        f"actor.execute failed for step {i} "
                        f"({step.action_type} "
                        f"'{step.target or step.key or step.value}')"
                    ),
                )

            if i < len(burst.steps) - 1:
                # Action-aware inter-step delay:
                #   right_click →  350 ms — wait for context menu to render
                #   click       →  300 ms — wait for submenu / dialog to open
                #   key_press   →  300 ms — wait for submenu animation to settle
                #   other       →   80 ms — default
                if step.action_type == "right_click":
                    step_delay = 0.35
                elif step.action_type in ("click", "key_press", "double_click"):
                    step_delay = 0.30
                else:
                    step_delay = self.delay
                time.sleep(step_delay)

        # ── Phase 3: optional final reflection ────────────────────────────────
        if burst.verify_at_end and self.reflector is not None:
            final_step = burst.steps[-1]
            try:
                reflection = self.reflector.verify(final_step, wait_s=1.0)
                if not reflection.success:
                    return BurstResult(
                        success=False,
                        failed_at_step=len(burst.steps) - 1,
                        reason=(
                            reflection.error_description
                            or "final burst reflection failed"
                        ),
                    )
            except Exception as exc:
                # Infrastructure failure in reflection — treat burst as succeeded
                # (same policy as the main orchestrator loop)
                logger.warning(f"[BURST] Reflection error: {exc} — treating burst as succeeded")

        return BurstResult(success=True, failed_at_step=None, reason="burst completed")


# ── Burst detection ───────────────────────────────────────────────────────────

def detect_burst(subtask: SubTask) -> ActionBurst | None:
    """
    Analyse a SubTask description and return an ActionBurst if one of the
    known fast-sequence patterns is recognised, else None.

    Patterns (checked in priority order, first match wins):
      1. Create / new folder   — right_click(desktop) → click(New) → click(Folder)
      2. Right-click + select  — right_click(target) → click(menu_item)
      3. Type + press Enter    — type(text) → key_press(enter)
      4. Select-all + type     — hotkey(ctrl+a) → type(text)
    """
    orig = subtask.description.strip()       # original case — used for name captures
    desc = orig.lower()
    sid  = subtask.id

    # ── Pattern 1b: full 5-step folder creation sequence ─────────────────────
    # Checked FIRST so it wins over the simpler keyword patterns below.
    # Matches compound instructions like:
    #   "Right click on the desktop, click New, click Folder, type Foo, press Enter."
    if re.search(
        r"right[- ]click.*?desktop.*?click\s+new.*?click\s+folder.*?"
        r"type\s+['\"]?\w+['\"]?.*?enter",
        desc,
    ):
        m = re.search(r"type\s+['\"]?(\w+)['\"]?.*?enter", orig, re.IGNORECASE)
        if m:
            return _folder_creation_burst(sid, m.group(1))

    # ── Pattern 1c: "create a folder called/named X" ─────────────────────────
    # Also checked before 1a so a named variant beats the unnamed 3-step burst.
    m = re.search(
        r"create\s+(?:a\s+)?(?:new\s+)?folder\s+(?:called|named)\s+['\"]?(\w+)['\"]?",
        orig, re.IGNORECASE,
    )
    if not m:
        m = re.search(r"new\s+folder\s+named\s+['\"]?(\w+)['\"]?", orig, re.IGNORECASE)
    if not m:
        m = re.search(r"right[- ]click\s+new\s+folder\s+['\"]?(\w+)['\"]?", orig, re.IGNORECASE)
    if m:
        return _folder_creation_burst(sid, m.group(1))

    # ── Pattern 1a: create new folder (short keywords, no explicit name) ─────
    if _has_any(desc, [
        "new folder", "create folder", "create a folder",
        "create new folder", "make a folder", "make folder",
    ]):
        return _folder_creation_burst(sid, "NewFolder")

    # ── Pattern 2: explicit right-click + menu item ───────────────────────────
    m = re.search(
        r"right[- ]click\s+(?:on\s+)?(.+?)"
        r"\s+(?:and\s+(?:then\s+)?(?:click|select|choose)"
        r"|,\s*(?:click|select))"
        r"\s+['\"]?(.+?)['\"]?$",
        desc,
    )
    if m:
        rc_target   = m.group(1).strip()
        menu_item   = m.group(2).strip()
        return ActionBurst(
            steps=[
                _s(sid, 1, "right_click", target=rc_target,
                   desc=f"Right-click {rc_target}",
                   verif="Context menu appeared"),
                _s(sid, 2, "click", target=menu_item,
                   desc=f"Click '{menu_item}' in context menu",
                   verif=f"'{menu_item}' action executed"),
            ],
            verify_at_end=True,
            timeout_ms=4000,
        )

    # ── Pattern 3: type text + press Enter ───────────────────────────────────
    m = re.search(
        r"type\s+['\"]?(.+?)['\"]?\s+(?:and\s+)?(?:press|hit|then)\s+enter",
        desc,
    )
    if m:
        text = m.group(1).strip()
        return ActionBurst(
            steps=[
                _s(sid, 1, "type", value=text,
                   desc=f"Type '{text}'",
                   verif=f"'{text}' visible in active input field"),
                _s(sid, 2, "key_press", key="enter",
                   desc="Press Enter to confirm",
                   verif="Input confirmed — dialog dismissed or command ran"),
            ],
            verify_at_end=False,
            timeout_ms=3000,
        )

    # ── Pattern 4: select-all + type ─────────────────────────────────────────
    m = re.search(
        r"(?:ctrl\+a|select all).*?type\s+['\"]?(.+?)['\"]?(?:\s|$)",
        desc,
    )
    if m:
        text = m.group(1).strip()
        return ActionBurst(
            steps=[
                _s(sid, 1, "hotkey", key="ctrl+a",
                   desc="Select all existing text",
                   verif="All text selected"),
                _s(sid, 2, "type", value=text,
                   desc=f"Type '{text}'",
                   verif=f"'{text}' visible in field"),
            ],
            verify_at_end=False,
            timeout_ms=3000,
        )

    return None


# ── Public instruction-level helper ──────────────────────────────────────────

def detect_burst_from_instruction(instruction: str) -> ActionBurst | None:
    """
    Match the full raw user instruction (before LLM routing) against burst patterns.

    Lets the orchestrator skip the router entirely for known compound-action
    sequences such as "right click → New → Folder → type name → Enter".
    Returns None when the instruction doesn't match any pattern.

    SAFETY (Fix C1): a burst replaces the ENTIRE instruction with one synthetic
    subtask and skips the router. If the instruction also contains other work
    ("create a folder X, then open it and add a file"), bursting would silently
    drop everything after the matched fragment. So we only allow an
    instruction-level burst when the matched pattern consumes essentially the
    whole instruction. Anything with additional clauses falls through to the
    router, which decomposes it properly; per-subtask burst detection can still
    fire later for the parts that genuinely are bursts.
    """
    class _Stub:
        id = 1   # synthetic subtask id — steps get subtask_id=1, consistent with orchestrator
        description = instruction

    burst = detect_burst(_Stub())  # type: ignore[arg-type]
    if burst is None:
        return None

    if _instruction_has_extra_clauses(instruction):
        logger.info(
            "[BURST] Instruction-level burst matched but the instruction contains "
            "additional clauses — deferring to the router for full decomposition"
        )
        return None
    return burst


# Connectives that signal a second, separate step follows the burst fragment.
# NOTE: the "and <verb>" list deliberately excludes click/select/type — those
# verbs are part of the burst patterns themselves (e.g. "right-click X and click
# New"). It only lists verbs that introduce genuinely new work (launching an app,
# navigating, saving), which a burst would silently drop.
_SEQUENCE_CONNECTIVES = re.compile(
    r"\b(?:then|after that|afterwards|next|followed by|and then|"
    r"and (?:open|launch|run|start|navigate|go to|browse|search|save|close|"
    r"write|email|send|install|compose))\b",
    re.IGNORECASE,
)


def _instruction_has_extra_clauses(instruction: str) -> bool:
    """Heuristic: True when the instruction looks like it has work beyond a single burst.

    Conservative — only treats clear sequencing connectives (then/next/and <verb>)
    or multiple sentences as 'extra'. A trailing simple clause like
    "create a folder called test" stays a burst.
    """
    text = instruction.strip()
    # More than one sentence worth of content
    sentences = [s for s in re.split(r"[.;\n]+", text) if s.strip()]
    if len(sentences) > 1:
        return True
    return bool(_SEQUENCE_CONNECTIVES.search(text))


# ── Private helpers ───────────────────────────────────────────────────────────

def _has_any(text: str, keywords: list) -> bool:
    return any(k in text for k in keywords)


def _folder_creation_burst(sid: int, folder_name: str) -> ActionBurst:
    """Return a 5-step visual burst for desktop folder creation.

    "New" and "Folder" are located by OCR at execution time (late grounding)
    rather than via hardcoded keyboard shortcuts.  This is robust across Windows
    versions, locales, and themes — the burst behaves like the planning loop but
    without the 10-second LLM delays between steps that would cause context menus
    to close.

    The right_click step uses an explicit physical screen-center coordinate so
    the click lands on the blank desktop background regardless of what UIA
    returns for "desktop" (which can be a taskbar element on some Windows builds).
    """
    import platform
    safe_x, safe_y = 960, 500  # fallback: assume 1920x1080
    if platform.system() == "Windows":
        try:
            import ctypes
            # GetDeviceCaps(DESKTOPHORZRES/DESKTOPVERTRES) always returns physical
            # pixel dimensions regardless of per-process DPI awareness.
            # GetSystemMetrics(SM_CXSCREEN) returns LOGICAL pixels when the process
            # is DPI-unaware (e.g., 1536 on a 125%-scaled 1920-wide display).
            hdc = ctypes.windll.user32.GetDC(0)
            sw = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES
            sh = ctypes.windll.gdi32.GetDeviceCaps(hdc, 117)  # DESKTOPVERTRES
            ctypes.windll.user32.ReleaseDC(0, hdc)
            if sw >= 640 and sh >= 480:
                safe_x = sw // 2
                safe_y = sh // 2  # vertical center — well above the taskbar
        except Exception:
            pass

    return ActionBurst(
        steps=[
            # Step 1: right_click at the physical screen center.
            # Explicit "x,y" in value bypasses UIA grounding (which returns the
            # taskbar element for "desktop").  target=None signals no UIA lookup.
            _s(sid, 1, "right_click", target=None, value=f"{safe_x},{safe_y}",
               desc=f"Right-click desktop center ({safe_x},{safe_y}) to open context menu",
               verif="Context menu appears"),
            # Steps 2–3: click menu items by name.  These are NOT pre-grounded
            # because the context menu hasn't opened yet.  The burst executor marks
            # them for late grounding and resolves them via OCR/UIA after the
            # preceding step's delay has elapsed.
            _s(sid, 2, "click", target="New",
               desc="Click 'New' in desktop context menu",
               verif="New submenu opens"),
            _s(sid, 3, "click", target="Folder",
               desc="Click 'Folder' in New submenu",
               verif="Folder rename mode active"),
            _s(sid, 4, "type", value=folder_name,
               desc=f"Type '{folder_name}'",
               verif=f"'{folder_name}' visible in rename box"),
            _s(sid, 5, "key_press", key="enter",
               desc="Press Enter to confirm folder creation",
               verif="New folder appears on desktop"),
        ],
        verify_at_end=True,
        timeout_ms=12000,
    )


def _s(
    subtask_id: int,
    step_id: int,
    action_type: str,
    *,
    target: str = None,
    value: str = None,
    key: str = None,
    desc: str = "",
    verif: str = "",
) -> ActionStep:
    return ActionStep(
        id=step_id,
        subtask_id=subtask_id,
        action_type=action_type,
        target=target,
        value=value,
        key=key,
        description=desc,
        verification=verif,
    )
