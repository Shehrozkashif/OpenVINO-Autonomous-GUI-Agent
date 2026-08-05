# core/groundtruth.py
"""Checks the operating system can prove, so the agent never has to ask a model.

This is the project's central rule: wherever the real world can be read
directly, read it. A file on disk, a process in the task list, a control's
value in the accessibility tree and a window title are all facts; an LLM
verdict about a screenshot is an opinion. Opinions are the fallback
(agents/reflection.py), never the default.

Every method returns a plain bool or a (bool, reason) pair. The reason string
is written for the PLANNER to read — it ends up in the step's failure record,
so it must name the real blocker, not just say "failed".
"""
import os
import re
import time

from loguru import logger

from core import apps, subtasks
from core.types import ActionStep, SubTask
from desktop import system
from desktop.capture import OCR_THUMB

# Shell error fragments that mean a command failed. Deliberately specific — a
# bare "error" would false-positive on the contents of a file being displayed.
_SHELL_ERROR_MARKERS = (
    "is denied", "cannot find", "could not find", "not recognized",
    "no such file", "exception", "categoryinfo", "command not found",
    "permission denied", "syntax error", "fatal:", "access denied",
)

# Teams/Office left-rail views switch via ctrl+<digit>. ctrl+3's own view is
# 'Teams', which collides with the app suffix in 'Microsoft Teams', so only the
# title HEAD (text before '|') is ever matched.
_TEAMS_VIEW_BY_KEY = {
    "1": "Activity", "2": "Chat", "3": "Teams",
    "4": "Calendar", "5": "Calls",
}
_NAV_VIEW_WORDS = (
    "calendar", "activity", "chat", "calls", "teams", "files", "mail", "inbox",
)


def file_saved_fresh(path: str, started_at: float) -> bool:
    """True when `path` exists AND was written during this subtask.

    The freshness window stops a stale file from an earlier run passing a save
    that never happened.
    """
    try:
        return os.path.exists(path) and os.path.getmtime(path) >= started_at - 2
    except OSError:
        return False


def launch_confirmed(exe_name: str) -> bool:
    """True when this executable is up.

    Always-running shell processes need a visible window to prove anything;
    normal apps accept a window OR a running process, since some draw through
    non-standard windows that enumeration misses.
    """
    if exe_name.lower() in apps.ALWAYS_RUNNING:
        return system.process_has_visible_window(exe_name)
    return (system.process_has_visible_window(exe_name)
            or system.is_process_running(exe_name))


def new_window_appeared(exe_name: str, baseline: int | None) -> bool:
    """Launch check for an app that was ALREADY running when the subtask began.

    With a baseline, only a NEW window counts — focusing the pre-existing
    window must not pass, because that window may be busy running something
    else. Without one, ordinary launch confirmation applies.
    """
    if not exe_name:
        return False
    if baseline is not None:
        return system.count_process_windows(exe_name) > baseline
    return launch_confirmed(exe_name)


def view_switch_confirmed(step: ActionStep) -> bool:
    """True when the foreground window title proves a left-rail view switch
    reached the view this step aimed for.

    The title (Win32 GetWindowText) is the one signal that survives OCR and UIA
    being unavailable, so this works on the VLM-only path too. Without it the
    OCR verifier scored every switch uncertain and the run retried ctrl+4 into
    a multi-minute stall on a Calendar that HAD opened.
    """
    key = (step.key or "").lower()
    low = (step.description or "").lower()
    is_nav_hotkey = bool(re.fullmatch(r"ctrl\+\d", key))
    is_nav_desc = any(p in low for p in
                      ("switch to", "sidebar", "left rail", "left-rail", "navigate to"))
    if not (is_nav_hotkey or is_nav_desc):
        return False

    # Candidate view names: what the hotkey digit maps to, plus any view word
    # attached to button/tab/view/icon ("Calendar button"). Requiring that
    # attachment stops app-context mentions ("with Teams already open, …")
    # from matching.
    candidates: list[str] = []
    if is_nav_hotkey:
        view = _TEAMS_VIEW_BY_KEY.get(key.split("+")[1])
        if view:
            candidates.append(view.lower())
    candidates += [m.group(1) for m in re.finditer(r"(\w+)\s+(?:button|tab|view|icon)", low)
                   if m.group(1) in _NAV_VIEW_WORDS]
    if not candidates:
        return False

    try:
        from desktop.snapshot import _get_foreground_hwnd_and_title
        _, title = _get_foreground_hwnd_and_title()
    except Exception:
        return False
    head = (title or "").split("|")[0].strip().lower()
    return bool(head) and any(c in head for c in candidates)


def click_holds_focus(xy: tuple[int, int] | None) -> bool:
    """True when the clicked point owns keyboard focus and takes text input.

    A click whose whole effect is placing the caret changes zero pixels, so the
    phash diff scores it a no-op and retries it forever. The accessibility tree
    settles it. The clicked PIXEL must be a text control too: WebView apps
    expose one focused DocumentControl spanning nearly the whole window, and
    "focused rect contains the point" alone rescued dead clicks on buttons for
    ten minutes straight, which also kept the dead-point blacklist from firing.
    """
    if not xy:
        return False
    try:
        from desktop import uia
        if uia.control_type_at_point(*xy) not in ("EditControl", "DocumentControl"):
            return False
        info = uia.focused_element_info()
        if not info or info["control_type"] not in ("EditControl", "DocumentControl"):
            return False
        left, top, right, bottom = info["rect"]
        return left <= xy[0] <= right and top <= xy[1] <= bottom
    except Exception:
        return False


class GroundTruth:
    """Deterministic checks that need to look at the screen.

    Holds the three collaborators those checks share — the screen capturer, the
    OCR engine, and the orchestrator's log function — so call sites stay short.
    Checks needing none of them are module-level functions above.
    """

    def __init__(self, capturer, ocr, log=None):
        self.capturer = capturer
        self.ocr = ocr
        self.log = log or (lambda _msg: None)

    # ── Reading the screen ────────────────────────────────────────────────────

    def _screen_text(self, min_conf: float = 0.6) -> str:
        """Lower-cased OCR text of the current screen, "" if OCR is unavailable."""
        img = self.capturer.capture()
        img.thumbnail(OCR_THUMB)
        return " ".join(w.text for w in self.ocr.extract(img) if w.conf >= min_conf).lower()

    # ── Typing and form fields ────────────────────────────────────────────────

    def typed_text_in_focused_control(
        self, text: str | None, target: str | None = None,
    ) -> bool:
        """The focused control's accessible value contains the text just typed —
        and, when the step named a destination field, the focused control IS
        that field.

        Content alone is not enough: an attendee email typed into the Title
        field still "contains the typed text". Reads the real widget from the
        tree in ~50 ms instead of a ~3-4 s OCR+LLM reflection call. False means
        "fall back to reflection", not "the typing failed".
        """
        if not text or len(text.strip()) < 2 or "{{cred:" in text:
            return False
        try:
            from desktop import uia
            info = uia.focused_element_info() or {}
            value = info.get("value") or ""
            if not value:
                return False

            def _norm(s: str) -> str:
                return re.sub(r"\s+", " ", s).strip().lower()

            if target:
                name = _norm(info.get("name") or "")
                want = _norm(target)
                if not name or (want not in name and name not in want):
                    self.log(
                        f"  [TYPE-VERIFY] Focus is on '{name or '?'}', not "
                        f"'{target}' — text landed in the wrong field"
                    )
                    return False
            return _norm(text) in _norm(value)
        except Exception:
            return False

    def value_on_screen(self, value: str | None) -> bool:
        """Strict OCR check: is `value` literally on the live screen?

        Salvages a set_value/select whose field LABEL cannot be grounded —
        filling a field REMOVES its placeholder ('Add title' vanishes once a
        title is typed), so a re-attempt can never find the label again even
        though the work is done.

        Strict by design: alphanumeric-normalised match, and the value must
        normalise to >= 8 characters. Short values like times ('3:00 PM' →
        '300pm') are always rejected — with fields side by side, a Start time
        of '3:30 PM' would otherwise satisfy a check for End time '3:30 PM' and
        silently skip setting it.
        """
        if not value or self.ocr is None:
            return False
        try:
            if not self.ocr.is_available():
                return False
        except Exception:
            return False

        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())

        variants = {_norm(value)}
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", value.strip())
        if m:
            mo, da, yr = m.groups()
            # Apps render dates un-padded ('7/29/2026') — match both forms.
            variants.add(f"{int(mo)}{int(da)}{yr}")
            variants.add(f"{int(mo):02d}{int(da):02d}{yr}")
        # Gate on the value's DISTINCTIVENESS (its longest form): a full date
        # passes at 8 chars ('07292026') and must still match the 7-char
        # un-padded rendering apps display. Times never pass the gate.
        if max(len(v) for v in variants) < 8:
            return False
        try:
            img = self.capturer.capture()
            img.thumbnail(OCR_THUMB)
            text = _norm(" ".join(w.text for w in self.ocr.extract(img)))
            return any(v in text for v in variants)
        except Exception:
            return False

    def unfilled_form_values(self, subtask: SubTask, controls_text: str) -> list[str]:
        """Required form values NOT provably present on the form yet.

        UIA path: read each value back from the CLICKABLE CONTROLS list.
        OCR path: emptiness cannot be proven, but the PRESENCE of long,
        distinctive values (title, full date, email — never short times) can be
        tested positively. Without that, the OCR path let Save fire with only
        the title set: Teams created a meeting at the DEFAULT date and time
        with no attendee, and the run reported full success.
        """
        if "CLICKABLE CONTROLS" in (controls_text or ""):
            return [v for v in subtasks.required_values(subtask)
                    if not subtasks.value_in_controls(v, controls_text)]
        missing = []
        for v in subtasks.required_values(subtask):
            is_date = bool(re.match(r"\d{1,2}/\d{1,2}/\d{4}$", v.strip()))
            # Short/ambiguous values (times) cannot be checked safely. Dates
            # always pass: un-padded '7/29/2026' normalises to 7 chars but is
            # date-shaped and therefore distinctive.
            if len(re.sub(r"[^a-z0-9]", "", v.lower())) < 8 and not is_date:
                continue
            if not self.value_on_screen(v):
                missing.append(v)
        return missing

    # ── Dialogs ───────────────────────────────────────────────────────────────

    def save_dialog_visible(self) -> bool:
        """True when a Windows Save/Save-As dialog is on screen.

        Confirms ctrl+s opened a dialog BEFORE any path is typed — otherwise a
        silent save of an already-named file would send the path into the
        document body. The accessibility tree answers first (exact control
        names); OCR is the fallback for when UIA is unavailable or times out.
        """
        try:
            from desktop import uia
            verdict = uia.save_dialog_open()
            if verdict is not None:
                return verdict
        except Exception:
            pass
        try:
            text = self._screen_text()
            return "file name" in text and ("save" in text or "cancel" in text)
        except Exception:
            return False

    # ── Shell commands ────────────────────────────────────────────────────────

    def command_effect(
        self, subtask: SubTask, started_at: float, typed_ok: bool,
    ) -> tuple[bool, str]:
        """Verify a "run: <command>" subtask against the real world.

        A successful shell command usually prints NOTHING — just a new empty
        prompt — which OCR reflection systematically mis-reads as failure.
        Check what actually happened instead:
          1. delete commands → the target is gone
          2. create/redirect → the target exists AND is fresh
          3. anything else   → no shell error text on screen
        Returns (ok, reason).
        """
        m = re.search(r"run:\s*(.+)$", subtask.description, re.IGNORECASE | re.DOTALL)
        cmd = (m.group(1) if m else subtask.description).strip()
        time.sleep(0.8)   # let the command finish writing

        path_rx = r"(\"[^\"]+\"|'[^']+'|\S+)"

        def _norm(p: str) -> str:
            return os.path.expandvars(p.strip().strip("'\""))

        m_del = re.search(rf"^(?:del|rm|remove-item)\s+(?:-\S+\s+)*{path_rx}",
                          cmd, re.IGNORECASE)
        if m_del:
            target = _norm(m_del.group(1))
            if not os.path.exists(target):
                return True, f"'{target}' no longer exists"
            return False, f"'{target}' still exists — delete did not run or failed"

        m_new = re.search(rf">>?\s*{path_rx}", cmd) or re.search(
            rf"^(?:ni|new-item|touch|mkdir|md)\s+(?:-\S+\s+)*{path_rx}",
            cmd, re.IGNORECASE)
        if m_new:
            target = _norm(m_new.group(1))
            if not os.path.exists(target):
                return False, (
                    f"expected '{target}' on disk but it does not exist — "
                    f"the command did not run or failed"
                )
            try:
                fresh = os.path.getmtime(target) >= started_at - 2
            except OSError:
                fresh = True
            if fresh:
                return True, f"'{target}' exists on disk (freshly written)"
            return False, (
                f"'{target}' exists but was NOT modified by this command "
                f"(stale file from an earlier run)"
            )

        if not typed_ok:
            return False, "Enter pressed but no command was typed first"
        try:
            text = self._screen_text()
            hit = next((mk for mk in _SHELL_ERROR_MARKERS if mk in text), None)
            if hit:
                return False, f"error text visible in terminal output ('{hit}')"
            return True, "no error output — shell silence means success"
        except Exception:
            return True, "no error detected (output check unavailable)"

    # ── App launches ──────────────────────────────────────────────────────────

    def verify_launch(self, subtask: SubTask, baselines: dict[str, int]) -> bool:
        """Confirm an app-launch subtask actually opened its app.

        Process checks come first because they are immune to the OCR false
        positive of reading an app's name out of the agent's own window. Only
        genuine launch subtasks are checked: the old bare "open" match fired on
        "right click on the desktop to open the context menu" and sent the
        agent off to the Start menu.
        """
        desc = subtask.description.lower()
        if "already open" in desc or "already running" in desc:
            return True
        app_vocab = set(apps.PROCESS_MAP) | set(apps.APP_SIGNALS)
        is_app_launch = (
            any(w in desc for w in ("launch", "search launcher"))
            or ("open" in desc and any(k in desc for k in app_vocab))
        )
        if not is_app_launch:
            return True

        proc = apps.process_for(desc)
        if proc:
            return self._verify_by_process(proc, baselines.get(proc))
        signals = apps.signals_for(subtask.description)
        return self._verify_by_ocr(signals) if signals else True

    def _verify_by_process(self, proc: str, baseline: int | None) -> bool:
        label = proc.split(".")[0]
        for attempt, wait_s in enumerate([1.5, 2.5]):
            time.sleep(wait_s)
            if baseline is not None:
                if system.count_process_windows(proc) > baseline:
                    self.log(f"  [CHECK] '{label}' NEW window confirmed")
                    return True
            elif launch_confirmed(proc):
                self.log(f"  [CHECK] '{label}' process confirmed running")
                return True
            if attempt == 0:
                self.log(f"  [CHECK] '{label}' not yet confirmed — retrying in 2.5s")
        self.log(f"  [CHECK] Launch of '{proc}' not confirmed")
        return False

    def _verify_by_ocr(self, signals: list[str]) -> bool:
        label = signals[0]
        for attempt, wait_s in enumerate([1.5, 2.5]):
            time.sleep(wait_s)
            try:
                from desktop.snapshot import capture_snapshot
                snapshot = capture_snapshot(self.capturer, self.ocr)
                # Foreground only — background text includes the agent's own window.
                seen = " ".join(r.text for r in snapshot.ocr_regions if r.is_in_foreground)
                if any(sig in seen for sig in signals):
                    self.log(f"  [CHECK] '{label}' confirmed on screen")
                    return True
                if attempt == 0:
                    self.log(f"  [CHECK] '{label}' not yet visible — retrying in 2.5s")
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Launch check error: {e}")
                return True
        self.log(f"  [CHECK] Expected '{label}' on screen but not found")
        return False

    # ── Settling ──────────────────────────────────────────────────────────────

    def wait_for_settle(
        self, min_s: float = 0.5, max_s: float = 3.0, poll_interval: float = 0.25,
    ) -> None:
        """Wait until the screen stops changing, or max_s elapses.

        Two identical frame hashes in a row mean the transition finished. Never
        returns sooner than min_s; falls back to a flat sleep on any error.
        """
        import imagehash
        try:
            time.sleep(min_s)
            deadline = time.time() + (max_s - min_s)
            prev_hash = None
            while time.time() < deadline:
                img = self.capturer.capture()
                img.thumbnail((320, 180))
                cur_hash = str(imagehash.phash(img))
                if prev_hash is not None and cur_hash == prev_hash:
                    self.log("  [SETTLE] Screen stable — proceeding")
                    return
                prev_hash = cur_hash
                time.sleep(poll_interval)
            self.log(f"  [SETTLE] Max wait {max_s:.1f}s reached — proceeding anyway")
        except Exception as e:
            logger.debug(f"[SETTLE] Error: {e} — falling back to fixed wait")
            time.sleep(max_s)
