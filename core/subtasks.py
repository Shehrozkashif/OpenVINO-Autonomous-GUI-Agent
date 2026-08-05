# core/subtasks.py
"""Reading a subtask's own words.

The Router writes subtasks in a stable phrasing ("save the report as C:/x.txt",
"set Title to 'Sync', … then click Save"). Parsing that phrasing tells the
orchestrator what DETERMINISTIC completion check applies — a file on disk, a
typed payload, a commit button — instead of asking a model whether the work is
done. Every function here is pure: text in, facts out, no I/O and no state, so
each one is trivially testable.
"""
import difflib
import os
import re

from loguru import logger

from core.types import SubTask

# Commit verbs that SUBMIT-and-CLOSE a form. A subtask ending in one of these is
# complete the moment that button is clicked and verified — the form is gone
# afterwards, so its filled state can never be re-verified.
SUBMIT_LABELS = (
    "save", "send", "schedule", "create", "submit", "book", "done",
    "update", "post", "finish", "confirm", "publish",
)


def save_target(subtask: SubTask) -> str | None:
    """The destination path of a "save … as <path>" subtask, else None.

    Lets the orchestrator confirm the save on disk instead of OCR-reading a
    silent editor. Only a real path (separator + filename) is disk-verifiable.
    """
    m = re.search(
        r"\bsave\b[^\n]*?\bas\s+['\"]?([^'\"\n]+?)['\"]?\s*$",
        subtask.description or "", re.IGNORECASE,
    )
    if not m:
        return None
    path = m.group(1).strip().rstrip(".")
    if "/" not in path and "\\" not in path:
        return None
    return os.path.expanduser(os.path.expandvars(path))


def type_payload(subtask: SubTask) -> str | None:
    """The literal text a "… type: <text>" subtask asks to type, else None.

    The subtask is complete the moment that text is typed and verified —
    without this the planner re-types the payload (duplicating it in the
    document) or drifts into save steps belonging to a later subtask. Short
    payloads ("type notepad") are launcher idioms, not a typing goal.
    """
    m = re.search(r"\btype:?\s+['\"]?(.+?)['\"]?\s*$",
                  subtask.description or "", re.IGNORECASE)
    if not m:
        return None
    payload = m.group(1).strip()
    return payload if len(payload) >= 6 else None


def submit_target(subtask: SubTask) -> str | None:
    """The commit button a form-fill subtask ends by clicking, else None.

    Router format is "…set X, set Y, then click Save to create the meeting" —
    a field-setting body followed by a terminal commit click. Returns the
    button label ("Save") so a verified click on it can close the subtask
    instead of looping. A bare "click Save" subtask returns None; the
    single-click path covers that one.
    """
    desc = subtask.description or ""
    if not re.search(r"\b(set|add|type|enter|fill|choose|select)\b", desc, re.IGNORECASE):
        return None
    labels = "|".join(SUBMIT_LABELS)
    matches = list(re.finditer(
        rf"\bclick\s+(?:the\s+|on\s+)?['\"]?({labels})\b", desc, re.IGNORECASE,
    ))
    return matches[-1].group(1) if matches else None


def required_values(subtask: SubTask) -> list[str]:
    """Concrete values a form-fill subtask must place before it commits.

    Parsed from the router's "set X to '<v>' / set the date to <v> / add
    attendee <v>" phrasing: quoted strings, dates, times and emails. The
    COMMIT-GUARD uses these to refuse a Save until each one is really on the
    form.
    """
    desc = subtask.description or ""
    vals: list[str] = []
    vals += re.findall(r"'([^']{2,})'", desc)                        # quoted (title)
    vals += re.findall(r'"([^"]{2,})"', desc)
    vals += [f"{int(mo)}/{int(da)}/{yr}"                             # dates, un-padded
             for mo, da, yr in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", desc)]
    vals += re.findall(r"\b\d{1,2}:\d{2}\s*[AaPp]\.?[Mm]\.?\b", desc)  # times
    vals += re.findall(r"[\w.+-]+@[\w.-]+\.\w+", desc)               # emails
    seen, out = set(), []
    for v in vals:
        k = v.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(v.strip())
    return out


def value_in_controls(value: str, controls_text: str) -> bool:
    """Is `value` present in the UIA control read-back text?

    Field values surface as "[Edit = 'PROJECT DISCUSSION']". Dates match in
    both padded and un-padded form (07/29/2026 vs 7/29/2026).
    """
    t = (controls_text or "").lower()
    v = value.strip().lower()
    if v and v in t:
        return True
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", value.strip())
    if m:
        mo, da, yr = m.groups()
        for var in (f"{int(mo)}/{int(da)}/{yr}", f"{int(mo):02d}/{int(da):02d}/{yr}"):
            if var.lower() in t:
                return True
    return False


def is_single_click(subtask: SubTask) -> bool:
    """True when the subtask's whole goal is ONE click/open of a control.

    Such a subtask is done the instant that click lands and materially changes
    the screen. On the OCR-only path the goal check cannot confirm it: the
    WebView2 result is invisible to OCR and the opened form often repeats the
    button's own label ("New meeting"), which the goal check reads as "button
    still there → not clicked" and loops forever. A second action verb means it
    is a form fill, not a bare click, and must run the full loop.
    """
    desc = (subtask.description or "").strip().lower()
    if not desc:
        return False
    if not re.match(
        r"^(with[^,]*,\s*)?(then\s+)?(click|open|select|press|tap|go to|"
        r"navigate to|switch to)\b",
        desc,
    ):
        return False
    # Drop the leading "with … open," preamble — its verbs describe prior
    # state, not this subtask's actions.
    body = re.sub(r"^with[^,]*,\s*", "", desc)
    return not re.search(
        r"\b(then|set|type|enter|fill|add|choose|save|create|send|"
        r"delete|remove|check|toggle|write)\b",
        body,
    )


def targets_match(a: str | None, b: str | None) -> bool:
    """True when two control labels are the same modulo case/spacing/quotes, or
    one is a whole-word subset of the other ('Save' ⊆ 'Save and close').
    """
    def _norm(s):
        return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    wa, wb = set(na.split()), set(nb.split())
    return wa <= wb or wb <= wa


def texts_equivalent(a: str, b: str) -> bool:
    """True when two strings are the same text modulo case/whitespace/quotes.

    Fuzzy, because the planner re-emits the payload from the subtask
    description and may vary punctuation. Containment counts only when the
    lengths are comparable, so typing a fragment never passes as the full text.
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().strip("'\"")).lower()

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if (na in nb or nb in na) and min(len(na), len(nb)) / max(len(na), len(nb)) >= 0.8:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85


def explicit_coords(value: str | None) -> tuple[int, int] | None:
    """Parse an explicit "x,y" pixel pair from a step's value field.

    Visual-planner click steps carry direct screen coordinates this way,
    bypassing grounding entirely.
    """
    if not value or "," not in value:
        return None
    parts = value.split(",", 1)
    if all(p.strip().lstrip("-").isdigit() for p in parts):
        return int(parts[0].strip()), int(parts[1].strip())
    return None


def topological_order(subtasks: list[SubTask]) -> list[SubTask]:
    """Order subtasks so each runs after everything it depends on.

    A dependency cycle falls back to plain id order — a bad plan must still
    run in a sensible sequence, never vanish.
    """
    by_id = {s.id: s for s in subtasks}
    in_degree = {s.id: len(s.depends_on) for s in subtasks}
    dependents: dict[int, list[int]] = {s.id: [] for s in subtasks}
    for s in subtasks:
        for dep_id in s.depends_on:
            if dep_id in dependents:
                dependents[dep_id].append(s.id)

    order, queue = [], [s for s in subtasks if not s.depends_on]
    while queue:
        current = queue.pop(0)
        order.append(current)
        for dep_id in dependents.get(current.id, []):
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(by_id[dep_id])

    if len(order) != len(subtasks):
        logger.warning("[ORCHESTRATOR] Dependency cycle — falling back to ID order")
        return sorted(subtasks, key=lambda s: s.id)
    return order
