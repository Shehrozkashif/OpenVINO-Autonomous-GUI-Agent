# agents/router.py
"""Router Agent — decomposes user instructions into sub-tasks."""

import json
import os
import re
import uuid

from loguru import logger

from core.protocols import InferenceClient, SubTask
from utils.platform_utils import get_desktop_path

_SUBTASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":          {"type": "integer"},
            "description": {"type": "string"},
            "depends_on":  {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["id", "description", "depends_on"],
    },
}

def _today_line() -> str:
    """Current date, injected per request so 'tomorrow at 3pm' resolves to a
    concrete date even in a long-running session.
    """
    from datetime import datetime
    now = datetime.now()
    return f"Today is {now.strftime('%A')}, {now.strftime('%m/%d/%Y')} {now.strftime('%I:%M %p')}."


# ── Runtime machine identity ───────────────────────────────────────────────────
_USER = os.getenv("USERNAME") or "user"
_SHELL_PROMPT = _USER  # username only — hostnames can be too long for OCR

_ROUTER_OS_CONTEXT = "Windows 11"
# Resolved LITERAL path (handles OneDrive-redirected Desktops, where
# $env:USERPROFILE\Desktop does not exist). Forward slashes: backslashes
# need \\ escaping in the JSON the LLM emits and small models mangle them
# (observed truncation to "C:\"). All Windows shells/dialogs accept "/".
_DESKTOP_PATH = get_desktop_path().replace("\\", "/")
# Closed-class English function words (≥4 letters) — never treated as app
# names by the installed-app hint. Purely grammatical words, not app-specific.
_FUNCTION_WORDS = frozenset({
    "with", "from", "this", "that", "then", "than", "when", "where", "will",
    "would", "your", "into", "onto", "over", "under", "have", "them", "they",
    "there", "here", "please", "want", "need", "make", "using", "about",
    "after", "before", "some", "just", "like", "only", "also", "very",
})

_TERMINAL_APP = "Windows Terminal"
_CALC_APP     = "Calculator"
_FILES_APP    = "File Explorer"
_SETTINGS_APP = "Settings"

# ── Router system prompt ───────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a desktop automation coordinator on ROUTER_OS_PLACEHOLDER.
Decompose any user instruction into ordered sub-tasks a GUI agent can execute —
covering the user's FULL intent with the fewest steps that leave nothing out.

━━━ CORE RULES ━━━
1. COVER THE FULL INTENT (most important rule). Scan the instruction and find
   EVERY action the user asks for. Each verb is an action: open, go to, search,
   click, type/write, run, save, rename, send, download, print, delete, close,
   etc. Emit ONE sub-task per action, IN ORDER, and make the LAST sub-task
   achieve the user's end result. Silently dropping or merging a requested
   action is the #1 failure — if the words "and X" appear, there is a sub-task
   for X. (e.g. "...and save it" → a save sub-task; "...and close it" → a close
   sub-task; "...and email it" → a send sub-task.)
2. FEWEST STEPS, but never fewer than rule 1 requires. Do not INVENT steps the
   user didn't ask for (no extra confirm / wait / re-open / close). "Minimum"
   limits invented steps only — it NEVER licenses skipping a requested action.
3. One distinct action per sub-task (launch app / navigate / click / type / run / save).
4. Set depends_on so each sub-task runs after its prerequisites complete.
   CRITICAL: If sub-task B opens/reads a file that sub-task A creates, set B's depends_on to [A_id].
   Example: create file (id=2) then open in Notepad (id=3) → Notepad sub-task has depends_on:[2].
5. Descriptions must be SPECIFIC — include exact URLs, filenames, commands, and app names.
6. STATE CONTEXT in every dependent sub-task description so the planner knows what is already open:
     "with the terminal already open, run: <command>"
     "with Teams already open, open the schedule-meeting form"
     "with VS Code already open, create a new file named <name>"
   This is MANDATORY for any sub-task that depends_on an app-launch sub-task.

━━━ HOW TO LAUNCH APPS ━━━
  Sub-task description states WHAT to open, never HOW: "open <AppName>".

  Do NOT write "...using the search launcher" or "...by clicking its icon" —
  that's an execution detail for the planner, which sees the live screen and
  picks the fastest reliable method itself (click a visible icon in one step,
  or fall back to the search launcher when nothing is visible). Baking the
  method into the description removes that choice and forces extra steps.

━━━ TASK → METHOD ━━━
  File / folder ops   →  terminal (touch / mkdir / rm / mv / echo / cp) — most reliable.
                         EXCEPTION: if the user explicitly says to use the mouse, GUI,
                         File Explorer, or right-click — honor that and describe the
                         GUI route instead (e.g. "right click on the desktop, click New,
                         click Text Document, type <name>, press enter").
  Code editing        →  VS Code
  Documents           →  LibreOffice Writer / Calc / Impress
  Email               →  Thunderbird
  Calculator          →  CALC_APP_PLACEHOLDER
  System settings     →  SETTINGS_APP_PLACEHOLDER
  Screenshot          →  Print Screen key (one sub-task, no app needed)
  Simple text files   →  echo command in terminal (single line) or nano (multi-line)
  Meetings / calls    →  the app the user names (Teams, Zoom, Skype, …).
                         Launch it → open its New/Schedule form → fill the
                         form → confirm. See FORM FILLING below.

━━━ FORM FILLING (schedule a meeting, compose an email, create an event) ━━━
  Decompose form work into exactly TWO kinds of sub-task:
    1. one sub-task to OPEN the form ("with Zoom already open, click the
       Schedule button to open the schedule-meeting form")
    2. ONE final sub-task that fills ALL the fields AND ends by clicking Save —
       the Save is the LAST action of this SAME sub-task, never its own:
       ("with the schedule form open, set Topic to 'Weekly Sync', set Date to
       07/10/2026, set Start time to 3:00 PM, set End time to 3:30 PM, add
       attendee alex@example.com, then click Save to create the meeting").
       So form work is exactly TWO sub-tasks — RIGHT: [open the form],
       [set every field …, then click Save]. WRONG: [open the form],
       [set every field …], [click Save] ← that trailing Save sub-task is a
       bug: Save CLOSES the form, so it lands on the "meeting created"
       confirmation, finds no form to save, and loops. Related fields of one
       form always belong in ONE sub-task, never one sub-task per field.
       ATTENDEES / INVITEES / RECIPIENTS ARE FORM FIELDS, not a later action:
       the email address(es) to invite go in THIS sub-task with the exact
       address from the instruction ("...add attendee alex@example.com"), filled
       BEFORE the Save. NEVER emit a separate "invite/add the attendee" sub-task
       — a meeting saved before its attendees are filled drops them, and a bare
       Save sub-task carries no email so the agent is forced to invent one.
       Add a further sub-task ONLY when the user asks to SEND/SHARE the
       invitation as a distinct action that happens after the meeting exists
       (e.g. "copy the join link and email it") — adding attendees to the form
       is NOT that.
  Use concrete values: resolve "tomorrow" to the actual date, "3pm" to 3:00 PM.
  A meeting length/duration is NOT a form field — the form has only Start time
  and End time. Convert any duration into an explicit End time (start + length)
  and set End time to it; NEVER write "set duration". e.g. start 3:00 PM for
  30 minutes → set End time to 3:30 PM; start 10:00 AM for 15 minutes → set End
  time to 10:15 AM.

━━━ AVAILABLE APPS ━━━
Trust the app name the user gives you (meeting apps, games,
utilities — anything installed): an app the user NAMES is always allowed.
When the instruction names no specific app for a generic task, default to:
VS Code, LibreOffice Writer/Calc/Impress, Thunderbird,
TERMINAL_APP_PLACEHOLDER, CALC_APP_PLACEHOLDER, SETTINGS_APP_PLACEHOLDER, FILES_APP_PLACEHOLDER.
NEVER invent an app the user did not name and that is not in the defaults
(no gedit, mousepad, kate, VLC, GIMP, …).
For text editing → nano (simple) or LibreOffice Writer (formatted docs).

━━━ OUTPUT ━━━
Valid JSON array only. No markdown, no explanation, nothing outside the array.
[{"id":1,"description":"...","depends_on":[]},{"id":2,"description":"...","depends_on":[1]}]

━━━ EXAMPLES ━━━

"open vs code"
→ [{"id":1,"description":"open Visual Studio Code","depends_on":[]}]

"open calculator"
→ [{"id":1,"description":"open CALC_APP_PLACEHOLDER","depends_on":[]}]

"open terminal"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]}]

"open terminal and run python3 --version"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: python3 --version","depends_on":[1]}]

"create a file hello.txt on the desktop and write hello world in it"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: echo 'hello world' > DESKTOP_PATH_PLACEHOLDER/hello.txt","depends_on":[1]}]

"delete file notes.txt from the desktop"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: rm DESKTOP_PATH_PLACEHOLDER/notes.txt","depends_on":[1]}]

"open vs code and create a new python file named app.py"
→ [{"id":1,"description":"open Visual Studio Code","depends_on":[]},
   {"id":2,"description":"with VS Code already open, create a new file named app.py","depends_on":[1]}]

"write a python script that prints hello world and run it"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: echo 'print(\"hello world\")' > DESKTOP_PATH_PLACEHOLDER/hello.py","depends_on":[1]},
   {"id":3,"description":"with the terminal already open, run: python3 DESKTOP_PATH_PLACEHOLDER/hello.py","depends_on":[2]}]

"install the requests python package"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: pip install requests","depends_on":[1]}]

"open libreoffice writer and type hello world"
→ [{"id":1,"description":"open LibreOffice Writer","depends_on":[]},
   {"id":2,"description":"with LibreOffice Writer open, click in the document area and type: hello world","depends_on":[1]}]

"open notepad, write a haiku about the sea, and save the file"
→ [{"id":1,"description":"open Notepad","depends_on":[]},
   {"id":2,"description":"with Notepad already open, click in the document area and type: <the haiku text>","depends_on":[1]},
   {"id":3,"description":"with the text written in Notepad, save the document as DESKTOP_PATH_PLACEHOLDER/haiku.txt","depends_on":[2]}]

"open a spreadsheet and enter sales data in cell A1"
→ [{"id":1,"description":"open LibreOffice Calc","depends_on":[]},
   {"id":2,"description":"with LibreOffice Calc open, click on cell A1 and enter the sales data","depends_on":[1]}]

"compose a new email in thunderbird"
→ [{"id":1,"description":"open Thunderbird","depends_on":[]},
   {"id":2,"description":"with Thunderbird open, click the Write new message button","depends_on":[1]}]

"schedule a zoom meeting titled Weekly Sync tomorrow at 3pm for 30 minutes and invite alex@example.com" (today = 07/09/2026)
→ [{"id":1,"description":"open Zoom","depends_on":[]},
   {"id":2,"description":"with Zoom already open, click the Schedule button to open the schedule-meeting form","depends_on":[1]},
   {"id":3,"description":"with the schedule-meeting form open, set Topic to 'Weekly Sync', set the date to 07/10/2026, set the start time to 3:00 PM, set the end time to 3:30 PM, add attendee alex@example.com, then click Save to create the meeting","depends_on":[2]}]

"schedule a teams meeting titled Standup tomorrow at 10am for 15 minutes and invite sam@example.com" (today = 07/09/2026)
→ [{"id":1,"description":"open Microsoft Teams","depends_on":[]},
   {"id":2,"description":"with Teams already open, open the calendar and click New meeting to open the schedule-meeting form","depends_on":[1]},
   {"id":3,"description":"with the schedule-meeting form open, set the title to 'Standup', add attendee sam@example.com, set the date to 07/10/2026, set the start time to 10:00 AM, set the end time to 10:15 AM, then click Save to create the meeting","depends_on":[2]}]

"take a screenshot"
→ [{"id":1,"description":"take a screenshot using the Print Screen keyboard shortcut","depends_on":[]}]

"open system settings"
→ [{"id":1,"description":"open SETTINGS_APP_PLACEHOLDER","depends_on":[]}]

"calculate 15 percent of 200"
→ [{"id":1,"description":"open CALC_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the calculator open, compute 15 percent of 200","depends_on":[1]}]

"open terminal, create notes.txt on the desktop, write hello world in it, then open it in notepad"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: echo 'hello world' > DESKTOP_PATH_PLACEHOLDER/notes.txt","depends_on":[1]},
   {"id":3,"description":"open Notepad","depends_on":[2]},
   {"id":4,"description":"with Notepad already open, open the file DESKTOP_PATH_PLACEHOLDER/notes.txt","depends_on":[3]}]

PATHS: always copy file paths EXACTLY as given above (DESKTOP_PATH_PLACEHOLDER is the
real desktop folder on this machine). Use forward slashes. NEVER invent paths like
C:/Users/Public — they require admin rights."""

# Apply runtime substitutions — no hardcoded machine values
ROUTER_SYSTEM_PROMPT = (
    ROUTER_SYSTEM_PROMPT
    .replace("ROUTER_OS_PLACEHOLDER", _ROUTER_OS_CONTEXT)
    .replace("DESKTOP_PATH_PLACEHOLDER", _DESKTOP_PATH)
    .replace("TERMINAL_APP_PLACEHOLDER", _TERMINAL_APP)
    .replace("CALC_APP_PLACEHOLDER", _CALC_APP)
    .replace("SETTINGS_APP_PLACEHOLDER", _SETTINGS_APP)
    .replace("FILES_APP_PLACEHOLDER", _FILES_APP)
)


class RouterAgent:
    def __init__(self, client: InferenceClient):
        self.client = client

    @staticmethod
    def _installed_app_hint(instruction: str) -> str:
        """Ground-truth hint: installed apps whose names appear in the instruction.

        General by construction — the app list comes from the OS
        (Get-StartApps: Win32 + Store apps) and the match comes from the
        user's own words. No app names are hardcoded. Stops the router from
        planning a browser/web route when the user has the real app installed
        (e.g. 'outlook meeting' → the installed Outlook, not outlook.live.com).
        """
        from utils.platform_utils import installed_apps
        apps = installed_apps()
        if not apps:
            logger.warning(
                "[ROUTER] Installed-app list unavailable (Get-StartApps "
                "failed/timed out) — no app hint for the planner"
            )
            return ""
        # WHOLE-WORD matching only, minus closed-class function words. Loose
        # substring matching sent a live run into the wrong app entirely:
        # "schedule a meeting WITH..." matched 'Task SCHEDULEr' and 'Firewall
        # WITH Advanced Security', and the router built the whole plan around
        # Task Scheduler. An app is hinted only when the user actually NAMES
        # a word of it ("outlook", "zoom", "spotify").
        tokens = set(re.findall(r"[a-zA-Z]{4,}", instruction.lower()))
        tokens -= _FUNCTION_WORDS
        hits = sorted({
            a for a in apps
            if tokens & set(re.findall(r"[a-zA-Z]{4,}", a.lower()))
        })
        if not hits:
            logger.info(
                f"[ROUTER] No installed app matches the instruction "
                f"({len(apps)} apps checked)"
            )
            return ""
        logger.info(f"[ROUTER] Installed-app hint: {hits[:8]}")
        return (
            "Ground truth — these apps mentioned in the instruction are "
            "INSTALLED on this machine: " + ", ".join(hits[:8]) + ". "
            "Use the installed app (open it via Start menu search), NOT a "
            "browser/web version of it."
        )

    def decompose(
        self,
        instruction: str,
        screen_context: str | None = None,
        memory_hint: str | None = None,
    ) -> tuple[str, list[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}\n{_today_line()}"
        app_hint = self._installed_app_hint(instruction)
        if app_hint:
            user_content += f"\n\n{app_hint}"
        if memory_hint:
            user_content += f"\n\n{memory_hint}"
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
            user_content += (
                "\nUse screen context to skip already-done steps and prefer "
                "clicking visible icons over searching."
            )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # 768 tokens truncated real 8-subtask decompositions mid-string (seen
        # live) — the parse failed and the retry produced junk. 1536 fits any
        # realistic plan; generation still stops at the closing bracket.
        resp = self.client.query_llm(messages, max_tokens=1536, temperature=0.1,
                                   response_schema=_SUBTASK_SCHEMA)
        try:
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("[ROUTER] Parse failed — retrying with JSON-only reminder")
            # Keep the FULL router system prompt: a bare "output JSON" retry
            # (no decomposition rules) returns conversational to-do items
            # ("Confirm the date with X") that no planner can act on.
            retry_messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT
                 + "\nOutput ONLY the JSON array — no prose, no explanation."},
                {"role": "user", "content": user_content},
            ]
            resp = self.client.query_llm(retry_messages, max_tokens=1536, temperature=0.0,
                                       response_schema=_SUBTASK_SCHEMA)
            subtasks = self._parse_subtasks(resp.content)

        # Completeness backstop. The prompt rule helps but an 8B router
        # intermittently drops a trailing requested action. Detection is
        # deterministic and general (a list of action verbs); the FIX is handed
        # back to the LLM (re-prompt) so filenames/steps stay model-chosen, not
        # hardcoded.
        subtasks = self._ensure_complete(instruction, user_content, subtasks)
        # Second backstop: a plan that references sub-task ids it never defined
        # was truncated (the router dropped the opening launch/navigation
        # steps). Re-prompt for the complete plan, then sanitize any remaining
        # dangling refs so _topological_sort doesn't false-cycle.
        subtasks = self._ensure_connected(instruction, user_content, subtasks)

        logger.info(f"[ROUTER] Decomposed into {len(subtasks)} sub-tasks:")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")

        return task_id, subtasks

    # ── Task-level replanning (mid-task recovery) ────────────────────────────────

    def replan(
        self,
        instruction: str,
        completed_descs: list[str],
        failed_desc: str,
        screen_context: str | None = None,
        pending_descs: list[str] | None = None,
    ) -> list[SubTask]:
        """Produce a fresh sub-task list for the work of the FAILED subtask.

        The router sees what already succeeded (never repeated), which subtask
        failed (a different approach is required), which subtasks are still
        queued (the orchestrator preserves those verbatim — re-planning them
        here would duplicate work, and on a live run the model re-deriving
        "everything remaining" silently dropped the final 'click Save' step),
        and the live screen. Returns [] when re-decomposition fails.
        """
        done_block = (
            "\n".join(f"  - {d}" for d in completed_descs)
            if completed_descs else "  (none)"
        )
        pending_block = ""
        if pending_descs:
            listed = "\n".join(f"  - {d}" for d in pending_descs)
            pending_block = (
                f"\nThese sub-tasks are still QUEUED and will run AFTER your "
                f"new plan, unchanged — do NOT include them or their work in "
                f"your answer:\n{listed}\n"
            )
        user_content = (
            f"Instruction: {instruction}\n{_today_line()}\n\n"
            f"This task is ALREADY IN PROGRESS. Sub-tasks completed successfully "
            f"(do NOT repeat them, their effects are already on the machine):\n"
            f"{done_block}\n\n"
            f"This sub-task FAILED: '{failed_desc}'\n"
            f"{pending_block}"
            f"Re-plan the FAILED sub-task's work as a fresh JSON array of "
            f"sub-tasks (you may split it into several). Use a DIFFERENT "
            f"approach (different app, method, or route — e.g. GUI instead "
            f"of terminal, or the search launcher instead of clicking an icon). "
            f"depends_on may only reference ids inside this new array.\n"
            f"THE OBJECTIVE NEVER CHANGES: your plan plus the completed and "
            f"queued sub-tasks must still accomplish the user's original goal "
            f"(re-read the Instruction above). If the current screen is a dead "
            f"end for that goal, plan to NAVIGATE AWAY (a different section, "
            f"tab, or app) — do NOT reinterpret the goal as whatever the "
            f"current screen happens to offer."
        )
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
        app_hint = self._installed_app_hint(instruction)
        if app_hint:
            user_content += f"\n\n{app_hint}"

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.2,
                response_schema=_SUBTASK_SCHEMA,
            )
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Replan parse failed: {e}")
            return []

        logger.info(f"[ROUTER] Replanned remaining work into {len(subtasks)} sub-task(s):")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")
        return subtasks

    # ── Missing-parameter detection (asked BEFORE execution starts) ─────────────

    # Instructions that involve composing/scheduling something for other people
    # commonly omit details the task cannot proceed without (a time, a recipient).
    # Deterministic trigger — the LLM is only consulted for instructions in this
    # class, so simple tasks pay zero extra latency.
    _PARAM_TRIGGER = re.compile(
        r"\b(meeting|schedule|invite|appointment|event|remind(?:er)?|"
        r"e-?mail|send|compose|book|call)\b",
        re.IGNORECASE,
    )

    def missing_parameters(self, instruction: str) -> list[str]:
        """Return up to 3 short questions about details the instruction omits
        but the task cannot be completed without. [] when nothing is missing.
        """
        if not self._PARAM_TRIGGER.search(instruction):
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You check a desktop-automation instruction for missing "
                    "REQUIRED details before an agent executes it. Output ONLY a "
                    "JSON array of short questions (strings), at most 3. Ask only "
                    "about details the task genuinely cannot be completed without "
                    "(a meeting needs a date/time; an email needs a recipient). "
                    "Never ask about details already given, optional settings, or "
                    "anything the agent can sensibly default. Output [] when "
                    "nothing essential is missing."
                ),
            },
            {"role": "user", "content": f"Instruction: {instruction}"},
        ]
        try:
            resp = self.client.query_llm(messages, max_tokens=200, temperature=0.0)
            text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL)
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                return []
            questions = json.loads(text[start:end + 1])
            return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:3]
        except Exception as e:
            logger.debug(f"[ROUTER] missing_parameters check skipped: {e}")
            return []

    # ── Completeness backstop (deterministic detection, LLM correction) ─────────

    # Trailing/finalizing actions a user commonly appends ("...and save it",
    # "...then close it"). General, not per-task: detection only — the fix is the
    # model's job. Each entry: canonical name → regex matching the user's verb.
    _FINALIZING_ACTIONS = {
        "save":     r"\bsave\b",
        "close":    r"\bclose\b",
        "send":     r"\b(?:send|e-?mail)\b",
        "print":    r"\bprint\b",
        "download": r"\bdownload\b",
        "delete":   r"\b(?:delete|remove)\b",
        "rename":   r"\brename\b",
    }

    @classmethod
    def _missing_actions(cls, instruction: str, subtasks: list[SubTask]) -> list[str]:
        """Return finalizing actions the user requested but no sub-task covers.

        Deterministic and general — it flags an OMISSION, it does not invent the
        missing step (that is the model's job on re-prompt).
        """
        instr = instruction.lower()
        covered = " ".join((st.description or "").lower() for st in subtasks)
        missing = []
        for name, pat in cls._FINALIZING_ACTIONS.items():
            if re.search(pat, instr) and not re.search(pat, covered):
                missing.append(name)
        return missing

    def _ensure_complete(
        self, instruction: str, user_content: str, subtasks: list[SubTask]
    ) -> list[SubTask]:
        """Re-prompt the router ONCE if it dropped an explicitly requested action.

        Keeps the model in charge of the fix (filenames, exact steps) — code only
        detects the gap. Accepts the retry only if it actually closes the gap;
        otherwise keeps the original so we never make things worse.
        """
        missing = self._missing_actions(instruction, subtasks)
        if not missing or not subtasks:
            return subtasks

        logger.warning(
            f"[ROUTER] Decomposition dropped requested action(s) {missing} — "
            f"re-prompting once to complete it"
        )
        prior = json.dumps([
            {"id": st.id, "description": st.description, "depends_on": st.depends_on}
            for st in subtasks
        ])
        correction = (
            f"Your sub-task list dropped these actions the user explicitly asked "
            f"for: {', '.join(missing)}. Re-output the COMPLETE ordered JSON array, "
            f"keeping every existing sub-task and adding one sub-task for each "
            f"missing action as the final step(s), with a concrete filename/target "
            f"where relevant. JSON array only."
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prior},
            {"role": "user", "content": correction},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.1,
                response_schema=_SUBTASK_SCHEMA,
            )
            fixed = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Re-prompt parse failed ({e}) — keeping original")
            return subtasks

        # Only take the retry if it genuinely covers more of what was missing.
        if len(self._missing_actions(instruction, fixed)) < len(missing) and fixed:
            return fixed
        logger.warning("[ROUTER] Re-prompt did not close the gap — keeping original")
        return subtasks

    @staticmethod
    def _dangling_deps(subtasks: list[SubTask]) -> bool:
        """True if any sub-task depends on an id not present in the list.

        The router prompt requires depends_on to reference only ids inside the
        array, so a dangling reference is a deterministic signal the model
        dropped earlier sub-tasks (live: a Teams task came back as ONLY the
        final 'fill the schedule-meeting form' sub-task with depends_on:[3];
        the three opening steps — launch, open Calendar, click New meeting —
        were gone, so the agent ran the form-fill on the bare desktop and
        clicked random taskbar buttons looking for a Title field).
        """
        ids = {st.id for st in subtasks}
        return any(dep not in ids for st in subtasks for dep in (st.depends_on or []))

    @staticmethod
    def _sanitize_deps(subtasks: list[SubTask]) -> list[SubTask]:
        """Drop depends_on references to ids that aren't in the list.

        A dangling reference makes _topological_sort report a false
        'dependency cycle' and fall back to ID order silently. Stripping the
        bad refs lets the remaining plan run in a well-defined order instead.
        """
        ids = {st.id for st in subtasks}
        for st in subtasks:
            if st.depends_on:
                st.depends_on = [d for d in st.depends_on if d in ids]
        return subtasks

    def _ensure_connected(
        self, instruction: str, user_content: str, subtasks: list[SubTask]
    ) -> list[SubTask]:
        """Re-decompose ONCE if the plan is not self-contained.

        A dangling depends_on means the router truncated the plan and dropped
        the leading steps. Re-prompt for the COMPLETE ordered plan; accept the
        retry only if it is connected and no shorter. If the retry can't fix
        it, at least sanitize the dangling refs so the plan runs in order
        rather than tripping a misleading 'dependency cycle'.
        """
        if not subtasks or not self._dangling_deps(subtasks):
            return subtasks

        logger.warning(
            "[ROUTER] Decomposition references missing sub-task ids "
            "(dangling depends_on) — earlier steps were dropped; re-prompting "
            "for the COMPLETE plan"
        )
        prior = json.dumps([
            {"id": st.id, "description": st.description, "depends_on": st.depends_on}
            for st in subtasks
        ])
        correction = (
            "Your sub-task list is INCOMPLETE: a depends_on points at an id "
            "that is not in the list, which means you dropped the earlier "
            "steps. Re-output the COMPLETE ordered plan from the FIRST action "
            "to the last — launch/open the app, navigate to the right screen, "
            "THEN act — with ids starting at 1 and every depends_on referencing "
            "an id that exists in THIS array. JSON array only."
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prior},
            {"role": "user", "content": correction},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.1,
                response_schema=_SUBTASK_SCHEMA,
            )
            fixed = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Re-prompt parse failed ({e}) — sanitizing deps")
            return self._sanitize_deps(subtasks)

        if fixed and not self._dangling_deps(fixed) and len(fixed) >= len(subtasks):
            return fixed
        logger.warning(
            "[ROUTER] Re-prompt did not produce a connected plan — sanitizing deps"
        )
        return self._sanitize_deps(subtasks)

    def _parse_subtasks(self, text: str) -> list[SubTask]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1:
            raise ValueError(f"No JSON array in router response: {text[:200]}")
        # A truncated generation (hit max_tokens mid-object) may have no
        # closing ']' at all — take everything and let the salvage below
        # recover the complete prefix.
        json_str = text[start_idx:end_idx + 1] if end_idx > start_idx else text[start_idx:]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Salvage a truncated array: cut back to the last complete object
            # and close the bracket. The complete prefix is real work the model
            # planned; anything dropped off the tail is restored by the
            # _ensure_complete() backstop, which re-prompts for missing actions.
            # Scan the FULL tail, not json_str — the rfind(']') trim above may
            # have cut at an inner bracket (e.g. "depends_on":[1]), dropping
            # the closing brace of the last complete object.
            tail = text[start_idx:]
            tail = re.sub(r",\s*([\]}])", r"\1", tail)
            tail = re.sub(r":\s*'([^']*)'", r': "\1"', tail)
            data = None
            for m in reversed(list(re.finditer(r"\}", tail))):
                try:
                    candidate = json.loads(tail[:m.end()] + "]")
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, list) and candidate:
                    data = candidate
                    logger.warning(
                        f"[ROUTER] Truncated JSON — salvaged "
                        f"{len(data)} complete sub-task(s)"
                    )
                    break
            if data is None:
                logger.error(f"[ROUTER] JSON parse error: {e}\nRaw: {json_str[:300]}")
                raise

        subtasks = []
        for item in data:
            if "description" not in item:
                logger.warning(f"[ROUTER] Skipping item with no description: {item}")
                continue
            subtasks.append(SubTask(**item))
        return subtasks

    def summarize_completion(
        self, task_id: str, completed: list, success: bool, blocker: str = "",
        skipped: list | None = None,
    ) -> str:
        content = f"Task {'succeeded' if success else 'failed'}. Sub-tasks completed: {completed}."
        if skipped:
            content += (
                f" These sub-tasks could NOT be completed and were skipped: "
                f"{skipped}. The summary MUST say the task partially "
                f"succeeded and name what was skipped."
            )
        if blocker and not success:
            content += (
                f" Screen evidence of what blocked it: {blocker}. "
                "State the blocker plainly so the user knows what to fix."
            )
        messages = [
            {"role": "system", "content": "Write a brief one-line task summary. No JSON."},
            {"role": "user", "content": content},
        ]
        resp = self.client.query_llm(messages, max_tokens=120, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
