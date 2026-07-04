# agents/router.py
"""Router Agent — decomposes user instructions into sub-tasks."""

import json
import os
import re
import uuid

from loguru import logger

from core.protocols import InferenceClient, SubTask
from utils.platform_utils import detect_firefox, get_desktop_path

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
_TERMINAL_APP = "Windows Terminal"
_CALC_APP     = "Calculator"
_FILES_APP    = "File Explorer"
_SETTINGS_APP = "Settings"

_FIREFOX_CMD = detect_firefox()
_FIREFOX_LAUNCH = _FIREFOX_CMD

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
     "with Firefox already open, navigate to <url>"
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
  Web browsing        →  the browser named in the instruction, else Firefox (specify exact URL or search query)
  Code editing        →  VS Code
  Documents           →  LibreOffice Writer / Calc / Impress
  Email               →  Thunderbird
  Calculator          →  CALC_APP_PLACEHOLDER
  System settings     →  SETTINGS_APP_PLACEHOLDER
  Screenshot          →  Print Screen key (one sub-task, no app needed)
  Simple text files   →  echo command in terminal (single line) or nano (multi-line)
  Meetings / calls    →  the app the user names (Zoom, Teams, Skype, …), else the
                         browser. Launch it → open its New/Schedule form → fill the
                         form → confirm. See FORM FILLING below.

━━━ FORM FILLING (schedule a meeting, compose an email, create an event) ━━━
  Decompose form work into exactly THREE kinds of sub-task:
    1. one sub-task to OPEN the form ("with Zoom already open, click the
       Schedule button to open the schedule-meeting form")
    2. ONE sub-task that fills ALL the fields, listing each field and its exact
       value ("with the schedule form open, set Topic to 'Weekly Sync', set
       Date to 07/10/2026, set Start time to 3:00 PM, set Duration to 30 min")
       — related fields of one form always belong in ONE sub-task, never one
       sub-task per field.
    3. one sub-task to CONFIRM ("with the details filled, click Save") and, if
       the user asked to invite people, one more to send/copy the invitation.
  Use concrete values: resolve "tomorrow" to the actual date, "3pm" to 3:00 PM.

━━━ AVAILABLE APPS ━━━
Trust the app name the user gives you (browsers, meeting apps, games,
utilities — anything installed): an app the user NAMES is always allowed.
When the instruction names no specific app for a generic task, default to:
Firefox, VS Code, LibreOffice Writer/Calc/Impress, Thunderbird,
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

"open brave browser"
→ [{"id":1,"description":"open Brave Browser","depends_on":[]}]

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

"open firefox and go to github.com"
→ [{"id":1,"description":"open Firefox","depends_on":[]},
   {"id":2,"description":"with Firefox already open, navigate to https://github.com","depends_on":[1]}]

"search for openai on google and click the first result"
→ [{"id":1,"description":"open Firefox","depends_on":[]},
   {"id":2,"description":"with Firefox already open, search for openai on Google","depends_on":[1]},
   {"id":3,"description":"with Google results open in Firefox, click the first search result link","depends_on":[2]}]

"open youtube and search for python tutorial"
→ [{"id":1,"description":"open Firefox","depends_on":[]},
   {"id":2,"description":"with Firefox already open, navigate to https://www.youtube.com","depends_on":[1]},
   {"id":3,"description":"with YouTube open in Firefox, search for python tutorial","depends_on":[2]}]

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

"schedule a zoom meeting titled Weekly Sync tomorrow at 3pm for 30 minutes" (today = 07/09/2026)
→ [{"id":1,"description":"open Zoom","depends_on":[]},
   {"id":2,"description":"with Zoom already open, click the Schedule button to open the schedule-meeting form","depends_on":[1]},
   {"id":3,"description":"with the schedule-meeting form open, set Topic to 'Weekly Sync', set the date to 07/10/2026, set the start time to 3:00 PM, set the duration to 30 minutes","depends_on":[2]},
   {"id":4,"description":"with the meeting details filled in, click Save to create the meeting","depends_on":[3]}]

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
    .replace("FIREFOX_LAUNCH_PLACEHOLDER", _FIREFOX_LAUNCH)
    .replace("DESKTOP_PATH_PLACEHOLDER", _DESKTOP_PATH)
    .replace("TERMINAL_APP_PLACEHOLDER", _TERMINAL_APP)
    .replace("CALC_APP_PLACEHOLDER", _CALC_APP)
    .replace("SETTINGS_APP_PLACEHOLDER", _SETTINGS_APP)
    .replace("FILES_APP_PLACEHOLDER", _FILES_APP)
)


class RouterAgent:
    def __init__(self, client: InferenceClient):
        self.client = client

    def decompose(
        self,
        instruction: str,
        screen_context: str | None = None,
        memory_hint: str | None = None,
    ) -> tuple[str, list[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}\n{_today_line()}"
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
        resp = self.client.query_llm(messages, max_tokens=768, temperature=0.1,
                                   response_schema=_SUBTASK_SCHEMA)
        try:
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("[ROUTER] Parse failed — retrying with schema-only prompt")
            retry_messages = [
                {"role": "system", "content": "Output ONLY a JSON array of sub-tasks."},
                {"role": "user", "content": f"Sub-tasks for: {instruction}"},
            ]
            resp = self.client.query_llm(retry_messages, max_tokens=512, temperature=0.0,
                                       response_schema=_SUBTASK_SCHEMA)
            subtasks = self._parse_subtasks(resp.content)

        # Completeness backstop. The prompt rule helps but an 8B router
        # intermittently drops a trailing requested action. Detection is
        # deterministic and general (a list of action verbs); the FIX is handed
        # back to the LLM (re-prompt) so filenames/steps stay model-chosen, not
        # hardcoded.
        subtasks = self._ensure_complete(instruction, user_content, subtasks)

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
    ) -> list[SubTask]:
        """Produce a fresh sub-task list covering ONLY the remaining work after a
        subtask failed mid-task.

        The router sees what already succeeded (never repeated), which subtask
        failed (a different approach is required), and the live screen. Returns
        [] when re-decomposition fails — the caller then aborts as before.
        """
        done_block = (
            "\n".join(f"  - {d}" for d in completed_descs)
            if completed_descs else "  (none)"
        )
        user_content = (
            f"Instruction: {instruction}\n{_today_line()}\n\n"
            f"This task is ALREADY IN PROGRESS. Sub-tasks completed successfully "
            f"(do NOT repeat them, their effects are already on the machine):\n"
            f"{done_block}\n\n"
            f"This sub-task FAILED: '{failed_desc}'\n"
            f"Re-plan the REMAINING work as a fresh JSON array of sub-tasks that "
            f"still achieves the user's full intent. Use a DIFFERENT approach for "
            f"the failed part (different app, method, or route — e.g. GUI instead "
            f"of terminal, search launcher instead of icon, browser instead of a "
            f"desktop app). depends_on may only reference ids inside this new array."
        )
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=768, temperature=0.2,
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
                messages, max_tokens=768, temperature=0.1,
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

    def _parse_subtasks(self, text: str) -> list[SubTask]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"No JSON array in router response: {text[:200]}")

        json_str = text[start_idx:end_idx + 1]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[ROUTER] JSON parse error: {e}\nRaw: {json_str[:300]}")
            raise

        subtasks = []
        for item in data:
            if "description" not in item:
                logger.warning(f"[ROUTER] Skipping item with no description: {item}")
                continue
            subtasks.append(SubTask(**item))
        return subtasks

    def summarize_completion(self, task_id: str, completed: list, success: bool) -> str:
        messages = [
            {"role": "system", "content": "Write a brief one-line task summary. No JSON."},
            {"role": "user", "content": f"Task {'succeeded' if success else 'failed'}. Sub-tasks completed: {completed}."},
        ]
        resp = self.client.query_llm(messages, max_tokens=80, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
