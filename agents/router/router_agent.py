# agents/router/router_agent.py
"""Router Agent — decomposes user instructions into sub-tasks."""

import json
import os
import platform
import re
import uuid
from typing import List, Optional, Tuple

from loguru import logger

from core.protocols.a2a import InferenceClient, SubTask
from utils.platform_utils import detect_firefox

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

# ── Runtime machine identity ───────────────────────────────────────────────────
_OS = platform.system()
_USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
_SHELL_PROMPT = _USER  # username only — hostnames can be too long for OCR

if _OS == "Windows":
    _ROUTER_OS_CONTEXT = "Windows 11"
    _DESKTOP_PATH = "%USERPROFILE%\\Desktop"
    _TERMINAL_APP = "Windows Terminal"
    _CALC_APP     = "Calculator"
    _FILES_APP    = "File Explorer"
    _SETTINGS_APP = "Settings"
elif _OS == "Darwin":
    _ROUTER_OS_CONTEXT = "macOS"
    _DESKTOP_PATH = "~/Desktop"
    _TERMINAL_APP = "Terminal"
    _CALC_APP     = "Calculator"
    _FILES_APP    = "Finder"
    _SETTINGS_APP = "System Settings"
else:
    _ROUTER_OS_CONTEXT = "Linux (GNOME)"
    _DESKTOP_PATH = "~/Desktop"
    _TERMINAL_APP = "GNOME Terminal"
    _CALC_APP     = "GNOME Calculator"
    _FILES_APP    = "Nautilus/Files"
    _SETTINGS_APP = "GNOME Settings"


_FIREFOX_CMD = detect_firefox()
_FIREFOX_LAUNCH = _FIREFOX_CMD if _OS == "Windows" else f"{_FIREFOX_CMD} &"

# ── Router system prompt ───────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are a desktop automation coordinator on ROUTER_OS_PLACEHOLDER.
Decompose any user instruction into the MINIMUM ordered sub-tasks a GUI agent can execute.

━━━ CORE RULES ━━━
1. One distinct action per sub-task (launch app / navigate / click / type / run command).
2. Minimum sub-tasks — never add unnecessary confirmation, wait, or close steps.
3. Set depends_on so each sub-task runs after its prerequisites complete.
   CRITICAL: If sub-task B opens/reads a file that sub-task A creates, set B's depends_on to [A_id].
   Example: create file (id=2) then open in Notepad (id=3) → Notepad sub-task has depends_on:[2].
4. Descriptions must be SPECIFIC — include exact URLs, filenames, commands, and app names.
5. STATE CONTEXT in every dependent sub-task description so the planner knows what is already open:
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
  File / folder ops   →  terminal (touch / mkdir / rm / mv / echo / cp). NEVER use the file manager.
  Web browsing        →  the browser named in the instruction, else Firefox (specify exact URL or search query)
  Code editing        →  VS Code
  Documents           →  LibreOffice Writer / Calc / Impress
  Email               →  Thunderbird
  Calculator          →  CALC_APP_PLACEHOLDER
  System settings     →  SETTINGS_APP_PLACEHOLDER
  Screenshot          →  Print Screen key (one sub-task, no app needed)
  Simple text files   →  echo command in terminal (single line) or nano (multi-line)

━━━ AVAILABLE APPS ━━━
Trust the app name the user gives you (browsers, games, utilities — anything
installed). When the instruction names no specific app for a generic task, default to:
Firefox, VS Code, LibreOffice Writer/Calc/Impress, Thunderbird,
TERMINAL_APP_PLACEHOLDER, CALC_APP_PLACEHOLDER, SETTINGS_APP_PLACEHOLDER, FILES_APP_PLACEHOLDER.
NEVER suggest: gedit, mousepad, kate, VLC, GIMP, or anything not listed above.
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

"open a spreadsheet and enter sales data in cell A1"
→ [{"id":1,"description":"open LibreOffice Calc","depends_on":[]},
   {"id":2,"description":"with LibreOffice Calc open, click on cell A1 and enter the sales data","depends_on":[1]}]

"compose a new email in thunderbird"
→ [{"id":1,"description":"open Thunderbird","depends_on":[]},
   {"id":2,"description":"with Thunderbird open, click the Write new message button","depends_on":[1]}]

"take a screenshot"
→ [{"id":1,"description":"take a screenshot using the Print Screen keyboard shortcut","depends_on":[]}]

"open system settings"
→ [{"id":1,"description":"open SETTINGS_APP_PLACEHOLDER","depends_on":[]}]

"calculate 15 percent of 200"
→ [{"id":1,"description":"open CALC_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the calculator open, compute 15 percent of 200","depends_on":[1]}]

"open terminal, create notes.txt on the desktop, write hello world in it, then open it in notepad"
→ [{"id":1,"description":"open TERMINAL_APP_PLACEHOLDER","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run: echo hello world > DESKTOP_PATH_PLACEHOLDER\\notes.txt","depends_on":[1]},
   {"id":3,"description":"open Notepad","depends_on":[2]},
   {"id":4,"description":"with Notepad already open, open the file DESKTOP_PATH_PLACEHOLDER\\notes.txt","depends_on":[3]}]"""

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
    def __init__(self, ovms_client: InferenceClient):
        self.ovms = ovms_client

    def decompose(
        self,
        instruction: str,
        screen_context: Optional[str] = None,
        memory_hint: Optional[str] = None,
    ) -> Tuple[str, List[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}"
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
        resp = self.ovms.query_llm(messages, max_tokens=768, temperature=0.1,
                                   response_schema=_SUBTASK_SCHEMA)
        try:
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("[ROUTER] Parse failed — retrying with schema-only prompt")
            retry_messages = [
                {"role": "system", "content": "Output ONLY a JSON array of sub-tasks."},
                {"role": "user", "content": f"Sub-tasks for: {instruction}"},
            ]
            resp = self.ovms.query_llm(retry_messages, max_tokens=512, temperature=0.0,
                                       response_schema=_SUBTASK_SCHEMA)
            subtasks = self._parse_subtasks(resp.content)

        logger.info(f"[ROUTER] Decomposed into {len(subtasks)} sub-tasks:")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")

        return task_id, subtasks

    def _parse_subtasks(self, text: str) -> List[SubTask]:
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
        resp = self.ovms.query_llm(messages, max_tokens=80, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
