# agents/router/router_agent.py
"""Router Agent — decomposes user instructions into sub-tasks."""

import json
import os
import platform
import re
import shutil
import socket
import uuid
from typing import List, Optional, Tuple

from loguru import logger

from core.protocols.a2a import InferenceClient, SubTask

# ── Runtime machine identity (same logic as planning_agent) ───────────────────
_OS = platform.system()
_USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
_SHELL_PROMPT = _USER  # username only — hostnames can be too long for OCR

if _OS == "Windows":
    _ROUTER_OS_CONTEXT = "Windows 11"
    _DESKTOP_PATH = "%USERPROFILE%\\Desktop"
elif _OS == "Darwin":
    _ROUTER_OS_CONTEXT = "macOS"
    _DESKTOP_PATH = "~/Desktop"
else:
    _ROUTER_OS_CONTEXT = "Linux (GNOME)"
    _DESKTOP_PATH = "~/Desktop"


def _detect_firefox() -> str:
    if _OS == "Windows":
        import winreg
        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\firefox.exe"
            ) as k:
                path = winreg.QueryValue(k, None)
                if path and os.path.exists(path):
                    return f'"{path}"'
        except Exception:
            pass
        for path in [
            os.path.expandvars(r"%ProgramFiles%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Mozilla Firefox\firefox.exe"),
        ]:
            if os.path.exists(path):
                return f'"{path}"'
        return "firefox"
    which = shutil.which("firefox")
    if which:
        return which
    for path in [
        os.path.expanduser("~/apps/firefox/firefox/firefox"),
        os.path.expanduser("~/firefox/firefox"),
        "/snap/bin/firefox",
        "/usr/bin/firefox",
        "/usr/local/bin/firefox",
        "/opt/firefox/firefox",
    ]:
        if os.path.exists(path):
            return path
    return "firefox"


_FIREFOX_CMD = _detect_firefox()
_FIREFOX_LAUNCH = _FIREFOX_CMD if _OS == "Windows" else f"{_FIREFOX_CMD} &"

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

ROUTER_SYSTEM_PROMPT = """You are a desktop automation task coordinator running on Linux (Ubuntu 22.04 / GNOME).
Break down user instructions into the minimum number of ordered sub-tasks a desktop agent can execute.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALLED APPS — only use these, never suggest others
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Browser      : Firefox — launch via Activities search "firefox" OR via terminal: /home/shehroz/apps/firefox/firefox/firefox &
Code editor  : VS Code  (command: code)
Office suite : LibreOffice Writer, Calc, Impress, Draw
Email        : Thunderbird (snap)
Terminal     : GNOME Terminal
Text editors : nano, vim  (terminal only — no GUI text editor installed)
File manager : Nautilus / Files  (only for browsing files visually)
System apps  : GNOME Calculator, GNOME Settings, Screenshot tool

DO NOT suggest gedit, mousepad, notepad, kate, sublime, atom, VLC, GIMP, or any
app not in the list above. If a task needs a text editor, use nano in terminal
(for simple text) or LibreOffice Writer (for formatted documents).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECOMPOSITION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. One distinct GUI action per sub-task (open app / navigate / type / click / run command).
2. Strict logical order. Set depends_on so later steps wait for earlier ones.
3. Never merge two different actions into one sub-task.
4. Never add wait / verify / confirm / close steps unless explicitly requested.
5. Every object: id (integer), description (string), depends_on (integer array).
6. Sub-task descriptions must be SPECIFIC — include exact filenames, URLs, commands,
   and app names so the planning agent never has to guess.
7. If screen context shows an app already open, skip its launch sub-task.
8. ALWAYS generate a launch sub-task when the user instruction says "open <app>" — even if screen context shows it open. The user is explicitly asking to open it.
8. STATE CONTEXT IN DESCRIPTIONS — when a sub-task depends on a previous one that
   opened an app, the description MUST say so explicitly. This tells the planner
   what is already open so it does not re-launch anything.
   PATTERN: "with the terminal already open, run the command: <cmd>"
   PATTERN: "with Firefox already open, navigate to <url>"
   PATTERN: "with Firefox already open, click on <target>"
   This is MANDATORY for any sub-task that depends_on an app-launch sub-task.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METHOD SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FILE / FOLDER CREATE, DELETE, MOVE, RENAME, WRITE:
  → Always use the terminal with shell commands. Never use the file manager.
  → Desktop path: ~/Desktop/
  → Create file    : touch ~/Desktop/filename.ext
  → Write to file  : echo 'content' > ~/Desktop/filename.ext
  → Append to file : echo 'content' >> ~/Desktop/filename.ext
  → Create folder  : mkdir ~/Desktop/foldername
  → Delete file    : rm ~/Desktop/filename.ext
  → Rename/move    : mv ~/Desktop/old.ext ~/Desktop/new.ext
  → Copy file      : cp ~/Desktop/src.ext ~/Desktop/dst.ext
  → Multi-line file: use nano — open terminal → nano ~/Desktop/filename → type → Ctrl+O → Enter → Ctrl+X

TEXT EDITING (simple content, no formatting needed):
  → Use nano in terminal: open terminal → nano ~/Desktop/filename.txt → edit → Ctrl+O → Enter → Ctrl+X

DOCUMENTS (formatted, .odt, .docx, spreadsheet, presentation):
  → Use LibreOffice Writer / Calc / Impress.
  → Sub-tasks: open app → type/edit content → save with Ctrl+S or Save As.

WEB BROWSING:
  → Use Firefox. Always specify exact URL or search query.
  → Sub-tasks: open Firefox → navigate/search → perform action on page.

EMAIL:
  → Use Thunderbird.

CODING / DEVELOPMENT:
  → Edit code: VS Code. Run code, git, pip, installs: terminal.

SYSTEM:
  → Settings: GNOME Settings app.
  → Screenshot: keyboard shortcut (Print Screen).
  → Volume/brightness: system tray icons.
  → Calculator: GNOME Calculator app.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

--- FILE OPERATIONS (terminal) ---

"create a file named notes.txt on desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: touch ~/Desktop/notes.txt","depends_on":[1]}]

"create a file named gui_agent on desktop and write hello in it"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: echo 'hello' > ~/Desktop/gui_agent","depends_on":[1]}]

"create a folder named projects on the desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: mkdir ~/Desktop/projects","depends_on":[1]}]

"delete the file test.txt from the desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: rm ~/Desktop/test.txt","depends_on":[1]}]

"rename file old.txt to new.txt on the desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: mv ~/Desktop/old.txt ~/Desktop/new.txt","depends_on":[1]}]

"copy file report.txt from Downloads to Desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: cp ~/Downloads/report.txt ~/Desktop/","depends_on":[1]}]

--- TEXT EDITING (nano in terminal) ---

"create a file named notes.txt on desktop and write Hello World in it"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"run the command: echo 'Hello World' > ~/Desktop/notes.txt","depends_on":[1]}]

"open notes.txt from the desktop in a text editor and add a new line saying goodbye"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"run the command: echo 'goodbye' >> ~/Desktop/notes.txt","depends_on":[1]}]

--- LIBREOFFICE DOCUMENTS ---

"open LibreOffice Writer and create a new document"
→ [{"id":1,"description":"open LibreOffice Writer","depends_on":[]},
   {"id":2,"description":"start typing in the blank LibreOffice Writer document","depends_on":[1]}]

"open a spreadsheet and enter some data"
→ [{"id":1,"description":"open LibreOffice Calc","depends_on":[]},
   {"id":2,"description":"click on cell A1 and enter the data","depends_on":[1]}]

--- WEB BROWSING ---

"open Firefox and go to github.com (Firefox is NOT already open)"
→ [{"id":1,"description":"open Firefox by running terminal command: /home/shehroz/apps/firefox/firefox/firefox &","depends_on":[]},
   {"id":2,"description":"with Firefox already open, navigate to https://github.com","depends_on":[1]}]

"search for intel openvino on Google"
→ [{"id":1,"description":"open Firefox by running terminal command: /home/shehroz/apps/firefox/firefox/firefox &","depends_on":[]},
   {"id":2,"description":"with Firefox already open, search for 'intel openvino' on Google using the address bar","depends_on":[1]}]

"open youtube.com and search for python tutorial"
→ [{"id":1,"description":"open Firefox by running terminal command: /home/shehroz/apps/firefox/firefox/firefox &","depends_on":[]},
   {"id":2,"description":"with Firefox already open, navigate to https://www.youtube.com","depends_on":[1]},
   {"id":3,"description":"with YouTube open in Firefox, type 'python tutorial' in the YouTube search bar and press enter","depends_on":[2]}]

"open a new tab in Firefox and go to stackoverflow.com"
→ [{"id":1,"description":"open a new tab in Firefox using Ctrl+T","depends_on":[]},
   {"id":2,"description":"with Firefox already open on a new tab, navigate to https://stackoverflow.com","depends_on":[1]}]

--- TERMINAL / DEVELOPMENT ---

"open terminal and run python3 --version"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: python3 --version","depends_on":[1]}]

"install the requests python package"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: pip install requests","depends_on":[1]}]

"run the script main.py on the desktop"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: python3 ~/Desktop/main.py","depends_on":[1]}]

"open VS Code"
→ [{"id":1,"description":"open Visual Studio Code","depends_on":[]}]

"open VS Code and open the folder ~/Desktop/myproject"
→ [{"id":1,"description":"open Visual Studio Code","depends_on":[]},
   {"id":2,"description":"with VS Code already open, open the folder ~/Desktop/myproject using File > Open Folder","depends_on":[1]}]

"check what python version is installed"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: python3 --version","depends_on":[1]}]

--- SYSTEM OPERATIONS ---

"take a screenshot"
→ [{"id":1,"description":"take a screenshot using the Print Screen keyboard shortcut","depends_on":[]}]

"open system settings"
→ [{"id":1,"description":"open GNOME Settings application","depends_on":[]}]

"increase the system volume"
→ [{"id":1,"description":"click the volume icon in the system tray and increase the volume","depends_on":[]}]

--- CALCULATOR ---

"open the calculator"
→ [{"id":1,"description":"open the GNOME Calculator application","depends_on":[]}]

"calculate 15 percent of 200"
→ [{"id":1,"description":"open the GNOME Calculator application","depends_on":[]},
   {"id":2,"description":"calculate 15 percent of 200 using the calculator","depends_on":[1]}]

--- EMAIL ---

"open email"
→ [{"id":1,"description":"open Thunderbird email client","depends_on":[]}]

"compose a new email"
→ [{"id":1,"description":"open Thunderbird email client","depends_on":[]},
   {"id":2,"description":"click the Write or New Message button in Thunderbird","depends_on":[1]}]

--- COMPLEX MULTI-STEP ---

"create a python script on the desktop that prints hello world and run it"
→ [{"id":1,"description":"open the terminal","depends_on":[]},
   {"id":2,"description":"with the terminal already open, run the command: echo 'print(\"hello world\")' > ~/Desktop/hello.py","depends_on":[1]},
   {"id":3,"description":"with the terminal already open, run the command: python3 ~/Desktop/hello.py","depends_on":[2]}]

"search for openai on google and open the first result"
→ [{"id":1,"description":"open Firefox by running terminal command: /home/shehroz/apps/firefox/firefox/firefox &","depends_on":[]},
   {"id":2,"description":"with Firefox already open, search for 'openai' on Google using the address bar","depends_on":[1]},
   {"id":3,"description":"with Google search results open in Firefox, click on the first search result link","depends_on":[2]}]

"open VS Code and create a new python file named app.py"
→ [{"id":1,"description":"open Visual Studio Code","depends_on":[]},
   {"id":2,"description":"with VS Code already open, create a new file named app.py using Ctrl+N then save as app.py","depends_on":[1]}]

Output ONLY a valid JSON array. No markdown. No explanation. No preamble."""

# Apply runtime substitutions so no machine-specific values are hardcoded
ROUTER_SYSTEM_PROMPT = (
    ROUTER_SYSTEM_PROMPT
    .replace("Linux (Ubuntu 22.04 / GNOME)", _ROUTER_OS_CONTEXT)
    .replace("/home/shehroz/apps/firefox/firefox/firefox &", _FIREFOX_LAUNCH)
    .replace("~/Desktop/", _DESKTOP_PATH + "/")
    .replace("~/Desktop", _DESKTOP_PATH)
)


class RouterAgent:
    def __init__(self, ovms_client: InferenceClient):
        self.ovms = ovms_client

    def decompose(
        self,
        instruction: str,
        screen_context: Optional[str] = None,
    ) -> Tuple[str, List[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}"
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
            user_content += (
                "\nOnly include sub-tasks for things NOT already done. "
                "If the target app is already open and visible, skip its launch sub-task."
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
        resp = self.ovms.query_llm(messages, max_tokens=100, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()