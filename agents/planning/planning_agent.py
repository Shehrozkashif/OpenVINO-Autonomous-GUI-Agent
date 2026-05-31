# agents/planning/planning_agent.py
"""
Planning Agent — generates precise step sequences for sub-tasks.
Uses Chain-of-Thought prompting for complex multi-step tasks.
"""

import json
import os
import platform
import re
import shutil
import socket
from typing import List

from loguru import logger

from core.protocols.a2a import ActionStep, InferenceClient, SubTask

_OS = platform.system()
if _OS == "Windows":
    _OS_CONTEXT     = "Microsoft Windows 11 desktop"
    _LAUNCHER_KEY   = "winleft"
    _LAUNCHER_NAME  = "Windows Start menu search"
    _CLOSE_WIN      = "alt+f4"
    _NEW_TAB        = "ctrl+t"
    _REOPEN_TAB     = "ctrl+shift+t"
    _DESKTOP_PATH   = "%USERPROFILE%\\Desktop"
    _SCREENSHOT_KEY = "print_screen"
elif _OS == "Darwin":
    _OS_CONTEXT     = "macOS desktop (Ventura/Sonoma)"
    _LAUNCHER_KEY   = "command+space"
    _LAUNCHER_NAME  = "Spotlight search"
    _CLOSE_WIN      = "command+w"
    _NEW_TAB        = "command+t"
    _REOPEN_TAB     = "command+shift+t"
    _DESKTOP_PATH   = "~/Desktop"
    _SCREENSHOT_KEY = "command+shift+3"
else:
    _OS_CONTEXT     = "Linux desktop (GNOME)"
    _LAUNCHER_KEY   = "super"
    _LAUNCHER_NAME  = "GNOME Activities overview"
    _CLOSE_WIN      = "alt+f4"
    _NEW_TAB        = "ctrl+t"
    _REOPEN_TAB     = "ctrl+shift+t"
    _DESKTOP_PATH   = "~/Desktop"
    _SCREENSHOT_KEY = "print_screen"

# ── Runtime machine identity ──────────────────────────────────────────────────
_USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
# Use only the username — hostnames can be very long (e.g. laptop model names)
# and OCR reliably finds the short username portion of the shell prompt.
_SHELL_PROMPT = _USER


def _detect_firefox() -> str:
    """Return the best available Firefox launch command on this machine."""
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
    # Linux / macOS — prefer PATH, then common install locations
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
# On Linux/macOS run Firefox in the background so the terminal stays usable
_FIREFOX_LAUNCH = _FIREFOX_CMD if _OS == "Windows" else f"{_FIREFOX_CMD} &"

_STEP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":           {"type": "integer"},
            "action_type":  {"type": "string", "enum": [
                "click", "right_click", "double_click",
                "type", "key_press", "hotkey", "scroll", "wait",
            ]},
            "target":       {"type": ["string", "null"]},
            "value":        {"type": ["string", "null"]},
            "key":          {"type": ["string", "null"]},
            "description":  {"type": "string"},
            "verification": {"type": "string"},
        },
        "required": ["id", "action_type", "target", "value", "key",
                     "description", "verification"],
    },
}

PLANNING_SYSTEM_PROMPT = f"""You are a desktop automation agent on {_OS_CONTEXT}.
Turn each sub-task into the SHORTEST correct sequence of atomic actions.

━━━ DECISION TREE — follow this order every time ━━━
1. READ screen context. If the target element or app icon is visible → click it. ONE step. Stop.
2. USE a keyboard shortcut if one exists (ctrl+l for browser bar, etc.).
3. USE the search launcher only as a last resort when 1 and 2 don't apply.

━━━ ACTION REFERENCE ━━━
click / right_click / double_click  →  target = exact visible text label (1-4 words MAX, never null)
                                       GOOD targets: "Calculator" "Code" "File" "Save" "OK" "shehrozbaloch"
                                       BAD targets:  "GNOME Calculator icon" "the VS Code app" "click here"
type                                →  value  = exact string to type (never null)
key_press                           →  key    = single key name:
                                         enter escape tab super backspace delete space
                                         f1-f12 up down left right home end print_screen
hotkey                              →  key    = key combination:
                                         ctrl+s  ctrl+c  ctrl+v  ctrl+z  ctrl+a  ctrl+l
                                         ctrl+t  ctrl+w  ctrl+f  ctrl+p  ctrl+n  ctrl+o
                                         ctrl+shift+s  ctrl+shift+p  ctrl+alt+t
                                         alt+f4  alt+tab  alt+left  super+d  super+l
scroll                              →  target = element to scroll over, value = "up" or "down"
wait                                →  value  = seconds as string: "0.5" "1.0" "2.0" "3.0"

━━━ LAUNCHING APPS ━━━
Use the search launcher for ALL apps — it works on every machine without configuration.
Do NOT use ctrl+alt+t for terminal — the shortcut may not be configured.

Search launcher pattern (use for every app including terminal):
    key_press "{_LAUNCHER_KEY}"  →  click "Type to search"  →  type "<app name>"  →  key_press "enter"
    wait 1.5–2.0s after launch, then click the shell prompt or window to confirm focus.

━━━ FOCUS MANAGEMENT ━━━
• Click a window before typing in it. Always. Except: fresh terminal (ctrl+alt+t) already has focus.
• Focus browser address bar  →  hotkey ctrl+l  (NEVER click the address bar visually)
• Focus terminal already open  →  click the username visible in the shell prompt (e.g. {_SHELL_PROMPT})
• After alt+tab or clicking taskbar  →  always click target window before typing

━━━ FIREFOX ━━━
Focus address bar  :  hotkey ctrl+l
Navigate to URL    :  ctrl+l → type URL → key_press enter
Web search         :  ctrl+l → type query → key_press enter
New tab            :  hotkey ctrl+t
Close tab          :  hotkey ctrl+w
Find on page       :  hotkey ctrl+f → type → key_press enter → key_press escape
Scroll page        :  scroll target="page content" value="down"
Go back            :  hotkey alt+left
Bookmark           :  hotkey ctrl+d
Downloads          :  hotkey ctrl+j

━━━ TERMINAL ━━━
Desktop path : {_DESKTOP_PATH}

Fresh terminal (use search launcher — reliable on all machines):
  key_press "{_LAUNCHER_KEY}" → click "Type to search" → type "gnome-terminal" → key_press enter → wait "2.0" → click {_SHELL_PROMPT} → type command → key_press enter

Terminal already open (from previous sub-task):
  click {_SHELL_PROMPT} → wait "0.5" → type command → key_press enter

Common commands:
  Create file  :  touch {_DESKTOP_PATH}/name.txt
  Write file   :  echo 'text' > {_DESKTOP_PATH}/name.txt
  Append file  :  echo 'text' >> {_DESKTOP_PATH}/name.txt
  Make folder  :  mkdir {_DESKTOP_PATH}/foldername
  Delete file  :  rm {_DESKTOP_PATH}/name.txt
  Move/rename  :  mv {_DESKTOP_PATH}/old {_DESKTOP_PATH}/new
  Copy file    :  cp source destination
  Run Python   :  python3 {_DESKTOP_PATH}/script.py
  Install pkg  :  pip install packagename
  List files   :  ls {_DESKTOP_PATH}
  Git clone    :  git clone https://github.com/user/repo

━━━ VS CODE ━━━
Launch   :  search launcher value="code" → wait "2.0"
Open folder  :  hotkey ctrl+k, then ctrl+o → navigate → key_press enter
New file     :  hotkey ctrl+n
Save         :  hotkey ctrl+s
Terminal     :  hotkey ctrl+grave
Command palette  :  hotkey ctrl+shift+p → type command → key_press enter

━━━ LIBREOFFICE ━━━
Launch Writer  :  search launcher value="libreoffice writer" → wait "2.0"
Launch Calc    :  search launcher value="libreoffice calc" → wait "2.0"
Click in doc before typing. Save: ctrl+s. Save As: ctrl+shift+s → type name → enter.

━━━ STRICT RULES ━━━
✓ Exact visible text as click target — never "button", "icon", "link"
✓ Click a window before typing in it (except fresh terminal)
✓ Combine all related text into ONE type step — never chain two type steps
✓ ctrl+l to focus browser address bar — never click the bar visually
✗ Never use gedit, mousepad, kate, VLC, GIMP — use nano or LibreOffice
✗ Never open Activities/search when a visible icon or hotkey works
✗ Never add steps just to be safe — minimum steps only
✗ Never type in terminal without first clicking the shell prompt (if terminal was already open)
✗ Never re-launch an app that the sub-task description says is already open

━━━ OUTPUT ━━━
Valid JSON array only. All 7 fields required. Unused fields = null. IDs start at 1.

━━━ EXAMPLES ━━━

EXAMPLE 1 — app icon visible in screen context (screen shows "Code"):
[
  {{"id":1,"action_type":"click","target":"Code","value":null,"key":null,"description":"Click VS Code icon visible in taskbar","verification":"VS Code window opens or comes to front"}},
  {{"id":2,"action_type":"wait","target":null,"value":"1.0","key":null,"description":"Wait for VS Code to load","verification":"VS Code editor is visible"}}
]

EXAMPLE 2 — open terminal and run a command (use search launcher, always reliable):
[
  {{"id":1,"action_type":"key_press","target":null,"value":null,"key":"{_LAUNCHER_KEY}","description":"Open {_LAUNCHER_NAME}","verification":"Search overlay appears"}},
  {{"id":2,"action_type":"click","target":"Type to search","value":null,"key":null,"description":"Focus search bar","verification":"Cursor in search bar"}},
  {{"id":3,"action_type":"type","target":null,"value":"gnome-terminal","key":null,"description":"Type app name","verification":"Terminal result visible"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Launch terminal","verification":"Terminal window opens"}},
  {{"id":5,"action_type":"wait","target":null,"value":"2.0","key":null,"description":"Wait for shell prompt","verification":"Shell prompt visible"}},
  {{"id":6,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Click shell prompt to confirm focus","verification":"Terminal is active and focused"}},
  {{"id":7,"action_type":"type","target":null,"value":"echo 'hello world' > {_DESKTOP_PATH}/hello.txt","key":null,"description":"Type command","verification":"Command visible at prompt"}},
  {{"id":8,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute command","verification":"New prompt appears, no error"}}
]

EXAMPLE 3 — terminal already open from previous sub-task:
[
  {{"id":1,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Click username to focus terminal","verification":"Terminal is active"}},
  {{"id":2,"action_type":"wait","target":null,"value":"0.5","key":null,"description":"Wait for focus","verification":"Cursor blinking"}},
  {{"id":3,"action_type":"type","target":null,"value":"mkdir {_DESKTOP_PATH}/projects","key":null,"description":"Type command","verification":"Command at prompt"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute","verification":"New prompt, no error"}}
]

EXAMPLE 4 — Firefox already open, navigate to URL:
[
  {{"id":1,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+l","description":"Focus address bar","verification":"Address bar highlighted"}},
  {{"id":2,"action_type":"type","target":null,"value":"https://github.com","key":null,"description":"Type URL","verification":"URL visible in address bar"}},
  {{"id":3,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Navigate","verification":"GitHub page loads"}}
]

EXAMPLE 5 — launch app with search (no icon visible, no hotkey):
[
  {{"id":1,"action_type":"key_press","target":null,"value":null,"key":"{_LAUNCHER_KEY}","description":"Open {_LAUNCHER_NAME}","verification":"Search overlay visible"}},
  {{"id":2,"action_type":"click","target":"Type to search","value":null,"key":null,"description":"Focus search bar","verification":"Cursor in search bar"}},
  {{"id":3,"action_type":"type","target":null,"value":"libreoffice writer","key":null,"description":"Type app name","verification":"LibreOffice result visible"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Launch","verification":"LibreOffice Writer opens"}},
  {{"id":5,"action_type":"wait","target":null,"value":"2.0","key":null,"description":"Wait for app to load","verification":"Document area visible"}}
]

EXAMPLE 6 — type in a LibreOffice Writer document:
[
  {{"id":1,"action_type":"click","target":"document area","value":null,"key":null,"description":"Click document to focus it","verification":"Cursor visible in document"}},
  {{"id":2,"action_type":"type","target":null,"value":"Hello World","key":null,"description":"Type text","verification":"Hello World visible in document"}},
  {{"id":3,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+s","description":"Save document","verification":"Title bar shows no unsaved indicator"}}
]"""


class PlanningAgent:
    def __init__(self, ovms_client: InferenceClient):
        self.ovms = ovms_client

    def plan(self, subtask: SubTask, context: dict = None,
             screen_context: str = None) -> List[ActionStep]:
        logger.info(f"[PLANNING] Planning: '{subtask.description}'")
        user_content = f"Generate steps for: {subtask.description}"
        if screen_context:
            user_content += (
                f"\n\nText currently visible on screen (use these exact labels as "
                f"click targets when relevant): {screen_context}"
            )
        if context:
            user_content += f"\nAdditional context: {json.dumps(context)}"
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3,
                                   response_schema=_STEP_SCHEMA)
        try:
            steps = self._parse_steps(resp.content, subtask.id)
        except (ValueError, json.JSONDecodeError) as first_err:
            logger.warning(f"[PLANNING] Parse failed ({first_err}) — retrying")
            retry_messages = [
                {"role": "system", "content": (
                    "Output ONLY a JSON array of desktop action steps.\n"
                    "RULES:\n"
                    "- click/right_click/double_click: 'target' must be the visible text or element name (not null)\n"
                    "- type: 'value' must be the text to type (not null)\n"
                    "- key_press/hotkey: 'key' must be the key name or combo (not null)\n"
                    "- scroll: 'target' is element, 'value' is 'up' or 'down'\n"
                    "- All other fields: null"
                )},
                {"role": "user", "content": f"Steps to accomplish: {subtask.description}"},
            ]
            resp = self.ovms.query_llm(retry_messages, max_tokens=2048, temperature=0.0,
                                       response_schema=_STEP_SCHEMA)
            steps = self._parse_steps(resp.content, subtask.id)
        logger.info(f"[PLANNING] {len(steps)} steps generated")
        for s in steps:
            logger.info(f"  [{s.id}] {s.action_type}: {s.description}")
        return steps

    def replan(
        self, failed_step: ActionStep, error: str, remaining: List[ActionStep]
    ) -> List[ActionStep]:
        logger.warning(f"[PLANNING] Replanning after step {failed_step.id} failure")
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Step {failed_step.id} ('{failed_step.description}') failed.\n"
                f"Error: {error}\n"
                f"Remaining planned steps: {[s.description for s in remaining]}\n\n"
                f"Generate a corrected sequence to recover and continue.",
            },
        ]
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3,
                                   response_schema=_STEP_SCHEMA)
        return self._parse_steps(resp.content, failed_step.subtask_id)

    def _parse_steps(self, text: str, subtask_id: int) -> List[ActionStep]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise ValueError(f"No JSON array in planning response: {text[:200]}")

        json_str = text[start_idx:end_idx + 1]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[PLANNING] JSON parse error: {e}\nRaw: {json_str[:300]}")
            raise

        steps = []
        for item in data:
            item["subtask_id"] = subtask_id
            item.setdefault("target", None)
            item.setdefault("value", None)
            item.setdefault("key", None)
            item.setdefault("description", "")
            item.setdefault("verification", "")
            step = ActionStep(**item)
            if step.action_type in ("hotkey", "key_press") and not step.key:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'key' is missing."
                )
            if step.action_type == "type" and step.value is None:
                raise ValueError(f"Step {step.id} is 'type' but 'value' is missing.")
            if step.action_type in ("click", "right_click", "double_click") and not step.target:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'target' is missing."
                )
            steps.append(step)
        return steps