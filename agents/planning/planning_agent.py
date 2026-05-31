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
_HOST = socket.gethostname().split(".")[0]
_SHELL_PROMPT = f"{_USER}@{_HOST}"


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

PLANNING_SYSTEM_PROMPT = f"""You are an expert desktop automation agent on {_OS_CONTEXT}.
Convert each sub-task instruction into the shortest correct JSON action sequence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTALLED APPS — only reference these
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Browser      : Firefox — ALWAYS launch via terminal command: {_FIREFOX_LAUNCH}
Code editor  : VS Code  (launch by searching "code" or "visual studio code")
Office suite : LibreOffice Writer ("libreoffice --writer"), Calc, Impress
Email        : Thunderbird
Terminal     : GNOME Terminal (hotkey: ctrl+alt+t  OR  search "terminal")
Text editors : nano, vim  (terminal only — NO gedit, NO mousepad, NO GUI text editor)
System apps  : GNOME Calculator, GNOME Settings, Screenshot
File manager : Nautilus/Files (for browsing only — never for creating/editing files)

NEVER generate steps for gedit, mousepad, notepad, kate, vlc, gimp — not installed.
For text editing tasks: use nano in terminal (simple) or LibreOffice Writer (documents).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART A — ACTION REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set ONLY the fields each action needs. Everything else → null.

  click / right_click / double_click  →  target = exact visible label (never null)
  type                                →  value  = exact string to type (never null)
  key_press  →  key = one key: "enter" "escape" "tab" "super" "backspace" "delete"
                    "f1"–"f12" "up" "down" "left" "right" "home" "end" "print_screen"
  hotkey     →  key = combo: "ctrl+s" "ctrl+c" "ctrl+v" "ctrl+z" "ctrl+a" "ctrl+l"
                    "ctrl+t" "ctrl+w" "ctrl+f" "ctrl+h" "ctrl+p" "ctrl+shift+s"
                    "ctrl+shift+t" "ctrl+alt+t" "alt+f4" "alt+tab" "alt+left"
                    "super+d" "super+l"
  scroll     →  target = area to scroll (never null), value = "up" or "down"
  wait       →  value  = seconds as string "0.5" "1.0" "2.0"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART B — UNIVERSAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. SCREEN CONTEXT IS TRUTH — if target is already visible, one click, nothing else.
   If target app is already open, skip its launch steps entirely.
2. STATE CONTEXT IN DESCRIPTION — if the sub-task description contains phrases like
   "with the terminal already open" or "with Firefox already open", that app is
   already running. Do NOT launch it. Do NOT open Activities. Go straight to the action:
   - "with the terminal already open" → click the shell prompt text e.g. {_SHELL_PROMPT} → wait 0.5s → type command → enter
   - "with Firefox already open"      → hotkey ctrl+l → type URL or query → enter
   - "with VS Code already open"      → use VS Code shortcuts directly
3. KEYBOARD FOCUS IS FRAGILE — any click moves focus. Before typing into any
   window: click that window first. Always. Even if it was open before.
4. MINIMUM STEPS — fewest steps possible. Shortcuts beat menu navigation.
5. PRECISE TARGETS — use exact visible text. Never "button", "icon", "link".
6. IGNORE UNRELATED WINDOWS — never click sidebars or panels in unrelated apps.
   Never click "Home" "Desktop" "Downloads" in Nautilus unless navigating files.
7. WAIT AFTER SLOW OPS — app launch: 1.0–1.5s. Snap apps (Firefox, Thunderbird): 2.0s.
   Page load: 1.5–2.0s. Install: 3–10s.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART C — LAUNCHING APPLICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only launch if NOT already visible in screen context.

Standard 4-step pattern:
  1. key_press  key="{_LAUNCHER_KEY}"          opens {_LAUNCHER_NAME}
  2. click      target="Type to search"        locks keyboard to search (MANDATORY)
  3. type       value="<app name>"
  4. key_press  key="enter"

Quick shortcuts (use instead of 4-step when available):
  Terminal → hotkey "ctrl+alt+t"
  After launch: wait 1.0–1.5s before interacting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART D — TERMINAL COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Desktop path: {_DESKTOP_PATH}

Opening terminal: hotkey "ctrl+alt+t" (fastest).

Pattern — terminal just launched:
  wait 2.0s → click the shell prompt text e.g. {_SHELL_PROMPT} → type command → key_press enter

Pattern — terminal already open from previous subtask:
  click the shell prompt text e.g. {_SHELL_PROMPT} → wait 0.5s → type command → key_press enter
  (NEVER re-launch. NEVER open Activities. Just click the shell prompt text and type.)

Common commands:
  Create file     : touch {_DESKTOP_PATH}/name.txt
  Write to file   : echo 'text' > {_DESKTOP_PATH}/name.txt
  Append to file  : echo 'text' >> {_DESKTOP_PATH}/name.txt
  Create folder   : mkdir {_DESKTOP_PATH}/foldername
  Delete file     : rm {_DESKTOP_PATH}/name.txt
  Rename/move     : mv {_DESKTOP_PATH}/old.txt {_DESKTOP_PATH}/new.txt
  Copy file       : cp source destination
  List files      : ls {_DESKTOP_PATH}
  Run python      : python3 {_DESKTOP_PATH}/script.py
  Install package : pip install packagename
  Git clone       : git clone https://github.com/user/repo
  Check version   : python3 --version

For nano editing:
  open terminal → type "nano {_DESKTOP_PATH}/filename.txt" → enter →
  wait 1.0s → click terminal → type content →
  hotkey "ctrl+o" → key_press enter (save) → hotkey "ctrl+x" (exit)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART E — FIREFOX BROWSER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Launch: ALWAYS use terminal to launch Firefox — open terminal → type "{_FIREFOX_LAUNCH}" → enter → wait 3.0s for Firefox to open.
IMPORTANT: If Activities search does not launch Firefox, use the terminal instead:
  open terminal → type "{_FIREFOX_LAUNCH}" → enter
Focus address bar : hotkey "ctrl+l"  (NEVER click address bar visually)
Navigate URL      : ctrl+l → type URL → enter
Web search        : ctrl+l → type query → enter
New tab           : hotkey "ctrl+t"
Close tab         : hotkey "ctrl+w"
Reload            : hotkey "ctrl+r"
Go back/forward   : hotkey "alt+left" / "alt+right"
Find on page      : ctrl+f → type → enter → escape
Scroll page       : scroll target="page content" value="down"
Zoom in/out/reset : ctrl+equal / ctrl+minus / ctrl+0
Fullscreen        : key_press "f11"
Downloads         : hotkey "ctrl+j"
Bookmark          : hotkey "ctrl+d" → click "Save"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART F — VS CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Launch: 4-step pattern with value="code", then wait 2.0s.
Open folder       : hotkey "ctrl+k" then "ctrl+o" → navigate → click "Open"
New file          : hotkey "ctrl+n"
Open file         : hotkey "ctrl+p" → type filename → enter
Save              : hotkey "ctrl+s"
New terminal      : hotkey "ctrl+grave"
Command palette   : hotkey "ctrl+shift+p" → type command → enter
Find in file      : hotkey "ctrl+f"
Find in project   : hotkey "ctrl+shift+f"
Comment line      : hotkey "ctrl+slash"
Close tab         : hotkey "ctrl+w"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART G — LIBREOFFICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Launch Writer : 4-step pattern value="libreoffice writer"
Launch Calc   : 4-step pattern value="libreoffice calc"
Launch Impress: 4-step pattern value="libreoffice impress"
Wait 2.0s after launch (LibreOffice is slow to start).

Writer:
  Click in document area first before typing.
  Disable autocorrect first: hotkey "ctrl+z" after typing if text changes unexpectedly.
  Save: ctrl+s  |  Save As: ctrl+shift+s → type name → enter
  Bold/Italic/Underline: ctrl+b / ctrl+i / ctrl+u
  Find & Replace: ctrl+h → type find → tab → type replace → click "Replace All"

Calc:
  Click cell → type value → enter (moves down) or tab (moves right)
  Formula: click cell → type "=SUM(A1:A5)" → enter
  Select range: click first cell → shift+click last

Impress:
  Click slide panel to select slide
  Next/prev slide: right/left arrow  |  Slideshow: f5  |  End: escape

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART H — SYSTEM OPERATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Screenshot     : hotkey "{_SCREENSHOT_KEY}"
Close window   : hotkey "{_CLOSE_WIN}"
Switch windows : hotkey "alt+tab"
Show desktop   : hotkey "super+d"
Lock screen    : hotkey "super+l"
Settings       : 4-step launch pattern value="settings"
Calculator     : 4-step launch pattern value="calculator"
Thunderbird    : 4-step launch pattern value="thunderbird", wait 2.0s
Volume         : click volume icon in system tray → drag slider
Brightness     : click brightness icon → drag slider
File manager   : 4-step launch pattern value="files"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART I — NEVER DO THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Never suggest gedit, mousepad, notepad, kate, vlc, gimp — not installed
❌ Never click browser address bar — always use hotkey ctrl+l
❌ Never skip "click Type to search" in launcher — text misses the bar
❌ Never re-launch an app already visible in screen context
❌ Never use vague targets like "button" "icon" "link" — use exact label text
❌ Never type without clicking/focusing the target window first
❌ Never click "Home" "Desktop" "Downloads" in Nautilus sidebar unless
   the task requires file manager navigation — steals keyboard focus silently
❌ Never type a terminal command without first clicking the shell prompt text e.g. {_SHELL_PROMPT}
❌ Never assume focus survived from a previous step
❌ Never add more steps than needed
❌ Never chain two type steps — combine into one

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART J — OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY a valid JSON array. No markdown. No explanation. No preamble.
All 7 fields on every object. Unused fields = null. IDs start at 1.

EXAMPLE A — run a terminal command (terminal already open):
[
  {{"id":1,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Re-focus terminal","verification":"Terminal is active with shell prompt"}},
  {{"id":2,"action_type":"wait","target":null,"value":"0.5","key":null,"description":"Wait for focus","verification":"Cursor blinking in terminal"}},
  {{"id":3,"action_type":"type","target":null,"value":"echo 'hello' > ~/Desktop/gui_agent","key":null,"description":"Type command","verification":"Command visible at prompt"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute command","verification":"New prompt appears, no error"}}
]

EXAMPLE B — open Firefox and navigate:
[
  {{"id":1,"action_type":"key_press","target":null,"value":null,"key":"super","description":"Open Activities","verification":"Activities overlay visible"}},
  {{"id":2,"action_type":"click","target":"Type to search","value":null,"key":null,"description":"Focus search bar","verification":"Cursor in search bar"}},
  {{"id":3,"action_type":"type","target":null,"value":"{_FIREFOX_LAUNCH}","key":null,"description":"Run Firefox launch command","verification":"Firefox window starts opening"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Launch Firefox","verification":"Firefox window opens"}},
  {{"id":5,"action_type":"wait","target":null,"value":"2.0","key":null,"description":"Wait for snap app to load","verification":"Firefox address bar visible"}},
  {{"id":6,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+l","description":"Focus address bar","verification":"Address bar highlighted"}},
  {{"id":7,"action_type":"type","target":null,"value":"https://github.com","key":null,"description":"Type URL","verification":"URL in address bar"}},
  {{"id":8,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Navigate","verification":"GitHub page loads"}}
]

EXAMPLE B2 — navigate in Firefox that is ALREADY OPEN (description says "with Firefox already open"):
[
  {{"id":1,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+l","description":"Focus Firefox address bar directly — Firefox is already open","verification":"Address bar highlighted and ready"}},
  {{"id":2,"action_type":"type","target":null,"value":"https://github.com/Shehrozkashif","key":null,"description":"Type URL","verification":"URL visible in address bar"}},
  {{"id":3,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Navigate to page","verification":"GitHub profile page loads"}}
]

EXAMPLE C — open terminal fresh and run command:
[
  {{"id":1,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+alt+t","description":"Open terminal","verification":"Terminal window appears"}},
  {{"id":2,"action_type":"wait","target":null,"value":"3.0","key":null,"description":"Wait for shell prompt","verification":"Shell prompt visible"}},
  {{"id":3,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Focus terminal","verification":"Terminal is active window"}},
  {{"id":4,"action_type":"type","target":null,"value":"touch ~/Desktop/notes.txt","key":null,"description":"Type command","verification":"Command at prompt"}},
  {{"id":5,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute","verification":"New prompt, no error"}}
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