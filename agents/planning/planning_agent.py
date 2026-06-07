# agents/planning/planning_agent.py
"""
Planning Agent — generates precise step sequences for sub-tasks.
Uses Chain-of-Thought prompting for complex multi-step tasks.
"""

import json
import os
import platform
import re
from typing import List, Optional

from loguru import logger
from utils.platform_utils import detect_firefox

from core.protocols.a2a import ActionStep, InferenceClient, SubTask

# Imported inside functions to avoid a hard circular-import at module load time.
# TYPE_CHECKING guard is sufficient for type hints only; we need the class at
# runtime for isinstance checks, so the deferred import pattern is used instead.
_ScreenSnapshot = None  # resolved on first use


def _get_snapshot_class():
    global _ScreenSnapshot
    if _ScreenSnapshot is None:
        from core.capture.screen_snapshot import ScreenSnapshot  # noqa: PLC0415
        _ScreenSnapshot = ScreenSnapshot
    return _ScreenSnapshot

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
    _ICON_NOTE = (
        "Windows exposes every icon's accessible name through UI Automation, so "
        "clicking a labelled desktop, taskbar, or Start-menu icon by its visible "
        "text is exact and instant (faster and more reliable than the launcher)."
    )
elif _OS == "Darwin":
    _OS_CONTEXT     = "macOS desktop (Ventura/Sonoma)"
    _LAUNCHER_KEY   = "command+space"
    _LAUNCHER_NAME  = "Spotlight search"
    _CLOSE_WIN      = "command+w"
    _NEW_TAB        = "command+t"
    _REOPEN_TAB     = "command+shift+t"
    _DESKTOP_PATH   = "~/Desktop"
    _SCREENSHOT_KEY = "command+shift+3"
    _ICON_NOTE = (
        "Dock and Launchpad icons are clearly labelled — click them directly by "
        "their visible name when present; it's faster than opening Spotlight."
    )
else:
    _OS_CONTEXT     = "Linux desktop (GNOME)"
    _LAUNCHER_KEY   = "super"
    _LAUNCHER_NAME  = "GNOME Activities overview"
    _CLOSE_WIN      = "alt+f4"
    _NEW_TAB        = "ctrl+t"
    _REOPEN_TAB     = "ctrl+shift+t"
    _DESKTOP_PATH   = "~/Desktop"
    _SCREENSHOT_KEY = "print_screen"
    _ICON_NOTE = (
        "Desktop and taskbar icons are matched by their visible label text — "
        "click them directly by name when present; it's faster than Activities."
    )

# ── Runtime machine identity ──────────────────────────────────────────────────
_USER = os.getenv("USER") or os.getenv("USERNAME") or "user"
# Use only the username — hostnames can be very long (e.g. laptop model names)
# and OCR reliably finds the short username portion of the shell prompt.
_SHELL_PROMPT = _USER


_FIREFOX_CMD = detect_firefox()
# On Linux/macOS run Firefox in the background so the terminal stays usable
_FIREFOX_LAUNCH = _FIREFOX_CMD if _OS == "Windows" else f"{_FIREFOX_CMD} &"

# Pre-compute OS-specific values used inside f-string examples
# (f-strings can't contain backslashes in expression parts before Python 3.12)
_TERM_APP   = "cmd" if _OS == "Windows" else "gnome-terminal"
_ECHO_CMD   = (f"echo hello > {_DESKTOP_PATH}\\hello.txt"
               if _OS == "Windows"
               else f"echo 'hello world' > {_DESKTOP_PATH}/hello.txt")
_MKDIR_CMD  = (f"mkdir {_DESKTOP_PATH}\\projects"
               if _OS == "Windows"
               else f"mkdir {_DESKTOP_PATH}/projects")

# Launcher instructions differ between Windows (no click-to-focus needed) and Linux/macOS
_LAUNCHER_NOTE = (
    "Do NOT open apps by navigating the Start menu manually — use search instead."
    if _OS == "Windows"
    else "Do NOT use ctrl+alt+t for terminal — the shortcut may not be configured."
)
_LAUNCHER_PATTERN_LABEL = (
    "Search launcher pattern for Windows (Win key focuses search immediately — no click needed):"
    if _OS == "Windows"
    else "Search launcher pattern (use for every app including terminal):"
)
_LAUNCHER_PATTERN_STEPS = (
    '    key_press "winleft"  →  wait "0.5"  →  type "<app name>"  →  key_press "enter"'
    if _OS == "Windows"
    else f'    key_press "{_LAUNCHER_KEY}"  →  click "Type to search"  →  type "<app name>"  →  key_press "enter"'
)

# Terminal section — pre-computed per OS so no backslashes appear inside f-string {}
_SEP = "\\" if _OS == "Windows" else "/"
_TERMINAL_FRESH_LAUNCH = (
    f'  key_press "winleft" → wait "0.5" → type "Windows Terminal" → key_press enter → wait "2.0"'
    f'\n  STOP here if goal is just "open terminal". Terminal is open when PS prompt is visible.'
    f'\n  If goal also includes running a command:'
    f'\n    click {_SHELL_PROMPT} → type command → key_press enter'
    if _OS == "Windows"
    else (f'  key_press "{_LAUNCHER_KEY}" → click "Type to search" → type "gnome-terminal"'
          f' → key_press enter → wait "2.0"'
          f'\n  STOP here if goal is just "open terminal". Terminal is open when prompt ($) is visible.'
          f'\n  If goal also includes running a command:'
          f'\n    click {_SHELL_PROMPT} → type command → key_press enter')
)

if _OS == "Windows":
    _D = _DESKTOP_PATH   # already uses backslash from the assignment above
    _TERMINAL_COMMANDS = (
        f"  Create file  :  type nul > {_D}\\name.txt\n"
        f"  Write file   :  echo text > {_D}\\name.txt\n"
        f"  Append file  :  echo text >> {_D}\\name.txt\n"
        f"  Make folder  :  mkdir {_D}\\foldername\n"
        f"  Delete file  :  del {_D}\\name.txt\n"
        f"  Move/rename  :  move {_D}\\old {_D}\\new\n"
        f"  Copy file    :  copy source destination\n"
        f"  Run Python   :  python {_D}\\script.py\n"
        f"  Install pkg  :  pip install packagename\n"
        f"  List files   :  dir {_D}\n"
        f"  Git clone    :  git clone https://github.com/user/repo"
    )
else:
    _D = _DESKTOP_PATH
    _TERMINAL_COMMANDS = (
        f"  Create file  :  touch {_D}/name.txt\n"
        f"  Write file   :  echo 'text' > {_D}/name.txt\n"
        f"  Append file  :  echo 'text' >> {_D}/name.txt\n"
        f"  Make folder  :  mkdir {_D}/foldername\n"
        f"  Delete file  :  rm {_D}/name.txt\n"
        f"  Move/rename  :  mv {_D}/old {_D}/new\n"
        f"  Copy file    :  cp source destination\n"
        f"  Run Python   :  python3 {_D}/script.py\n"
        f"  Install pkg  :  pip install packagename\n"
        f"  List files   :  ls {_D}\n"
        f"  Git clone    :  git clone https://github.com/user/repo"
    )

_STEP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":           {"type": "integer"},
            "action_type":  {"type": "string", "enum": [
                "click", "right_click", "double_click",
                "type", "key_press", "hotkey", "scroll", "wait", "drag", "extract",
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
1. READ screen context. If the EXACT app name or element label appears in the
   visible screen text as an interactive element (desktop icon, taskbar button,
   pinned tile) → click it by that label. ONE step. Stop.
   CRITICAL: NEVER click an app icon using your training-data knowledge of where
   it "should" be. Only click it if its name appears in the screen context text.
   If the app name is NOT in the visible text → skip to step 3.
2. USE a keyboard shortcut if one exists (ctrl+l for browser bar, etc.).
3. OPEN the search launcher when the app is not visible in the screen text.
   This is the correct default for launching any app not shown in screen context.

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
scroll                              →  target = element to scroll over (null = scroll page center)
                                       value  = "up" or "down" (default "down")
drag                                →  target = source element text label (what to drag FROM)
                                       value  = destination element text label (where to drag TO)
                                       Use for: drag-and-drop files, reorder list items, resize panes
extract                             →  target = description of what to read from screen
                                       e.g. "the error message", "the page title", "the file path"
                                       Use when the task says: "tell me", "what is", "read", "get the value"
                                       The extracted text is returned to the user at task end.
wait                                →  value  = seconds as string: "0.5" "1.0" "2.0" "3.0"

━━━ LAUNCHING APPS ━━━
PRIMARY — click a visible icon (ONLY when the exact app name appears in screen text):
  The app name must appear in the visible screen context text as an interactive
  element. If it is NOT there → skip this and use the search launcher below.
  {_ICON_NOTE}
  FORBIDDEN: Do NOT guess that "Calculator", "Notepad", or any app is in the
  taskbar from training knowledge. If you cannot see the name in the screen text,
  treat it as absent and use the search launcher.
  CRITICAL: Company names ("Microsoft", "Google", "Apple") in copyright text,
  window headers, or log output are NOT app icons. Only the EXACT short app name
  (e.g. "Notepad", "Firefox") visible as a taskbar/desktop element counts.

FALLBACK → PREFERRED — search launcher (use whenever the app is not in screen text):
  {_LAUNCHER_NOTE}

  {_LAUNCHER_PATTERN_LABEL}
  {_LAUNCHER_PATTERN_STEPS}
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
{_TERMINAL_FRESH_LAUNCH}

Terminal already open (from previous sub-task):
  click {_SHELL_PROMPT} → wait "0.5" → type command → key_press enter

Common commands ({_OS_CONTEXT}):
{_TERMINAL_COMMANDS}

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

━━━ LOGIN / CREDENTIALS ━━━
When a task requires entering a username or password, use credential tokens:
  username field  →  type  value="{{cred:site:username}}"
  password field  →  type  value="{{cred:site:password}}"
  Replace "site" with the actual site or app (github.com, gmail.com, localhost, etc.)
  The credentials are substituted from the user's stored credential file at runtime.

Typical login flow:
  1. click  target="username field visible text"  (or use Tab to focus it)
  2. type   value="{{cred:site:username}}"
  3. key_press  key="tab"                          (move to password field)
  4. type   value="{{cred:site:password}}"
  5. key_press  key="enter"                        (submit form)

━━━ DEEP NAVIGATION / LONG PAGES ━━━
To find content that may be below the fold (not visible on screen):
  1. scroll  target=null  value="down"   (the system auto-scrolls and retries grounding)
  2. Continue generating steps — grounding will retry after each scroll automatically.
  Do NOT generate a long chain of scroll steps manually — one step is enough.

━━━ POPUP / DIALOG HANDLING ━━━
If the screen shows an unexpected dialog, dismiss it BEFORE continuing:
  Error / alert dialog      →  key_press "escape" or click "OK" / "Close"
  "Save before closing?"    →  click "Don't Save" (or "Discard") to proceed, or "Save" to preserve
  "Replace file?"           →  click "Replace" to overwrite
  "Allow / Deny" permission →  click "Allow"
  Any unrelated notification popup → key_press "escape"
Always handle visible dialogs first — they block all other actions.

━━━ TEXT SELECTION ━━━
Select all text in a field  →  hotkey ctrl+a
Select word under cursor     →  double_click on the word
Select line in terminal      →  hotkey ctrl+a (bash) or triple-click
Clear a text field           →  hotkey ctrl+a  → key_press "delete"
Copy selected text           →  hotkey ctrl+c
Paste                        →  hotkey ctrl+v
Cut                          →  hotkey ctrl+x

━━━ FILE DIALOGS (GTK "Open" / "Save As") ━━━
GTK file dialogs have a hidden path bar. Fastest pattern — type the full path directly:
  1. hotkey ctrl+l          (reveals the path-entry bar, works in Nautilus and GTK dialogs)
  2. hotkey ctrl+a          (select any existing text in the bar)
  3. type   value="<full path or filename>"
  4. key_press "enter"

Examples:
  Open a specific file  :  ctrl+l → ctrl+a → type "/home/user/Documents/file.txt" → enter
  Navigate to folder    :  ctrl+l → ctrl+a → type "/home/user/Downloads" → enter
  Save with new name    :  ctrl+l (if available) → ctrl+a → type "report_v2.pdf" → enter
                           OR click the filename field directly → ctrl+a → type name → enter

Tab-based form navigation (when multiple fields exist):
  key_press "tab" moves forward between fields; "shift+tab" moves backward.
  Use tab to move from one field to the next instead of clicking each field.

━━━ WINDOWS SAVE / SAVE-AS DIALOG ━━━
How to detect: screen text contains "File name" AND "Save" AND "Cancel".
When this pattern is visible, a Save-As dialog is open. Your ONLY valid actions are:
  a) If a specific filename is required:
       hotkey ctrl+a              (select all text in the filename field)
       type  value="hello.txt"    (type the new filename — this REPLACES the old name)
       key_press "enter"          (confirm — this is the step IMMEDIATELY after type)
  b) If no specific name is required: key_press "enter"  (save with current name)
  c) "Replace existing file?" prompt → key_press "enter"  (confirm overwrite)
CRITICAL: After typing the filename (step a), your very next step MUST be key_press "enter".
          Do NOT type the filename again — it is already in the field.
  ✗ NEVER press Ctrl+S when "File name" and "Cancel" are visible — it has no effect.
     Ctrl+S opens the dialog from the editor. Once the dialog is open, use Enter only.

━━━ STRICT RULES ━━━
✓ Exact visible text as click target — never "button", "icon", "link"
✓ Click a window before typing in it (except fresh terminal)
✓ Combine all related text into ONE type step — never chain two type steps
✓ ctrl+l to focus browser address bar — never click the bar visually
✓ Handle any dialog/popup you see before doing the next planned step
✓ When the goal says "right click" or "right-click", ALWAYS output action_type: "right_click" — NEVER output "click" for a right-click action
✗ Never use gedit, mousepad, kate, VLC, GIMP — use nano or LibreOffice
✗ Never open Activities/search when a visible icon or hotkey works
✗ Never add steps just to be safe — minimum steps only
✗ Never type in terminal without first clicking the shell prompt (if terminal was already open)
✗ Never re-launch an app that the sub-task description says is already open
✗ Never use ctrl+alt+del — Windows intercepts it; it cannot be sent by automation
✗ Never click an app icon whose name is not in the visible screen text
✗ Never press Ctrl+S when a Save or Save-As dialog is already visible — use Enter instead

━━━ OUTPUT ━━━
Valid JSON array only. All 7 fields required. Unused fields = null. IDs start at 1.

━━━ EXAMPLES ━━━

EXAMPLE 1 — app icon visible in screen context (screen shows "Code"):
[
  {{"id":1,"action_type":"click","target":"Code","value":null,"key":null,"description":"Click VS Code icon visible in taskbar","verification":"VS Code window opens or comes to front"}},
  {{"id":2,"action_type":"wait","target":null,"value":"1.0","key":null,"description":"Wait for VS Code to load","verification":"VS Code editor is visible"}}
]

EXAMPLE 2 — open terminal and run a command (no terminal icon visible — use search launcher):
[
  {{"id":1,"action_type":"key_press","target":null,"value":null,"key":"{_LAUNCHER_KEY}","description":"Open {_LAUNCHER_NAME}","verification":"Search box appears"}},
  {{"id":2,"action_type":"wait","target":null,"value":"0.5","key":null,"description":"Wait for search to open","verification":"Search ready"}},
  {{"id":3,"action_type":"type","target":null,"value":"{_TERM_APP}","key":null,"description":"Type terminal app name","verification":"Terminal result visible"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Launch terminal","verification":"Terminal window opens"}},
  {{"id":5,"action_type":"wait","target":null,"value":"2.0","key":null,"description":"Wait for prompt","verification":"Shell prompt visible"}},
  {{"id":6,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Click prompt to confirm focus","verification":"Terminal is active"}},
  {{"id":7,"action_type":"type","target":null,"value":"{_ECHO_CMD}","key":null,"description":"Type command","verification":"Command visible at prompt"}},
  {{"id":8,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute command","verification":"New prompt appears, no error"}}
]

EXAMPLE 3 — terminal already open from previous sub-task:
[
  {{"id":1,"action_type":"click","target":"{_SHELL_PROMPT}","value":null,"key":null,"description":"Click prompt to focus terminal","verification":"Terminal is active"}},
  {{"id":2,"action_type":"wait","target":null,"value":"0.5","key":null,"description":"Wait for focus","verification":"Cursor blinking"}},
  {{"id":3,"action_type":"type","target":null,"value":"{_MKDIR_CMD}","key":null,"description":"Type command","verification":"Command at prompt"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Execute","verification":"New prompt, no error"}}
]

EXAMPLE 4 — browser already open, navigate to URL:
[
  {{"id":1,"action_type":"hotkey","target":null,"value":null,"key":"ctrl+l","description":"Focus address bar","verification":"Address bar highlighted"}},
  {{"id":2,"action_type":"type","target":null,"value":"https://github.com","key":null,"description":"Type URL","verification":"URL visible in address bar"}},
  {{"id":3,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Navigate","verification":"GitHub page loads"}}
]

EXAMPLE 5 — launch app with search (no icon visible, no shortcut):
[
  {{"id":1,"action_type":"key_press","target":null,"value":null,"key":"{_LAUNCHER_KEY}","description":"Open {_LAUNCHER_NAME}","verification":"Search box visible"}},
  {{"id":2,"action_type":"wait","target":null,"value":"0.5","key":null,"description":"Wait for search to open","verification":"Search ready"}},
  {{"id":3,"action_type":"type","target":null,"value":"notepad","key":null,"description":"Type app name","verification":"App result visible"}},
  {{"id":4,"action_type":"key_press","target":null,"value":null,"key":"enter","description":"Launch","verification":"App opens"}},
  {{"id":5,"action_type":"wait","target":null,"value":"2.0","key":null,"description":"Wait for app to load","verification":"App window visible"}}
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

    def plan_next_step(
        self,
        subtask: SubTask,
        screen_context: str = None,
        completed: List[str] = None,
        task_context: List[str] = None,
        failure_hints: List[str] = None,
        snapshot=None,   # Optional[ScreenSnapshot] — when provided overrides screen_context
    ) -> Optional[ActionStep]:
        """
        Dynamic planning: return the ONE next action step toward the subtask goal.
        Returns None when the goal is already achieved (planner returns empty array).

        task_context:   descriptions of subtasks already completed in this overall task.
        failure_hints:  known-bad target/action patterns from episodic failure memory.
        snapshot:       ScreenSnapshot from capture_snapshot(); when provided its
                        format_for_planner() output replaces the raw screen_context string.
        """
        # Use structured snapshot context when available (Fix 1.2)
        SnapshotClass = _get_snapshot_class()
        if snapshot is not None and isinstance(snapshot, SnapshotClass):
            screen_context = snapshot.format_for_planner()
        # Inter-subtask context — what was done before this subtask
        ctx_block = ""
        if task_context:
            ctx_lines = "\n".join(f"  - {d}" for d in task_context)
            ctx_block = f"\nSubtasks already completed in this task:\n{ctx_lines}\n"

        # Within-subtask step history
        history = ""
        if completed:
            lines = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(completed))
            history = f"\nSteps completed so far toward this goal:\n{lines}\n"

        # Episodic failure hints — avoid patterns that failed before
        failure_block = ""
        if failure_hints:
            hint_lines = "\n".join(f"  - {h}" for h in failure_hints)
            failure_block = f"\nKnown failure patterns to avoid:\n{hint_lines}\n"

        user_content = f"Goal: {subtask.description}{ctx_block}{history}{failure_block}"

        # Reinforce the right_click rule in the user prompt when relevant
        _sd_lower = subtask.description.lower()
        if "right click" in _sd_lower or "right-click" in _sd_lower:
            user_content += (
                '\nRULE: The goal contains "right click" — you MUST output '
                'action_type: "right_click", NEVER "click".\n'
            )

        if screen_context:
            user_content += f"\nText currently visible on screen: {screen_context}"

        user_content += (
            "\n\nReturn the NEXT SINGLE action step needed to achieve the goal.\n"
            "Return [] ONLY when the goal is DEFINITIVELY complete:\n"
            "  • ‘open terminal / Windows Terminal’: a shell prompt is visible "
            "(text like ‘PS’, ‘C:\\>’, ‘$’, or a username prompt) = goal achieved.\n"
            "  • ‘run: <command>’ in terminal: The command has been TYPED (in history) AND "
            "Enter has been pressed (in history). If a shell prompt is now visible = goal achieved. "
            "Do NOT press Enter again.\n"
            "  • ‘open browser’: the URL/address bar or browser tab is visible = achieved.\n"
            "  • ‘open <any app>’: the app’s actual running content is on screen "
            "(document area, file list, settings panel, etc.) = achieved.\n"
            "  • ‘click <menu item>’: if a submenu panel or dialog opened AFTER the click = "
            "goal achieved. Do NOT re-click the same item just because it is still visible "
            "in the parent menu — parent menu items remain visible after their submenu opens.\n"
            "  • ‘type X and press enter’ (rename/create dialog): step 1 = type X; "
            "step 2 = key_press enter; step 3 = [] (done). "
            "Do NOT type X again after it already appears in history — go straight to enter.\n"
            "CAUTION: An app name appearing in text does NOT mean the app is open — "
            "it may be from the task description shown in the GUI agent’s own log window, "
            "or a search result. Only return [] when the app’s active running content "
            "is clearly visible.\n"
            "LOOP PREVENTION: If the step you are about to plan has the same action_type, "
            "target, value, and key as the immediately preceding completed step, do NOT "
            "repeat it — plan the next logical action in sequence or return [].\n"
            "When in doubt whether the goal is complete, return [] rather than adding "
            "speculative steps."
        )

        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        resp = self.ovms.query_llm(
            messages, max_tokens=768, temperature=0.2,
            response_schema=_STEP_SCHEMA,
        )
        try:
            steps = self._parse_steps(resp.content, subtask.id)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[PLANNING] plan_next_step parse error: {e} — skipping")
            return None

        if not steps:
            logger.info(f"[PLANNING] Goal achieved — no more steps needed")
            return None

        step = steps[0]

        # Deterministic right_click override — the LLM occasionally outputs "click"
        # even when the subtask description clearly says "right click".
        _sub_lower = subtask.description.lower()
        if step.action_type == "click" and (
            _sub_lower.startswith("right click")
            or "right click on" in _sub_lower
            or _sub_lower.startswith("right-click")
            or "right-click on" in _sub_lower
        ):
            step.action_type = "right_click"
            logger.info("[PLANNING] Overrode action_type to right_click (subtask says 'right click')")

        step.id = len(completed or []) + 1
        logger.info(f"[PLANNING] Next: [{step.action_type}] {step.description}")
        return step

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