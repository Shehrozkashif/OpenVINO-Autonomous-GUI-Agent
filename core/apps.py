# core/apps.py
"""What the agent knows about desktop applications.

Two curated tables plus a fallback:
  PROCESS_MAP  — description keyword → Windows executable. A process check is
                 immune to the OCR false positives that plagued launch
                 verification (the agent's own window text says "Notepad" too).
  APP_SIGNALS  — description keyword → words that appear on screen once the app
                 is up. Used only for apps with no known executable.
  launch_signals() — derives signal words straight from "open <AppName>" so
                 launch verification also works for apps in neither table.
"""
import re

# Description keyword → Windows process executable name.
PROCESS_MAP: dict[str, str] = {
    "windows terminal": "WindowsTerminal.exe",
    "terminal":         "WindowsTerminal.exe",
    "command prompt":   "cmd.exe",
    "powershell":       "powershell.exe",
    "notepad":          "notepad.exe",
    "calculator":       "CalculatorApp.exe",
    "paint":            "mspaint.exe",
    "file explorer":    "explorer.exe",
    "explorer":         "explorer.exe",
    "edge":             "msedge.exe",
    "firefox":          "firefox.exe",
    "chrome":           "chrome.exe",
    "brave":            "brave.exe",
    "vs code":          "Code.exe",
    "visual studio":    "devenv.exe",
    "task manager":     "Taskmgr.exe",
    "snipping tool":    "SnippingTool.exe",
    "wordpad":          "wordpad.exe",
    "settings":         "SystemSettings.exe",
    "zoom":             "Zoom.exe",
    "teams":            "ms-teams.exe",
    "slack":            "slack.exe",
    "skype":            "Skype.exe",
    "thunderbird":      "thunderbird.exe",
}

# Description keyword → OCR words that prove the app is on screen.
APP_SIGNALS: dict[str, list[str]] = {
    "calculator":      ["Calculator", "Standard", "Scientific"],
    "notepad":         ["Notepad", "Untitled", "File"],
    "paint":           ["Paint", "Home", "Image"],
    "command prompt":  ["cmd", "C:\\", "Microsoft"],
    "powershell":      ["PowerShell", "PS", "Windows"],
    "windows terminal": ["Windows Terminal", "Terminal", "PowerShell"],
    "terminal":        ["Terminal", "PowerShell", "cmd"],
    "file explorer":   ["File Explorer", "This PC", "Documents", "Quick access"],
    "explorer":        ["File Explorer", "This PC", "Documents", "Quick access"],
    "edge":            ["Microsoft Edge", "New Tab", "Search"],
    "firefox":         ["Firefox", "Mozilla", "Search"],
    "chrome":          ["Google Chrome", "New Tab", "Search"],
    "vs code":         ["Explorer", "Extensions", "Welcome", "Visual Studio"],
    "visual studio":   ["Explorer", "Extensions", "Welcome", "Visual Studio"],
    "libreoffice":     ["Writer", "Calc", "Impress", "LibreOffice"],
    "thunderbird":     ["Thunderbird", "Inbox", "Compose"],
    "settings":        ["Settings", "System", "Bluetooth", "Windows Update"],
    "task manager":    ["Task Manager", "Processes", "CPU"],
    "snipping tool":   ["Snipping Tool", "New", "Mode"],
    "wordpad":         ["WordPad", "Home", "Document"],
    "zoom":            ["Zoom", "New Meeting", "Schedule", "Join"],
    "teams":           ["Teams", "Chat", "Calendar", "Meet"],
    "slack":           ["Slack", "Channels", "Direct messages"],
}

# Processes that are effectively always running, so "is it running" proves
# nothing about a launch — only a NEW visible window does.
ALWAYS_RUNNING = frozenset({
    "explorer.exe",     # the Windows shell itself
})

# Words that carry no signal on their own when derived from an app name.
_GENERIC_NAME_WORDS = frozenset((
    "the", "a", "an", "app", "application", "browser", "program", "tool",
))

# Terminals keep the stricter "a NEW window is required" launch rule: reusing a
# pre-existing console can hand the agent a window that is busy running
# something else.
TERMINAL_LAUNCH_KEYS = (
    "windows terminal", "terminal", "command prompt", "powershell",
)


def process_for(description: str) -> str | None:
    """The executable a launch description refers to, or None if unknown."""
    desc = (description or "").lower()
    return next((exe for key, exe in PROCESS_MAP.items() if key in desc), None)


def signals_for(description: str) -> list[str]:
    """OCR words proving the described app is up: curated table first, then
    words derived from the app's own name.
    """
    desc = (description or "").lower()
    key = next((k for k in APP_SIGNALS if k in desc), None)
    return APP_SIGNALS[key] if key else launch_signals(description)


def launch_signals(description: str) -> list[str]:
    """Derive OCR signal words from "open <AppName>" — "open Brave Browser" →
    ["Brave Browser", "Brave"].

    Keeps launch verification working for ANY installed app, not just the
    curated ones. Original casing is preserved because OCR text is
    case-sensitive.
    """
    m = re.search(
        r"\b(?:open|launch|start)\s+(?:the |a |an )?"
        r"([A-Za-z0-9][\w+-]*(?:[ \t]+[A-Za-z0-9][\w+-]*)*?)"
        r"(?:[ \t]+(?:using|via|by|with|and)\b|[.,!?]|$)",
        description or "",
    )
    if not m:
        return []
    name = m.group(1).strip().rstrip(".")
    words = [w for w in name.split()
             if len(w) >= 3 and w.lower() not in _GENERIC_NAME_WORDS]
    return [name] + words[:2] if words else []


def is_launch_description(description: str) -> bool:
    """True when the description is a genuine app launch, not an in-app action.

    A subtask that ACTS inside an already-open app reads "with the Calendar
    view open in Teams, click New meeting" — it contains the word "open" and an
    app name yet launches nothing. Classifying it as a launch made the agent
    demand a NEW Teams window and fail a click that had already succeeded, so
    the leading verb of the BODY decides, not any word in the sentence.
    """
    desc = (description or "").lower()
    if "already open" in desc or "already running" in desc:
        return False
    body = re.sub(r"^\s*with\b[^,]*,\s*", "", desc).strip()
    if re.match(
        r"(then\s+)?(click|set|select|type|enter|fill|add|choose|save|"
        r"schedule|create|send|pick|toggle|check|dismiss|close|scroll)\b",
        body,
    ):
        return False
    return body.startswith("launch") or (
        body.startswith("open") and any(k in body for k in PROCESS_MAP)
    )
