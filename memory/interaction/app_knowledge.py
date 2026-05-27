# memory/interaction/app_knowledge.py
"""
Known shortcuts and layout hints for common applications.
Grounding Agent checks this first — avoids VLM calls for known patterns.
"""
from typing import Optional

APP_KNOWLEDGE = {
    "vscode": {
        "file menu": {"shortcut": None, "region": "top-left", "hint": "first menu item in top menu bar"},
        "settings": {"shortcut": "ctrl+,"},
        "terminal": {"shortcut": "ctrl+`"},
        "save": {"shortcut": "ctrl+s"},
        "save as": {"shortcut": "ctrl+shift+s"},
        "explorer sidebar": {"region": "left", "hint": "folder icon in left activity bar"},
        "search": {"shortcut": "ctrl+shift+f"},
        "command palette": {"shortcut": "ctrl+shift+p"},
    },
    "chrome": {
        "address bar": {"region": "top-center", "hint": "URL input box"},
        "new tab": {"shortcut": "ctrl+t"},
        "refresh": {"key": "f5"},
        "settings menu": {"region": "top-right", "hint": "three dots icon"},
        "close tab": {"shortcut": "ctrl+w"},
    },
    "notepad": {
        "file menu": {"region": "top-left"},
        "save": {"shortcut": "ctrl+s"},
        "save as": {"shortcut": "ctrl+shift+s"},
    },
    "file_explorer": {
        "address bar": {"region": "top"},
        "search box": {"region": "top-right"},
        "new folder": {"shortcut": "ctrl+shift+n"},
    }
}


def get_shortcut(app: str, action: str) -> Optional[str]:
    """Return keyboard shortcut if known, else None (fall back to VLM grounding)."""
    app_data = APP_KNOWLEDGE.get(app.lower(), {})
    return app_data.get(action.lower(), {}).get("shortcut")


def get_hint(app: str, element: str) -> Optional[str]:
    """Return region/hint if known, to narrow VLM grounding search."""
    app_data = APP_KNOWLEDGE.get(app.lower(), {})
    return app_data.get(element.lower(), {}).get("hint")
