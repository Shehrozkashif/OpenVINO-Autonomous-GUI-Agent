# utils/clipboard.py
"""
Cross-platform clipboard read/write.

On Linux: tries xclip then xsel (no Python deps needed).
On Windows/macOS: uses pyperclip.
"""
import platform
import subprocess
import threading
import time
from typing import Optional

_OS = platform.system()

# ── Availability check (done once at import) ──────────────────────────────────

def _probe_linux_clipboard() -> Optional[str]:
    """Return the first working clipboard tool name, or None."""
    for tool, args_write, args_read in [
        ("xclip",
         ["xclip", "-selection", "clipboard"],
         ["xclip", "-selection", "clipboard", "-out"]),
        ("xsel",
         ["xsel", "--clipboard", "--input"],
         ["xsel", "--clipboard", "--output"]),
    ]:
        try:
            p = subprocess.run(args_write, input=b"_probe_",
                               capture_output=True, timeout=1)
            if p.returncode == 0:
                return tool
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


_LINUX_TOOL: Optional[str] = _probe_linux_clipboard() if _OS == "Linux" else None


# ── Write ─────────────────────────────────────────────────────────────────────

def write(text: str) -> bool:
    """Set clipboard text. Returns True on success."""
    if _OS == "Linux":
        if _LINUX_TOOL == "xclip":
            cmd = ["xclip", "-selection", "clipboard"]
        elif _LINUX_TOOL == "xsel":
            cmd = ["xsel", "--clipboard", "--input"]
        else:
            return False
        try:
            p = subprocess.run(cmd, input=text.encode(), capture_output=True, timeout=2)
            return p.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False
    else:
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except Exception:
            return False


# ── Read ──────────────────────────────────────────────────────────────────────

def read() -> str:
    """Get current clipboard text."""
    if _OS == "Linux":
        if _LINUX_TOOL == "xclip":
            cmd = ["xclip", "-selection", "clipboard", "-out"]
        elif _LINUX_TOOL == "xsel":
            cmd = ["xsel", "--clipboard", "--output"]
        else:
            return ""
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=2)
            return r.stdout.decode(errors="replace") if r.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            return ""
    else:
        try:
            import pyperclip
            return pyperclip.paste() or ""
        except Exception:
            return ""


# ── Available check ───────────────────────────────────────────────────────────

def available() -> bool:
    """Return True if clipboard operations are available on this system."""
    if _OS == "Linux":
        return _LINUX_TOOL is not None
    try:
        import pyperclip  # noqa: F401
        return True
    except ImportError:
        return False


# ── Clipboard-paste typing ────────────────────────────────────────────────────

def paste_type(text: str, send_hotkey_fn, sensitive: bool = False) -> bool:
    """
    Type text by setting the clipboard and sending ctrl+v.
    Restores previous clipboard content after 600 ms (async).

    send_hotkey_fn: callable that accepts *key_names (e.g. controller._send_hotkey)
    sensitive:      when True, the clipboard is cleared synchronously right after
                    the paste so a password/secret never lingers, even briefly,
                    in addition to the normal async restore.
    """
    if not available():
        return False
    old = read()
    if not write(text):
        return False
    time.sleep(0.08)
    send_hotkey_fn("ctrl", "v")
    time.sleep(0.12)

    if sensitive:
        # Overwrite the secret immediately. Don't wait — a clipboard manager
        # could otherwise snapshot the password during the 600 ms window.
        write(old or "")

    # Always restore previous clipboard content (even when it was empty, which
    # clears our text) so we never leave typed content sitting in the clipboard.
    def _restore():
        time.sleep(0.6)
        write(old or "")
    threading.Thread(target=_restore, daemon=True).start()

    return True
