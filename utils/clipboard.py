# utils/clipboard.py
"""Windows clipboard read/write via pyperclip."""
import threading
import time

# ── Write ─────────────────────────────────────────────────────────────────────

def write(text: str) -> bool:
    """Set clipboard text. Returns True on success."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


# ── Read ──────────────────────────────────────────────────────────────────────

def read() -> str:
    """Get current clipboard text."""
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:
        return ""


# ── Available check ───────────────────────────────────────────────────────────

def available() -> bool:
    """Return True if clipboard operations are available on this system."""
    try:
        import pyperclip  # noqa: F401
        return True
    except ImportError:
        return False


# ── Clipboard-paste typing ────────────────────────────────────────────────────

def paste_type(text: str, send_hotkey_fn, sensitive: bool = False) -> bool:
    """Type text by setting the clipboard and sending ctrl+v.
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
