# core/capture/screen_snapshot.py
"""
ScreenSnapshot — structured world model for the current display state.

Captures foreground window identity and assigns each OCR word to its
owning window, marking whether it belongs to the foreground or not.
This lets the planner ignore background-window text and the grounder
optionally restrict matching to interactive (foreground) regions.

Non-Windows: foreground detection falls back to marking all regions as
foreground (safe default — no filtering applied).
"""
import platform
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import imagehash

_IS_WINDOWS = platform.system() == "Windows"


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class OCRRegion:
    text: str
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in screen pixels
    confidence: float
    window_id: int                     # hwnd (0 on non-Windows or unknown)
    window_title: str
    is_in_foreground: bool


@dataclass
class ScreenSnapshot:
    timestamp: float
    foreground_window_title: str
    foreground_process: str
    screen_hash: str
    ocr_regions: list[OCRRegion] = field(default_factory=list)
    # (name, control_type) pairs from the foreground window's UIA tree —
    # ground-truth clickable elements, Windows only ([] elsewhere).
    interactive_elements: list[tuple[str, str]] = field(default_factory=list)

    def foreground_texts(self) -> list[str]:
        """Deduplicated text tokens visible in the foreground window."""
        seen, out = set(), []
        for r in self.ocr_regions:
            if r.is_in_foreground and r.text.lower() not in seen:
                seen.add(r.text.lower())
                out.append(r.text)
        return out

    def background_by_window(self) -> dict:
        """{window_title: [text, ...]} for all non-foreground regions."""
        result: dict = {}
        for r in self.ocr_regions:
            if not r.is_in_foreground:
                bucket = result.setdefault(r.window_title, [])
                if r.text not in bucket:
                    bucket.append(r.text)
        return result

    def format_for_planner(self) -> str:
        """Human-readable context string for the planning prompt."""
        proc = self.foreground_process or "unknown"
        lines = [f'Foreground window: "{self.foreground_window_title}" ({proc})']

        if self.interactive_elements:
            quoted = ", ".join(
                f'"{name}" ({ctype})' for name, ctype in self.interactive_elements[:40]
            )
            lines.append(
                f"Clickable controls (accessibility tree — these labels are "
                f"reliable click targets): {quoted}"
            )

        fg_texts = self.foreground_texts()
        if fg_texts:
            quoted = ", ".join(f'"{t}"' for t in fg_texts[:40])
            lines.append(f"UI elements in foreground: {quoted}")
        else:
            lines.append("UI elements in foreground: (none detected)")

        bg = self.background_by_window()
        if bg:
            lines.append("Other visible windows (background — not interactive):")
            for title, texts in list(bg.items())[:5]:
                sample = ", ".join(f'"{t}"' for t in texts[:6])
                lines.append(f"  [{title}]: {sample}")

        return "\n".join(lines)


# ── Windows helpers ────────────────────────────────────────────────────────────

def _get_foreground_hwnd_and_title() -> tuple[int, str]:
    if not _IS_WINDOWS:
        return 0, "Desktop"
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return hwnd, buf.value or "Unknown"
    except Exception:
        return 0, "Unknown"


def _get_foreground_process(hwnd: int) -> str:
    if not _IS_WINDOWS or not hwnd:
        return "unknown"
    try:
        import ctypes
        import ctypes.wintypes
        import os
        pid = ctypes.wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h_proc = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid.value)
        if not h_proc:
            return "unknown"
        buf = ctypes.create_unicode_buffer(260)
        # GetModuleFileNameExW lives in psapi / kernel32 depending on Windows version
        try:
            ctypes.windll.psapi.GetModuleFileNameExW(h_proc, None, buf, 260)
        except Exception:
            ctypes.windll.kernel32.GetModuleFileNameExW(h_proc, None, buf, 260)
        ctypes.windll.kernel32.CloseHandle(h_proc)
        return os.path.basename(buf.value) if buf.value else "unknown"
    except Exception:
        return "unknown"


def _enum_visible_windows() -> list[tuple[int, str, tuple[int, int, int, int]]]:
    """[(hwnd, title, (left, top, right, bottom))] for all non-minimised windows."""
    if not _IS_WINDOWS:
        return []
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        results: list = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
        def _cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if rect.right <= rect.left or rect.bottom <= rect.top:
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
            results.append((hwnd, title, (rect.left, rect.top, rect.right, rect.bottom)))
            return True

        user32.EnumWindows(_cb, 0)
        return results
    except Exception:
        return []


def _point_in_rect(px: int, py: int, rect: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px < x2 and y1 <= py < y2


# ── Public API ─────────────────────────────────────────────────────────────────

def capture_snapshot(capturer, ocr) -> ScreenSnapshot:
    """
    Build a ScreenSnapshot:
    1. Get foreground window title + process via Win32 API.
    2. Enumerate all visible window rects (front-to-back order).
    3. Run OCR on the current screenshot thumbnail.
    4. Assign each OCR word to the topmost containing window rect.
    5. Mark each region is_in_foreground based on window ownership.
    """
    # Avoid importing at module level to prevent circular imports.
    # OCREngine lives in agents.grounding; screen_snapshot lives in core.capture.
    from agents.grounding.grounding_agent import OCRWord  # noqa: PLC0415

    ts = time.time()

    fg_hwnd, fg_title = _get_foreground_hwnd_and_title()
    fg_process = _get_foreground_process(fg_hwnd)
    windows = _enum_visible_windows()

    interactive: list = []
    if _IS_WINDOWS:
        try:
            from core.grounding.windows_uia import get_interactive_elements  # noqa: PLC0415
            interactive = get_interactive_elements()
        except Exception:
            interactive = []

    img = capturer.capture()
    thumb = img.copy()
    thumb.thumbnail((960, 540))
    screen_hash = str(imagehash.phash(thumb))

    words: list[OCRWord] = ocr.extract(thumb) if ocr.is_available() else []

    # OCR ran on the 960×540 thumbnail; scale word coords back to full-screen pixels
    scale_x = img.width / thumb.width if thumb.width > 0 else 1.0
    scale_y = img.height / thumb.height if thumb.height > 0 else 1.0

    regions: list[OCRRegion] = []
    for w in words:
        # Skip low-quality or special-character tokens (mirrors _get_screen_context)
        if (
            len(w.text) < 2
            or w.conf < 0.70
            or any(c in w.text for c in ('/', '\\', '=', '{', '}', '$', '#'))
        ):
            continue

        cx = int((w.x + w.w / 2) * scale_x)
        cy = int((w.y + w.h / 2) * scale_y)
        bbox = (
            int(w.x * scale_x),
            int(w.y * scale_y),
            int((w.x + w.w) * scale_x),
            int((w.y + w.h) * scale_y),
        )

        # Assign to the topmost window that contains this point
        owner_hwnd, owner_title = 0, "Desktop"
        for hwnd, title, rect in windows:
            if _point_in_rect(cx, cy, rect):
                owner_hwnd, owner_title = hwnd, title
                break

        # On non-Windows (fg_hwnd == 0) treat everything as foreground
        is_fg = (fg_hwnd == 0) or (owner_hwnd == fg_hwnd)

        # Tag the OCRWord so downstream grounding can apply foreground_only filtering
        if is_fg:
            w.element_type = "foreground_interactive"
        # else: leave as default "document_text"

        regions.append(OCRRegion(
            text=w.text,
            bbox=bbox,
            confidence=w.conf,
            window_id=owner_hwnd,
            window_title=owner_title or "Desktop",
            is_in_foreground=is_fg,
        ))

    return ScreenSnapshot(
        timestamp=ts,
        foreground_window_title=fg_title,
        foreground_process=fg_process,
        screen_hash=screen_hash,
        ocr_regions=regions,
        interactive_elements=interactive,
    )
