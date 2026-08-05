# desktop/capture.py
"""Screen capture (GDI via PIL.ImageGrab), frame hashing, and the mask that
keeps the agent's own window out of its own screenshots.
"""
import imagehash
from loguru import logger
from PIL import Image


def _pil_grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    """Capture screen (or region) via PIL.ImageGrab (GDI BitBlt)."""
    from PIL import ImageGrab
    if w and h:
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    else:
        img = ImageGrab.grab()
    return img.convert("RGB")


def _grab(x: int = 0, y: int = 0, w: int = 0, h: int = 0) -> Image.Image:
    return _pil_grab(x, y, w, h)


# Nominal size reported when the machine has no display at all. Nothing can be
# grounded or clicked in that state, so the number is never used for real work —
# it exists so constructing the agent does not explode on a headless box.
_NO_DISPLAY_SIZE = (1920, 1080)
_no_display_logged = False


def _screen_size() -> tuple[int, int]:
    """Physical pixel size of the primary display.

    GetDeviceCaps answers without capturing a frame, so it is tried first on
    Windows. PIL's grab is the cross-platform fallback — and it needs a display.

    A headless machine (CI, a server session, a container) has none, and
    ImageGrab raises "X connection failed". That must not take the caller down:
    TaskOrchestrator.__init__ and UIGroundingAgent.__init__ both ask for the
    screen size, so an exception here makes the entire policy layer
    unconstructible off Windows — which is exactly where its tests run.
    desktop/ reports facts; "there is no screen" is a fact, not a crash.
    """
    try:
        import ctypes
        hdc = ctypes.windll.user32.GetDC(0)
        w = ctypes.windll.gdi32.GetDeviceCaps(hdc, 118)  # DESKTOPHORZRES
        h = ctypes.windll.gdi32.GetDeviceCaps(hdc, 117)  # DESKTOPVERTRES
        ctypes.windll.user32.ReleaseDC(0, hdc)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    try:
        img = _pil_grab()
        return img.width, img.height
    except Exception as e:
        global _no_display_logged
        if not _no_display_logged:
            _no_display_logged = True
            logger.warning(
                f"[CAPTURE] No display available ({e}) — reporting a nominal "
                f"{_NO_DISPLAY_SIZE[0]}x{_NO_DISPLAY_SIZE[1]} screen. Capture "
                f"and grounding cannot work here; only offline logic will run."
            )
        return _NO_DISPLAY_SIZE


# ── Frame comparison ──────────────────────────────────────────────────────────

# Resolution and hash size used for before/after change detection. The previous
# 320×180 / 8-bit-DCT phash was far too coarse: a context menu or small dialog on
# a 1080p+ display changed zero hash bits, so legitimately-opened menus were
# scored as "no change → click failed" (a systematic false failure, H2). A larger
# thumbnail with a 16×16 hash (256-bit) makes small but real UI changes detectable
# while staying cheap. Both the orchestrator (pre-action) and the reflection agent
# (post-action) MUST use this helper so the two hashes are directly comparable.
_FRAME_HASH_SIZE = 16
_FRAME_THUMB = (960, 540)

# Shared thumbnail size for ALL OCR passes (grounding, screen snapshot,
# reflection, dialog checks). One size means identical pixels → identical
# phash → every consumer hits the same OCR cache entry. Raised from 960×540:
# halving a desktop screenshot made small UI text (menu items, dialog labels)
# unreadable to OCR, which cost both element recall and verification accuracy.
# The price is ~1.8× OCR pixels, paid only on cache misses.
OCR_THUMB = (1280, 720)


def frame_phash(img: Image.Image) -> "imagehash.ImageHash":
    """Perceptual hash tuned for before/after UI change detection (see H2)."""
    thumb = img.copy()
    thumb.thumbnail(_FRAME_THUMB, Image.LANCZOS)
    return imagehash.phash(thumb, hash_size=_FRAME_HASH_SIZE)


# ── Public API ────────────────────────────────────────────────────────────────

class ScreenCapture:
    def __init__(self, monitor: int = 1):
        self.monitor = monitor
        self._last_hash: imagehash.ImageHash | None = None
        # Regions (x1, y1, x2, y2) to black out in every captured frame.
        # Used to mask the agent's own GUI window so its text doesn't pollute OCR.
        # NOTE: the orchestrator overwrites this list every refresh cycle.
        self.exclude_regions: list = []
        # Additional always-applied mask regions that survive the orchestrator's
        # exclude_regions refresh — used for the always-on-top mission HUD,
        # whose text would otherwise contaminate OCR/planning.
        self.persistent_exclude_regions: list = []

    def capture(self) -> Image.Image:
        """Full-screen capture. Does not alter input focus on any platform."""
        img = _grab()
        regions = self.exclude_regions + self.persistent_exclude_regions
        if regions:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            for (x1, y1, x2, y2) in regions:
                draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        return img

    def capture_resized(self, width: int, height: int) -> Image.Image:
        return self.capture().resize((width, height), Image.LANCZOS)

    def has_changed(self, threshold: float = 0.05) -> bool:
        current = self.capture()
        current_hash = imagehash.phash(current)
        if self._last_hash is None:
            self._last_hash = current_hash
            return True
        changed = (self._last_hash - current_hash) / 64.0 > threshold
        if changed:
            self._last_hash = current_hash
        return changed


# ── Hiding the agent's own window from itself ─────────────────────────────────

class OwnWindowMask:
    """Blacks out the agent's own GUI window in every frame it captures.

    Why this exists: the GUI prints every step it takes ("CLICK, [visual] click
    at (847,39)"). Left visible, OCR reads that log back and the verifier
    accepts the agent's own output as proof the screen changed — the root cause
    of a long run of hallucinated "verified" verdicts.

    Masking is decided by hit-testing a GRID over the window rect and blacking
    out ONLY the cells the OS says our process still owns. A blanket mask of the
    whole rect was catastrophic when the window is near-fullscreen: the task's
    app sits on top of most of it, and masking the rect erased the app's pixels
    too (OCR collapsed to 13 regions and the agent spent five subtasks clicking
    the unmasked edge slivers).
    """

    GRID = (12, 8)          # 96 hit-tests, ~1 ms total
    _TITLE = "Desktop GUI Agent"

    def __init__(self, capturer: ScreenCapture, log=None):
        self.capturer = capturer
        self._log = log or (lambda _msg: None)
        self._hwnd: int | None = None
        self._looked_up = False

    @property
    def hwnd(self) -> int | None:
        """Handle of the agent's own window, found once and cached."""
        if not self._looked_up:
            self._looked_up = True
            self._hwnd = self._find_own_window()
        return self._hwnd

    @hwnd.setter
    def hwnd(self, value: int | None) -> None:
        """Set the handle directly, skipping the title search."""
        self._hwnd = value
        self._looked_up = True

    def _find_own_window(self) -> int | None:
        """Locate our window by title substring.

        EnumWindows + substring match, not FindWindowW: the title contains a
        dash whose exact character encoding differs between sources, and an
        exact match silently found nothing.
        """
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32
            found = [0]

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
            def _cb(hwnd, _lparam):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if self._TITLE in buf.value:
                        found[0] = hwnd
                        return False   # stop enumeration
                return True

            user32.EnumWindows(_cb, 0)
            if found[0]:
                logger.info(f"[MASK] GUI window handle cached: hwnd={found[0]}")
                return found[0]
            logger.debug("[MASK] GUI window not found via EnumWindows — no masking")
        except Exception as e:
            logger.warning(f"[MASK] GUI window lookup failed: {e}")
        return None

    def refresh(self) -> None:
        """Recompute the mask for the CURRENT screen. Call immediately before
        any capture that feeds OCR, grounding or verification — the action just
        performed may have raised our own window.
        """
        hwnd = self.hwnd
        if not hwnd:
            self.capturer.exclude_regions = []
            return
        try:
            import ctypes
            import ctypes.wintypes
            user32 = ctypes.windll.user32

            if user32.IsIconic(hwnd):        # minimised — none of our pixels show
                if self.capturer.exclude_regions:
                    self.capturer.exclude_regions = []
                    logger.debug("[MASK] GUI window minimized — mask cleared")
                return

            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            our_pid = ctypes.wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(our_pid))
            cells = self._owned_cells(rect, our_pid.value)

            if cells:
                if self.capturer.exclude_regions != cells:
                    self.capturer.exclude_regions = cells
                    gx, gy = self.GRID
                    logger.info(
                        f"[MASK] GUI window partially visible — masked "
                        f"{len(cells)}/{gx * gy} cells of "
                        f"({rect.left}, {rect.top}, {rect.right}, {rect.bottom})"
                    )
            elif self.capturer.exclude_regions:
                self.capturer.exclude_regions = []
                logger.debug("[MASK] GUI window covered — mask cleared")
        except Exception as e:
            # WARNING, not debug: a silent failure here re-enables the agent
            # reading its own GUI text, so it must be visible in the log.
            logger.warning(f"[MASK] GUI window mask lookup failed: {e}")

    def _owned_cells(self, rect, our_pid: int) -> list[tuple[int, int, int, int]]:
        """Grid cells of the window rect whose pixels our process still owns."""
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32
        user32.WindowFromPoint.argtypes = [ctypes.wintypes.POINT]
        user32.WindowFromPoint.restype = ctypes.wintypes.HWND

        gx_n, gy_n = self.GRID
        w = max(rect.right - rect.left, 1)
        h = max(rect.bottom - rect.top, 1)
        cw, ch = w / gx_n, h / gy_n
        cells = []
        for gy in range(gy_n):
            for gx in range(gx_n):
                sx = int(rect.left + (gx + 0.5) * cw)
                sy = int(rect.top + (gy + 0.5) * ch)
                hit = user32.WindowFromPoint(ctypes.wintypes.POINT(sx, sy))
                if not hit:
                    continue
                root = user32.GetAncestor(hit, 2) or hit    # 2 = GA_ROOT
                pid = ctypes.wintypes.DWORD(0)
                user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
                if pid.value == our_pid:
                    cells.append((
                        int(rect.left + gx * cw),
                        int(rect.top + gy * ch),
                        int(rect.left + (gx + 1) * cw),
                        int(rect.top + (gy + 1) * ch),
                    ))
        return cells
