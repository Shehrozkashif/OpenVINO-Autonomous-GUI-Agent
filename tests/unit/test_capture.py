# tests/unit/test_capture.py
"""ScreenCapture against a real screen, plus the headless fallback.

The first three tests grab actual pixels, so they need a display and skip
without one. Everything else in tests/unit/ must run headless — CI has no
display, and that is the point: a suite that only passes in front of a desktop
would not have caught `_screen_size()` raising on a bare runner.
"""
import pytest
from PIL import ImageGrab

from desktop.capture import _NO_DISPLAY_SIZE, ScreenCapture, _screen_size


def _has_display() -> bool:
    try:
        ImageGrab.grab()
        return True
    except Exception:
        return False


needs_display = pytest.mark.skipif(
    not _has_display(), reason="no display: real screen capture is impossible here"
)


@needs_display
def test_screenshot_returns_pil_image():
    cap = ScreenCapture()
    img = cap.capture()
    assert img.width > 0
    assert img.height > 0


@needs_display
def test_resize_works():
    cap = ScreenCapture()
    img = cap.capture_resized(512, 512)
    assert img.width == 512
    assert img.height == 512


@needs_display
def test_has_changed_returns_bool():
    cap = ScreenCapture()
    result = cap.has_changed()
    assert isinstance(result, bool)


def _no_display(*a, **k):
    raise OSError("X connection failed: error 5")


class TestScreenSizeWithoutADisplay:
    """_screen_size() must answer on a machine with no screen.

    It is called from TaskOrchestrator.__init__ and UIGroundingAgent.__init__,
    so a raise here makes the whole policy layer unconstructible — which is how
    it broke CI: every orchestrator test errored with "X connection failed"
    while passing locally, where a display happens to exist.
    """

    def test_returns_nominal_size_instead_of_raising(self, monkeypatch):
        monkeypatch.setattr("desktop.capture._pil_grab", _no_display)
        monkeypatch.setattr("desktop.capture._no_display_logged", False)
        assert _screen_size() == _NO_DISPLAY_SIZE

    def test_the_orchestrator_can_still_be_built(self, monkeypatch):
        """The regression this guards: constructing the agent off Windows."""
        from unittest.mock import MagicMock

        from core.orchestrator import TaskOrchestrator
        from core.runstate import OrchestratorConfig
        from tests.unit.conftest import make_grounder, make_history, make_reflector

        monkeypatch.setattr("desktop.capture._pil_grab", _no_display)
        orch = TaskOrchestrator(
            router=MagicMock(), planner=MagicMock(), grounder=make_grounder(),
            actor=MagicMock(), reflector=make_reflector(), capturer=MagicMock(),
            history=make_history(), config=OrchestratorConfig(),
            on_step_log=lambda _: None, ocr=MagicMock(),
        )
        assert (orch._screen_w, orch._screen_h) == _NO_DISPLAY_SIZE
