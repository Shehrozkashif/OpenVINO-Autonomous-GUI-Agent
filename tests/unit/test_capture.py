# tests/unit/test_capture.py
from desktop.capture import ScreenCapture


def test_screenshot_returns_pil_image():
    cap = ScreenCapture()
    img = cap.capture()
    assert img.width > 0
    assert img.height > 0


def test_resize_works():
    cap = ScreenCapture()
    img = cap.capture_resized(512, 512)
    assert img.width == 512
    assert img.height == 512


def test_has_changed_returns_bool():
    cap = ScreenCapture()
    result = cap.has_changed()
    assert isinstance(result, bool)
