# tests/unit/test_capture.py
import base64
import pytest
from core.capture.screenshot import ScreenCapture


def test_screenshot_returns_pil_image():
    cap = ScreenCapture()
    img = cap.capture()
    assert img.width > 0
    assert img.height > 0


def test_base64_encoding_is_valid_jpeg():
    cap = ScreenCapture()
    b64 = cap.capture_as_base64()
    assert isinstance(b64, str)
    raw = base64.b64decode(b64)
    # JPEG magic bytes: FF D8
    assert raw[:2] == b'\xff\xd8'


def test_resize_works():
    cap = ScreenCapture()
    img = cap.capture_resized(512, 512)
    assert img.width == 512
    assert img.height == 512


def test_has_changed_returns_bool():
    cap = ScreenCapture()
    result = cap.has_changed()
    assert isinstance(result, bool)


def test_dpi_scale_is_positive():
    cap = ScreenCapture()
    scale = cap.get_dpi_scale()
    assert scale > 0
