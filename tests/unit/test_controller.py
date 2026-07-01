# tests/unit/test_controller.py
from unittest.mock import MagicMock

import pytest

from core.controller import DesktopController


@pytest.fixture(autouse=True)
def mock_winapi(monkeypatch):
    """Patch the low-level Win32 SendInput wrappers so no real input is sent."""
    mocks = {
        "mouse_event": MagicMock(),
        "key_event": MagicMock(),
        "unicode_event": MagicMock(),
        "set_pos": MagicMock(),
        # VK_A..VK_Z equal ord('A')..ord('Z') on real Windows, so this mirrors
        # VkKeyScanW's real behavior for plain ASCII letters/digits.
        "vk_for_char": MagicMock(side_effect=lambda ch: (ord(ch.upper()), ch.isupper())),
    }
    monkeypatch.setattr("core.controller._mouse_event", mocks["mouse_event"])
    monkeypatch.setattr("core.controller._key_event", mocks["key_event"])
    monkeypatch.setattr("core.controller._unicode_key_event", mocks["unicode_event"])
    monkeypatch.setattr("core.controller._set_cursor_pos", mocks["set_pos"])
    monkeypatch.setattr("core.controller._vk_for_char", mocks["vk_for_char"])
    return mocks


def test_click(mock_winapi):
    controller = DesktopController()
    result = controller.click(100, 200)
    assert result is True
    mock_winapi["set_pos"].assert_called_once_with(100, 200)
    assert mock_winapi["mouse_event"].call_count == 2  # down + up


def test_double_click(mock_winapi):
    controller = DesktopController()
    result = controller.double_click(300, 400)
    assert result is True
    mock_winapi["set_pos"].assert_called_once_with(300, 400)
    assert mock_winapi["mouse_event"].call_count == 4  # 2x (down + up)


def test_type_text(mock_winapi):
    controller = DesktopController()
    result = controller.type_text("hi")
    assert result is True
    # one keydown + one keyup per character, via Unicode injection
    assert mock_winapi["unicode_event"].call_count == 4


def test_press_key(mock_winapi):
    controller = DesktopController()
    result = controller.press_key("enter")
    assert result is True
    assert mock_winapi["key_event"].call_count == 2  # down + up


def test_hotkey(mock_winapi):
    controller = DesktopController()
    result = controller.hotkey("ctrl", "s")
    assert result is True
    assert mock_winapi["key_event"].called


def test_scroll(mock_winapi):
    controller = DesktopController()
    result = controller.scroll(150, 250, clicks=3, direction="down")
    assert result is True
    mock_winapi["set_pos"].assert_called_once_with(150, 250)
    mock_winapi["mouse_event"].assert_called_once_with(0x0800, data=-360)


def test_screenshot_base64(mock_winapi):
    import base64
    controller = DesktopController()
    result = controller.screenshot_base64()
    assert isinstance(result, str)
    # Must be valid base64
    base64.b64decode(result)
