# tests/unit/test_controller.py
from unittest.mock import MagicMock

import pytest

from desktop.input import DesktopController


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
    # A click also glides the cursor, focuses the window under it, and sends
    # hover MOVE events — all of which call Win32 directly.
    mocks["move_abs"] = MagicMock()
    mocks["glide"] = MagicMock()
    mocks["focus_at"] = MagicMock()
    monkeypatch.setattr("desktop.input._mouse_event", mocks["mouse_event"])
    monkeypatch.setattr("desktop.input._key_event", mocks["key_event"])
    monkeypatch.setattr("desktop.input._unicode_key_event", mocks["unicode_event"])
    monkeypatch.setattr("desktop.input._set_cursor_pos", mocks["set_pos"])
    monkeypatch.setattr("desktop.input._vk_for_char", mocks["vk_for_char"])
    monkeypatch.setattr("desktop.input._mouse_move_abs", mocks["move_abs"])
    monkeypatch.setattr("desktop.input._glide_cursor", mocks["glide"])
    monkeypatch.setattr("desktop.input._focus_window_at", mocks["focus_at"])
    return mocks


def test_click(mock_winapi):
    """A click glides to the point, hovers it, then presses and releases."""
    controller = DesktopController()
    result = controller.click(100, 200)
    assert result is True
    mock_winapi["glide"].assert_called_once_with(100, 200)
    mock_winapi["focus_at"].assert_called_once_with(100, 200)
    # Last hover move must land exactly on the target (the 1px settle move
    # before it is what guarantees a WM_MOUSEMOVE arrives there at all).
    assert mock_winapi["move_abs"].call_args.args == (100, 200)
    assert mock_winapi["mouse_event"].call_count == 2  # down + up


def test_double_click(mock_winapi):
    controller = DesktopController()
    result = controller.double_click(300, 400)
    assert result is True
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
    mock_winapi["glide"].assert_called_once_with(150, 250)
    mock_winapi["mouse_event"].assert_called_once_with(0x0800, data=-360)


