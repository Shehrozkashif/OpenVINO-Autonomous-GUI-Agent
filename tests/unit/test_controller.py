# tests/unit/test_controller.py
import pytest
from unittest.mock import MagicMock, patch, call
from tools.desktop_control.controller import DesktopController


@pytest.fixture(autouse=True)
def mock_pynput(monkeypatch):
    """Patch the module-level mouse/keyboard singletons so no real input is sent."""
    mock_mouse = MagicMock()
    mock_kb = MagicMock()
    monkeypatch.setattr("tools.desktop_control.controller._mouse", mock_mouse)
    monkeypatch.setattr("tools.desktop_control.controller._pynput_kb", mock_kb)
    monkeypatch.setattr("tools.desktop_control.controller._XTEST_OK", False)
    return mock_mouse, mock_kb


def test_click(mock_pynput):
    mouse, _ = mock_pynput
    controller = DesktopController()
    result = controller.click(100, 200)
    assert result is True
    assert mouse.press.called
    assert mouse.release.called


def test_double_click(mock_pynput):
    mouse, _ = mock_pynput
    controller = DesktopController()
    result = controller.double_click(300, 400)
    assert result is True
    assert mouse.press.call_count == 2
    assert mouse.release.call_count == 2


def test_type_text(mock_pynput):
    _, kb = mock_pynput
    controller = DesktopController()
    result = controller.type_text("hi")
    assert result is True
    assert kb.type.call_count == 2  # one call per character


def test_press_key(mock_pynput):
    _, kb = mock_pynput
    controller = DesktopController()
    result = controller.press_key("enter")
    assert result is True
    kb.press.assert_called_once()
    kb.release.assert_called_once()


def test_hotkey(mock_pynput):
    _, kb = mock_pynput
    controller = DesktopController()
    result = controller.hotkey("ctrl", "s")
    assert result is True
    assert kb.press.called


def test_scroll(mock_pynput):
    mouse, _ = mock_pynput
    controller = DesktopController()
    result = controller.scroll(150, 250, clicks=3, direction="down")
    assert result is True
    mouse.scroll.assert_called_once_with(0, -3)


def test_screenshot_base64(mock_pynput):
    import base64
    controller = DesktopController()
    result = controller.screenshot_base64()
    assert isinstance(result, str)
    # Must be valid base64
    base64.b64decode(result)


def test_is_server_running(mock_pynput):
    controller = DesktopController()
    assert controller.is_server_running() is True
