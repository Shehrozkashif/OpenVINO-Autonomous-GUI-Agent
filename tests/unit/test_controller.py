# tests/unit/test_controller.py
import pytest
from unittest.mock import MagicMock, patch
from tools.desktop_control.controller import DesktopController


@pytest.fixture
def mock_httpx_client():
    with patch("tools.desktop_control.controller.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        yield mock_client


def test_click(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"action": "left_click", "x": 100, "y": 200}
    }
    controller = DesktopController()
    assert controller.click(100, 200) is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "mouse_click", "arguments": {"x": 100, "y": 200, "button": "left"}}
    )


def test_double_click(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"action": "left_click", "clicks": 2}
    }
    controller = DesktopController()
    assert controller.double_click(300, 400) is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "mouse_click", "arguments": {"x": 300, "y": 400, "clicks": 2}}
    )


def test_type_text(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"typed": "hello"}
    }
    controller = DesktopController()
    assert controller.type_text("hello") is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "type_text", "arguments": {"text": "hello"}}
    )


def test_press_key(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"key": "enter"}
    }
    controller = DesktopController()
    assert controller.press_key("enter") is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "press_key", "arguments": {"key": "enter"}}
    )


def test_hotkey(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"hotkey": "ctrl+s"}
    }
    controller = DesktopController()
    assert controller.hotkey("ctrl", "s") is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "hotkey", "arguments": {"keys": ["ctrl", "s"]}}
    )


def test_scroll(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"direction": "down", "clicks": 3}
    }
    controller = DesktopController()
    assert controller.scroll(150, 250, clicks=3, direction="down") is True
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "scroll", "arguments": {"x": 150, "y": 250, "clicks": 3, "direction": "down"}}
    )


def test_screenshot_base64(mock_httpx_client):
    mock_httpx_client.post.return_value.json.return_value = {
        "success": True,
        "result": {"image_base64": "aaaa", "width": 800, "height": 600}
    }
    controller = DesktopController()
    assert controller.screenshot_base64() == "aaaa"
    mock_httpx_client.post.assert_called_once_with(
        "http://127.0.0.1:8015/tools/call",
        json={"name": "screenshot", "arguments": {}}
    )


def test_is_server_running(mock_httpx_client):
    mock_httpx_client.get.return_value.status_code = 200
    controller = DesktopController()
    assert controller.is_server_running() is True

    mock_httpx_client.get.side_effect = Exception("conn error")
    assert controller.is_server_running() is False
