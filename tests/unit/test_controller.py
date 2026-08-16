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




class TestGlideDuration:
    """AGENT_GLIDE_MAX_S slows the cursor so a person can watch it work.

    The glide is cosmetic (GDI screenshots do not capture the cursor), so the
    only thing to pin is that the override is read, clamped, and never lets a
    bad value break a click.
    """

    def test_default_cap_without_the_env_var(self, monkeypatch):
        from desktop import input as di
        monkeypatch.delenv("AGENT_GLIDE_MAX_S", raising=False)
        assert di._glide_max_s() == di._GLIDE_MAX_S

    def test_env_var_raises_the_cap(self, monkeypatch):
        from desktop import input as di
        monkeypatch.setenv("AGENT_GLIDE_MAX_S", "1.5")
        assert di._glide_max_s() == 1.5

    def test_absurd_value_is_clamped(self, monkeypatch):
        """A typo must not park the cursor mid-flight for a minute."""
        from desktop import input as di
        monkeypatch.setenv("AGENT_GLIDE_MAX_S", "600")
        assert di._glide_max_s() == 10.0

    def test_below_the_floor_is_clamped_up(self, monkeypatch):
        from desktop import input as di
        monkeypatch.setenv("AGENT_GLIDE_MAX_S", "0")
        assert di._glide_max_s() == di._GLIDE_MIN_S

    def test_garbage_falls_back_to_the_default(self, monkeypatch):
        from desktop import input as di
        monkeypatch.setenv("AGENT_GLIDE_MAX_S", "slow please")
        assert di._glide_max_s() == di._GLIDE_MAX_S


class TestArithmeticKeyNames:
    """Live failure: 'add any two numbers using calculator'.

    The planner emitted key_press "add" between the operands. The key table had
    no arithmetic keys, so every attempt logged "Unknown key 'add' - skipping",
    failed three times, and pushed the run into a task-level re-plan. It never
    got past the first operator.
    """

    @pytest.mark.parametrize("name, vk", [
        ("add", 0x6B), ("plus", 0x6B),
        ("subtract", 0x6D), ("minus", 0x6D),
        ("multiply", 0x6A), ("asterisk", 0x6A), ("star", 0x6A),
        ("divide", 0x6F), ("slash", 0x6F),
        ("decimal", 0x6E),
        ("equals", 0x0D), ("equal", 0x0D),
    ])
    def test_operator_resolves(self, name, vk):
        from desktop.input import _resolve_key
        assert _resolve_key(name) is not None, f"{name!r} is still unknown"
        assert _resolve_key(name)[0] == vk

    @pytest.mark.parametrize("digit", range(10))
    def test_numpad_digits_resolve(self, digit):
        from desktop.input import _resolve_key
        assert _resolve_key(f"numpad{digit}")[0] == 0x60 + digit

    def test_names_are_case_insensitive(self):
        from desktop.input import _resolve_key
        assert _resolve_key("ADD") == _resolve_key("add")

    def test_operators_need_no_shift(self):
        """Numpad codes were chosen so layout and shift state cannot matter."""
        from desktop.input import _resolve_key
        for name in ("add", "subtract", "multiply", "divide", "decimal"):
            _vk, _extended, needs_shift = _resolve_key(name)
            assert needs_shift is False, name

    def test_a_genuinely_unknown_key_still_fails(self):
        """The honest-failure path must survive the additions."""
        from desktop.input import _resolve_key, _send_key_name
        assert _resolve_key("frobnicate") is None
        assert _send_key_name("frobnicate") is False

    def test_plus_as_a_character_is_not_split_as_a_chord(self, mock_winapi):
        """press_key routes 'a+b' to the hotkey path; a bare '+' must not go there."""
        c = DesktopController()
        assert c.press_key("+") is True
