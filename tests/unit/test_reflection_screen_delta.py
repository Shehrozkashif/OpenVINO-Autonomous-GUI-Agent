# tests/unit/test_reflection_screen_delta.py
"""
Unit tests for perceptual-hash screen-delta check as ground truth for clicks.

If the perceptual hash of the screen does not change between the moment just
after the click fires and the moment after the adaptive wait, the reflector
returns an immediate failure (confidence=0.98) without calling the LLM or VLM.

Three properties are verified:

1. DELTA-ZERO     — identical before/after screenshots → immediate failure,
                    LLM/VLM not called, confidence=0.98, should_retry=True.
2. DELTA-NONZERO  — different before/after screenshots → LLM/VLM is called,
                    no short-circuit.
3. ACTION-SCOPING — delta check applies only to click/right_click/double_click,
                    NOT to type/key_press/hotkey/scroll.
"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock
from PIL import Image

from agents.reflection.reflection_agent import ReflectionAgent
from core.protocols.a2a import ActionStep


# ── helpers ───────────────────────────────────────────────────────────────────

def _step(action_type="click", target="SomeButton", key=None, value=None):
    s = ActionStep(
        id=1, subtask_id=1,
        action_type=action_type,
        target=target, value=value, key=key,
        description=f"{action_type} {target or key or value}",
        verification="expected result",
    )
    return s


def _solid(color=(128, 128, 128), size=(200, 100)):
    return Image.new("RGB", size, color)


def _make_agent_click(before_img, after_img):
    """
    Agent configured for a click path test.
    capture() is called exactly twice: first for the before-hash, then for the
    after-screenshot (_adaptive_wait is mocked so it never calls has_changed).
    """
    agent = ReflectionAgent.__new__(ReflectionAgent)
    agent._adaptive_wait = MagicMock()   # instant, no extra capture() calls
    agent.capturer = MagicMock()
    agent.capturer.capture = MagicMock(side_effect=[before_img, after_img])
    agent._ocr = MagicMock()
    agent._ocr.extract = MagicMock(return_value=[])
    agent.client = MagicMock()
    agent.client.query_llm = MagicMock()
    agent.client.query_vlm = MagicMock()
    agent.min_confidence = 0.75
    return agent


def _make_agent_nonclick(after_img):
    """
    Agent configured for a non-click path test.
    capture() is called exactly once (the after-screenshot; no before-hash).
    """
    agent = ReflectionAgent.__new__(ReflectionAgent)
    agent._adaptive_wait = MagicMock()
    agent.capturer = MagicMock()
    agent.capturer.capture = MagicMock(return_value=after_img)
    agent._ocr = MagicMock()
    agent._ocr.extract = MagicMock(return_value=[
        MagicMock(text="bar", conf=0.9),
        MagicMock(text="foo", conf=0.9),
        MagicMock(text="baz", conf=0.9),
    ])
    agent.client = MagicMock()
    agent.client.query_llm = MagicMock(return_value=MagicMock(
        content='{"success": true, "confidence": 0.9, "observation": "ok"}'
    ))
    agent.client.query_vlm = MagicMock()
    agent.min_confidence = 0.75
    return agent


# ── 1. Delta-zero tests (click actions, screen did not change) ────────────────

class TestDeltaZero:
    """Identical before/after → immediate failure without LLM call."""

    def test_click_unchanged_returns_failure(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("click"))

        assert result.success is False

    def test_click_unchanged_confidence_is_098(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("click"))

        assert result.confidence == pytest.approx(0.98)

    def test_click_unchanged_should_retry_is_true(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("click"))

        assert result.should_retry is True

    def test_click_unchanged_llm_not_called(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        agent.verify(_step("click"))

        agent.client.query_llm.assert_not_called()
        agent.client.query_vlm.assert_not_called()

    def test_right_click_unchanged_returns_failure(self):
        img = _solid((50, 100, 150))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("right_click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.98)
        agent.client.query_llm.assert_not_called()

    def test_double_click_unchanged_returns_failure(self):
        img = _solid((0, 0, 0))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("double_click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.98)

    def test_click_unchanged_observation_mentions_screen_unchanged(self):
        img = _solid()
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("click"))

        assert "unchanged" in result.observation.lower()

    def test_click_unchanged_recovery_hint_mentions_regrounding(self):
        img = _solid()
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step("click"))

        assert "ground" in result.recovery_hint.lower()


# ── 2. Delta-nonzero tests (click actions, screen changed) ───────────────────

class TestDeltaNonzero:
    """Different before/after → screen-delta check passes, LLM/VLM path runs."""

    def test_click_changed_calls_vlm(self):
        """
        Sparse OCR (0 words) + click → VLM path.
        LLM must NOT be called; VLM must be called.
        """
        before = _solid((0, 0, 0))
        after = _solid((255, 255, 255))
        agent = _make_agent_click(before_img=before, after_img=after)
        agent.client.query_vlm = MagicMock(return_value=MagicMock(
            content='{"success": true, "confidence": 0.85, "observation": "menu appeared"}'
        ))

        agent.verify(_step("click"))

        agent.client.query_vlm.assert_called_once()
        agent.client.query_llm.assert_not_called()

    def test_click_changed_with_rich_ocr_calls_llm(self):
        """
        Rich OCR (≥3 words) + click → LLM path.
        """
        before = _solid((0, 0, 0))
        after = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=before, after_img=after)
        agent._ocr.extract = MagicMock(return_value=[
            MagicMock(text="New", conf=0.9),
            MagicMock(text="Folder", conf=0.9),
            MagicMock(text="File", conf=0.9),
        ])
        agent.client.query_llm = MagicMock(return_value=MagicMock(
            content='{"success": true, "confidence": 0.9, "observation": "context menu visible"}'
        ))

        agent.verify(_step("click"))

        agent.client.query_llm.assert_called_once()

    def test_click_changed_result_follows_llm(self):
        """Result comes from LLM, not from the delta check."""
        before = _solid((0, 0, 0))
        after = _solid((255, 0, 0))
        agent = _make_agent_click(before_img=before, after_img=after)
        agent._ocr.extract = MagicMock(return_value=[
            MagicMock(text="aaa", conf=0.9),
            MagicMock(text="bbb", conf=0.9),
            MagicMock(text="ccc", conf=0.9),
        ])
        agent.client.query_llm = MagicMock(return_value=MagicMock(
            content='{"success": false, "confidence": 0.8, "observation": "wrong menu"}'
        ))

        result = agent.verify(_step("click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.8)   # not 0.98


# ── 3. Action-scoping tests (non-click actions skip the delta check) ──────────

class TestActionScoping:
    """
    type, key_press, hotkey, scroll must not be subject to the delta check.
    - Only one capture() call (no before-hash capture).
    - Even with an "unchanged" screen, the LLM is still called.
    """

    def test_type_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("type", value="hello"))
        assert agent.capturer.capture.call_count == 1

    def test_key_press_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("key_press", key="super"))
        assert agent.capturer.capture.call_count == 1

    def test_hotkey_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("hotkey", key="ctrl+s"))
        assert agent.capturer.capture.call_count == 1

    def test_scroll_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("scroll", value="down"))
        assert agent.capturer.capture.call_count == 1

    def test_type_calls_llm_regardless_of_screen(self):
        """type always calls LLM (forced, regardless of screen change)."""
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("type", value="hello"))
        agent.client.query_llm.assert_called_once()

    def test_key_press_calls_llm_regardless_of_screen(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step("key_press", key="enter"))
        agent.client.query_llm.assert_called_once()

    def test_type_result_not_098_confidence(self):
        """type result must never be the delta-check sentinel confidence=0.98."""
        agent = _make_agent_nonclick(_solid())
        result = agent.verify(_step("type", value="hello"))
        assert result.confidence != pytest.approx(0.98) or result.success is True
