# tests/unit/test_reflection_vlm_escalation.py
"""
Tests for VLM screenshot escalation on uncertain visual verdicts.

The OCR→LLM path is blind to visual-only changes (selection highlight, toggle,
icon state). When it is uncertain about a *visual* action, the reflector takes a
second look with the VLM on the real screenshot and reconciles the two verdicts.
"""
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock

from PIL import Image

from agents.reflection.reflection_agent import ReflectionAgent, ReflectionResult
from core.capture.screenshot import frame_phash
from core.protocols.a2a import ActionStep


def _step(action_type="click", target="Item"):
    return ActionStep(id=1, subtask_id=1, action_type=action_type, target=target,
                      value="hello" if action_type == "type" else None,
                      key=None, description=f"{action_type} {target}",
                      verification="something changed")


def _verdict_json(success, conf):
    return ('{"success": %s, "confidence": %s, "observation": "x", '
            '"error_description": "", "should_retry": true, "recovery_hint": ""}'
            % ("true" if success else "false", conf))


def _make_agent(llm_json, vlm_json, ocr_words=3, escalate=True):
    client = MagicMock()
    client.query_llm = MagicMock(return_value=MagicMock(content=llm_json))
    client.query_vlm = MagicMock(return_value=MagicMock(content=vlm_json))

    ocr = MagicMock()
    ocr.is_available = MagicMock(return_value=True)
    ocr.extract = MagicMock(return_value=[
        MagicMock(text=f"word{i}", conf=0.9) for i in range(ocr_words)
    ])

    capturer = MagicMock()
    capturer.capture = MagicMock(return_value=Image.new("RGB", (200, 100), (255, 255, 255)))

    agent = ReflectionAgent(client, capturer, min_confidence=0.75, ocr=ocr,
                            escalate_uncertain=escalate)
    agent._adaptive_wait = MagicMock()
    return agent, client


# Pre-hash of a different image so the delta check never short-circuits.
_PRE = frame_phash(Image.new("RGB", (200, 100), (0, 0, 0)))


# ── _reconcile unit tests ─────────────────────────────────────────────────────

class TestReconcile:
    def _agent(self):
        a = ReflectionAgent.__new__(ReflectionAgent)
        a.min_confidence = 0.75
        return a

    def _r(self, success, conf):
        return ReflectionResult(success, conf, "obs", "", True, "")

    def test_confident_vlm_success_overrides_uncertain_fail(self):
        out = self._agent()._reconcile(self._r(False, 0.5), self._r(True, 0.9))
        assert out.success is True

    def test_confident_vlm_fail_overrides_uncertain_success(self):
        out = self._agent()._reconcile(self._r(True, 0.6), self._r(False, 0.9))
        assert out.success is False

    def test_both_uncertain_keeps_higher_confidence(self):
        out = self._agent()._reconcile(self._r(False, 0.4), self._r(True, 0.6))
        assert out.confidence == 0.6

    def test_both_uncertain_ties_keep_primary(self):
        primary = self._r(False, 0.5)
        out = self._agent()._reconcile(primary, self._r(True, 0.5))
        assert out is primary


# ── verify() escalation behaviour ─────────────────────────────────────────────

class TestEscalation:
    def test_uncertain_click_escalates_and_vlm_rescues(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9))
        result = agent.verify(_step("click"), pre_hash=_PRE)
        assert client.query_vlm.called, "VLM should be consulted on uncertain click"
        assert result.success is True

    def test_uncertain_click_vlm_confirms_failure(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(False, 0.95))
        result = agent.verify(_step("click"), pre_hash=_PRE)
        assert client.query_vlm.called
        assert result.success is False
        assert result.confidence >= 0.75  # now a confident failure

    def test_confident_llm_does_not_escalate(self):
        agent, client = _make_agent(_verdict_json(True, 0.95), _verdict_json(False, 0.9))
        result = agent.verify(_step("click"), pre_hash=_PRE)
        assert not client.query_vlm.called, "confident verdict must not escalate"
        assert result.success is True

    def test_type_action_does_not_escalate(self):
        # type is a text action — the OCR path is appropriate; no VLM escalation.
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9))
        agent.verify(_step("type"))
        assert not client.query_vlm.called

    def test_escalation_disabled_flag(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9),
                                    escalate=False)
        agent.verify(_step("click"), pre_hash=_PRE)
        assert not client.query_vlm.called
