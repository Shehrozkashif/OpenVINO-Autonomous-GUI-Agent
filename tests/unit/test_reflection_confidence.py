# tests/unit/test_reflection_confidence.py
"""
Unit tests for reflection prompt safety and raw confidence pass-through.

Three properties are verified:

1. PROMPT STANCE  — the LLM prompt no longer instructs the model to default to
                    success on ambiguous cases; the new default is uncertain→fail.

2. NO CLAMPING    — _parse() returns the raw confidence supplied by the LLM
                    without forcing successes up to ≥0.75 or failures down to ≤0.60.

3. INTERACTION    — a confident failure (conf=0.80) is reachable end-to-end because
                    clamping no longer caps it at 0.60 before the threshold comparison.
"""
import sys
sys.path.insert(0, ".")

import pytest
from agents.reflection.reflection_agent import (
    ReflectionAgent,
    ReflectionResult,
    _LLM_REFLECTION_PROMPT,
    _VLM_REFLECTION_PROMPT,
)


# ── 1. Prompt-stance tests (static string checks) ────────────────────────────

class TestPromptStance:
    """The prompts must not bias toward success on uncertain outcomes."""

    def test_llm_prompt_no_lenient_instruction(self):
        assert "lenient" not in _LLM_REFLECTION_PROMPT.lower(), (
            "LLM prompt must not tell the model to be lenient"
        )

    def test_llm_prompt_no_ambiguous_success(self):
        assert "ambiguous case → success=true" not in _LLM_REFLECTION_PROMPT.lower(), (
            "Old 'ambiguous case → success=true' instruction must be removed"
        )
        assert "any ambiguous" not in _LLM_REFLECTION_PROMPT.lower()

    def test_llm_prompt_ambiguous_defaults_to_false(self):
        assert "success=false" in _LLM_REFLECTION_PROMPT.lower(), (
            "LLM prompt must instruct the model to return success=false for ambiguous cases"
        )

    def test_vlm_prompt_no_mid_range_success(self):
        """The old line 'When confidence is between 0.5 and 0.8, return success=true' is gone."""
        assert "0.5 and 0.8" not in _VLM_REFLECTION_PROMPT, (
            "VLM prompt must not hardcode success=true for mid-range confidence"
        )
        assert "return success=true" not in _VLM_REFLECTION_PROMPT.lower()

    def test_vlm_prompt_uncertain_defaults_to_false(self):
        assert "success=false" in _VLM_REFLECTION_PROMPT.lower()


# ── 2. Confidence-clamping tests (unit tests on _parse) ──────────────────────

class TestNoConfidenceClamping:
    """
    _parse() must pass through the LLM's raw confidence unchanged.
    Old code clamped:
      success=True  → conf forced up   to ≥0.75
      success=False → conf forced down to ≤0.60
    """

    def _parse(self, json_str: str) -> ReflectionResult:
        from unittest.mock import MagicMock
        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent._ocr = MagicMock()
        return agent._parse(json_str)

    # success=True cases — low confidence must NOT be raised to 0.75
    def test_success_low_conf_preserved(self):
        result = self._parse('{"success": true, "confidence": 0.3, "observation": "ok"}')
        assert result.success is True
        assert result.confidence == pytest.approx(0.3), (
            f"Expected 0.3, got {result.confidence}. "
            "Old code would have clamped this up to 0.75."
        )

    def test_success_zero_conf_preserved(self):
        result = self._parse('{"success": true, "confidence": 0.1, "observation": "ok"}')
        assert result.success is True
        assert result.confidence == pytest.approx(0.1)

    def test_success_normal_conf_preserved(self):
        result = self._parse('{"success": true, "confidence": 0.9, "observation": "ok"}')
        assert result.success is True
        assert result.confidence == pytest.approx(0.9)

    # success=False cases — high confidence must NOT be lowered to 0.60
    def test_failure_high_conf_preserved(self):
        result = self._parse(
            '{"success": false, "confidence": 0.85, "observation": "nothing happened"}'
        )
        assert result.success is False
        assert result.confidence == pytest.approx(0.85), (
            f"Expected 0.85, got {result.confidence}. "
            "Old code would have clamped this down to 0.60."
        )

    def test_failure_max_conf_preserved(self):
        result = self._parse(
            '{"success": false, "confidence": 1.0, "observation": "definitively failed"}'
        )
        assert result.success is False
        assert result.confidence == pytest.approx(1.0)

    def test_failure_low_conf_preserved(self):
        """Uncertain failure — confidence 0.5 must stay at 0.5, not be clamped."""
        result = self._parse(
            '{"success": false, "confidence": 0.5, "observation": "uncertain"}'
        )
        assert result.success is False
        assert result.confidence == pytest.approx(0.5)

    def test_failure_mid_conf_preserved(self):
        """Failure at 0.72 — old code clamped to 0.60, new code keeps 0.72."""
        result = self._parse(
            '{"success": false, "confidence": 0.72, "observation": "failed"}'
        )
        assert result.success is False
        assert result.confidence == pytest.approx(0.72)


# ── 3. Interaction with Fix 0.1 — confident failures now reachable ────────────

class TestInteractionWithFix01:
    """
    With clamping removed, the orchestrator's fail_threshold comparison
    (from Fix 0.1) can now actually detect confident failures.

    Before Fix 0.2: failure conf was always ≤0.60. With fail_threshold=0.75
    for clicks, 0.60 < 0.75 → treated as uncertain → retried but never fatal.

    After Fix 0.2: failure conf can be 0.85. With fail_threshold=0.75,
    0.85 >= 0.75 → confident failure → step fails immediately (no more retries
    unless should_retry=True).
    """

    def _parse(self, json_str: str) -> ReflectionResult:
        from unittest.mock import MagicMock
        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent._ocr = MagicMock()
        return agent._parse(json_str)

    def test_confident_failure_conf_above_click_threshold(self):
        """
        LLM says success=false, confidence=0.85.
        With Fix 0.2: conf=0.85 (not clamped to 0.60).
        In Fix 0.1 orchestrator: 0.85 >= min_confidence(0.75) → confident failure.
        """
        result = self._parse(
            '{"success": false, "confidence": 0.85, '
            '"observation": "no menu appeared", "should_retry": false}'
        )
        assert result.success is False
        assert result.confidence == pytest.approx(0.85)
        # Orchestrator uses: conf >= fail_threshold(0.75) → confident failure
        assert result.confidence >= 0.75, (
            "Confident failure must be detectable by the orchestrator's threshold"
        )

    def test_uncertain_failure_still_triggers_retry(self):
        """
        LLM says success=false, confidence=0.5 (ambiguous case from new prompt).
        Fix 0.1 orchestrator: 0.5 < fail_threshold(0.75) → uncertain → retry.
        """
        result = self._parse(
            '{"success": false, "confidence": 0.5, '
            '"observation": "unclear", "should_retry": true}'
        )
        assert result.success is False
        assert result.confidence < 0.75, (
            "Uncertain failure must still fall below the threshold, triggering retries"
        )
        assert result.should_retry is True

    def test_should_retry_field_preserved(self):
        """should_retry from the LLM is forwarded unchanged."""
        result = self._parse(
            '{"success": false, "confidence": 0.9, '
            '"observation": "click had no effect", "should_retry": false}'
        )
        assert result.should_retry is False

    def test_fallback_parse_failure_conf_unchanged(self):
        """
        When JSON parsing fails, the keyword-based fallback uses confidence=0.5
        for failures. This must not be changed by clamping.
        """
        result = self._parse("The action clearly did not succeed at all.")
        assert result.success is False
        assert result.confidence == pytest.approx(0.5), (
            "Fallback failure confidence must be 0.5, unchanged by removed clamping"
        )
