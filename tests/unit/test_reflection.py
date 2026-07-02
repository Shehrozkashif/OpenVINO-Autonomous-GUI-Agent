# tests/unit/test_reflection.py
"""Unit tests for agents/reflection.py — ReflectionAgent.

Organised by concern (each section was originally its own file):

  1. Prompt safety and raw confidence pass-through (no clamping)
  2. Perceptual-hash screen-delta check as ground truth for clicks
  3. Success condition in _execute_subtask (confident vs. uncertain verdicts)
  4. VLM screenshot escalation on uncertain visual verdicts
"""
import sys

sys.path.insert(0, ".")

from unittest.mock import MagicMock

import pytest
from PIL import Image

from agents.reflection import (
    _LLM_REFLECTION_PROMPT,
    _VLM_REFLECTION_PROMPT,
    ReflectionAgent,
    ReflectionResult,
)
from core.capture.screenshot import frame_phash
from core.protocols import ActionStep, SubTask

# ═══════════════════════════════════════════════════════════════════════════
# 1. Prompt safety and raw confidence pass-through
#
# Three properties are verified:
#
# PROMPT STANCE  — the LLM prompt no longer instructs the model to default to
#                 success on ambiguous cases; the new default is uncertain→fail.
# NO CLAMPING    — _parse() returns the raw confidence supplied by the LLM
#                 without forcing successes up to ≥0.75 or failures down to ≤0.60.
# INTERACTION    — a confident failure (conf=0.80) is reachable end-to-end because
#                 clamping no longer caps it at 0.60 before the threshold comparison.
# ═══════════════════════════════════════════════════════════════════════════

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


class TestNoConfidenceClamping:
    """_parse() must pass through the LLM's raw confidence unchanged.
    Old code clamped:
      success=True  → conf forced up   to ≥0.75
      success=False → conf forced down to ≤0.60
    """

    def _parse(self, json_str: str) -> ReflectionResult:
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


class TestInteractionWithFix01:
    """With clamping removed, the orchestrator's fail_threshold comparison
    (from Fix 0.1) can now actually detect confident failures.

    Before Fix 0.2: failure conf was always ≤0.60. With fail_threshold=0.75
    for clicks, 0.60 < 0.75 → treated as uncertain → retried but never fatal.

    After Fix 0.2: failure conf can be 0.85. With fail_threshold=0.75,
    0.85 >= 0.75 → confident failure → step fails immediately (no more retries
    unless should_retry=True).
    """

    def _parse(self, json_str: str) -> ReflectionResult:
        agent = ReflectionAgent.__new__(ReflectionAgent)
        agent._ocr = MagicMock()
        return agent._parse(json_str)

    def test_confident_failure_conf_above_click_threshold(self):
        """LLM says success=false, confidence=0.85.
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
        """LLM says success=false, confidence=0.5 (ambiguous case from new prompt).
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
        """When JSON parsing fails, the keyword-based fallback uses confidence=0.5
        for failures. This must not be changed by clamping.
        """
        result = self._parse("The action clearly did not succeed at all.")
        assert result.success is False
        assert result.confidence == pytest.approx(0.5), (
            "Fallback failure confidence must be 0.5, unchanged by removed clamping"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Perceptual-hash screen-delta check as ground truth for clicks
#
# If the perceptual hash of the screen does not change between the moment just
# after the click fires and the moment after the adaptive wait, the reflector
# returns an immediate failure (confidence=0.98) without calling the LLM or VLM.
#
# Three properties are verified:
#
# DELTA-ZERO     — identical before/after screenshots → immediate failure,
#                 LLM/VLM not called, confidence=0.98, should_retry=True.
# DELTA-NONZERO  — different before/after screenshots → LLM/VLM is called,
#                 no short-circuit.
# ACTION-SCOPING — delta check applies only to click/right_click/double_click,
#                 NOT to type/key_press/hotkey/scroll.
# ═══════════════════════════════════════════════════════════════════════════

def _step_delta(action_type="click", target="SomeButton", key=None, value=None):
    return ActionStep(
        id=1, subtask_id=1,
        action_type=action_type,
        target=target, value=value, key=key,
        description=f"{action_type} {target or key or value}",
        verification="expected result",
    )


def _solid(color=(128, 128, 128), size=(200, 100)):
    return Image.new("RGB", size, color)


def _make_agent_click(before_img, after_img):
    """Agent configured for a click path test.
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
    """Agent configured for a non-click path test.
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


class TestDeltaZero:
    """Identical before/after → immediate failure without LLM call."""

    def test_click_unchanged_returns_failure(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("click"))

        assert result.success is False

    def test_click_unchanged_confidence_is_098(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("click"))

        assert result.confidence == pytest.approx(0.98)

    def test_click_unchanged_should_retry_is_true(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("click"))

        assert result.should_retry is True

    def test_click_unchanged_llm_not_called(self):
        img = _solid((200, 200, 200))
        agent = _make_agent_click(before_img=img, after_img=img)

        agent.verify(_step_delta("click"))

        agent.client.query_llm.assert_not_called()
        agent.client.query_vlm.assert_not_called()

    def test_right_click_unchanged_returns_failure(self):
        img = _solid((50, 100, 150))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("right_click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.98)
        agent.client.query_llm.assert_not_called()

    def test_double_click_unchanged_returns_failure(self):
        img = _solid((0, 0, 0))
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("double_click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.98)

    def test_click_unchanged_observation_mentions_screen_unchanged(self):
        img = _solid()
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("click"))

        assert "unchanged" in result.observation.lower()

    def test_click_unchanged_recovery_hint_mentions_regrounding(self):
        img = _solid()
        agent = _make_agent_click(before_img=img, after_img=img)

        result = agent.verify(_step_delta("click"))

        assert "ground" in result.recovery_hint.lower()


class TestDeltaNonzero:
    """Different before/after → screen-delta check passes, LLM/VLM path runs."""

    def test_click_changed_calls_vlm(self):
        """Sparse OCR (0 words) + click → VLM path.
        LLM must NOT be called; VLM must be called.
        """
        before = _solid((0, 0, 0))
        after = _solid((255, 255, 255))
        agent = _make_agent_click(before_img=before, after_img=after)
        agent.client.query_vlm = MagicMock(return_value=MagicMock(
            content='{"success": true, "confidence": 0.85, "observation": "menu appeared"}'
        ))

        agent.verify(_step_delta("click"))

        agent.client.query_vlm.assert_called_once()
        agent.client.query_llm.assert_not_called()

    def test_click_changed_with_rich_ocr_calls_llm(self):
        """Rich OCR (≥3 words) + click → LLM path."""
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

        agent.verify(_step_delta("click"))

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

        result = agent.verify(_step_delta("click"))

        assert result.success is False
        assert result.confidence == pytest.approx(0.8)   # not 0.98


class TestActionScoping:
    """type, key_press, hotkey, scroll must not be subject to the delta check.
    - Only one capture() call (no before-hash capture).
    - Even with an "unchanged" screen, the LLM is still called.
    """

    def test_type_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("type", value="hello"))
        assert agent.capturer.capture.call_count == 1

    def test_key_press_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("key_press", key="winleft"))
        assert agent.capturer.capture.call_count == 1

    def test_hotkey_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("hotkey", key="ctrl+s"))
        assert agent.capturer.capture.call_count == 1

    def test_scroll_capture_called_once(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("scroll", value="down"))
        assert agent.capturer.capture.call_count == 1

    def test_type_calls_llm_regardless_of_screen(self):
        """Type always calls LLM (forced, regardless of screen change)."""
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("type", value="hello"))
        agent.client.query_llm.assert_called_once()

    def test_key_press_calls_llm_regardless_of_screen(self):
        agent = _make_agent_nonclick(_solid())
        agent.verify(_step_delta("key_press", key="enter"))
        agent.client.query_llm.assert_called_once()

    def test_type_result_not_098_confidence(self):
        """Type result must never be the delta-check sentinel confidence=0.98."""
        agent = _make_agent_nonclick(_solid())
        result = agent.verify(_step_delta("type", value="hello"))
        assert result.confidence != pytest.approx(0.98) or result.success is True


# ═══════════════════════════════════════════════════════════════════════════
# 3. Success condition in _execute_subtask
#
# Two invariants are guarded here:
#
#   1. A CONFIDENT failure (conf >= threshold) is never converted into success
#      (the original Fix 0.1 bug, where every step silently passed).
#
#   2. An UNCERTAIN verdict (conf < threshold) is handled by action class:
#        - idempotent  (click / scroll): retried — a dead click is caught
#          reliably by the screen-delta check, so retrying is safe and cheap.
#        - non-idempotent (type / Enter / Ctrl-V): ACCEPTED and handed to the
#          next planning step, because the action already fired and re-doing it
#          corrupts state (double-typing, double-submit). Correctness is
#          backstopped by _verify_command_effect / _verify_launch / loop-guard.
# ═══════════════════════════════════════════════════════════════════════════

def _make_subtask(description="click a button"):
    return SubTask(id=1, description=description, depends_on=[])


def _make_step(action_type="click", target="SomeButton"):
    return ActionStep(
        id=1, subtask_id=1,
        action_type=action_type,
        target=target, value=None, key=None,
        description=f"{action_type} {target}",
        verification="",
    )


def _make_reflection(success: bool, confidence: float, should_retry: bool = True):
    return ReflectionResult(
        success=success,
        confidence=confidence,
        observation="test observation",
        error_description="" if success else "test failure",
        should_retry=should_retry,
        recovery_hint="",
        ocr_text="some ocr text",
    )


def _make_orchestrator(reflection_results: list, step_override: ActionStep = None):
    """Build a TaskOrchestrator with all collaborators mocked.

    reflection_results: list of ReflectionResult objects returned by reflector.verify()
    in sequence (one per call).

    Config uses consecutive_failures_limit=1 so the subtask aborts after the
    first outer-loop planning step exhausts all its retries and still fails.
    This prevents the planner's second call (which returns None/"goal achieved")
    from masking a step failure — isolating Fix 0.1's logic cleanly.
    """
    from core.orchestrator import OrchestratorConfig, TaskOrchestrator

    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(side_effect=reflection_results)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    from agents.grounding import GroundingResult
    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(return_value=GroundingResult(
        found=True, confidence=0.9, x=100, y=200,
        latency_ms=5.0, target="SomeButton",
        element_type="foreground_interactive",
    ))

    step = step_override or _make_step()
    planner = MagicMock()
    # Return the step on the first planning call, then None.
    # With consecutive_failures_limit=1 the subtask aborts after the first
    # failing step — the None is never reached in failure-path tests.
    planner.plan_steps = MagicMock(side_effect=[[step], None])

    capturer = MagicMock()
    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=planner,
        grounder=grounder,
        actor=actor,
        reflector=reflector,
        capturer=capturer,
        task_memory=memory,
        config=OrchestratorConfig(
            max_retries_per_step=3,
            max_steps_per_subtask=5,
            # One outer-loop step failing all its retries → subtask fails immediately.
            # This isolates Fix 0.1 from the "planner says None after failure" issue.
            consecutive_failures_limit=1,
        ),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"SomeButton"')
    return orch


class TestFix01_SuccessCondition:
    """Core invariant: only reflection.success=True can produce step_success=True.
    Low-confidence failure must NOT be treated as success.
    """

    def test_success_true_high_conf_passes(self):
        """Baseline: reflector says success → step succeeds."""
        orch = _make_orchestrator([_make_reflection(success=True, confidence=0.9)])
        result = orch._execute_subtask(_make_subtask())
        assert result is True

    def test_failure_low_conf_does_not_pass_click(self):
        """Click step. Reflector says success=False, conf=0.60.
        Old code: 0.60 < 0.75 (min_confidence) → treated as success.
        New code: success=False → step fails → retried → subtask fails.
        """
        # Provide 3 identical failures (one per retry attempt).
        reflections = [_make_reflection(success=False, confidence=0.60)] * 3
        orch = _make_orchestrator(reflections)
        result = orch._execute_subtask(_make_subtask("click SomeButton"))
        assert result is False, (
            "A low-confidence failure on a click step must NOT produce subtask success"
        )

    def test_uncertain_type_accepts_without_retyping(self):
        """Type step, reflector UNCERTAIN (success=False, conf=0.60).

        A type action has already physically fired and an uncertain verdict is
        usually just "couldn't OCR-read the field", not real evidence of
        failure. Re-typing produces 'hellohello' corruption and loops, so the
        verdict is ACCEPTED and the next planning step verifies against the live
        screen. Critically, the action must NOT be re-executed: verify() — and
        therefore the action — is called exactly once.
        """
        step = _make_step(action_type="type", target=None)
        step.value = "hello"
        reflections = [_make_reflection(success=False, confidence=0.60)] * 3
        orch = _make_orchestrator(reflections, step_override=step)
        result = orch._execute_subtask(_make_subtask("type hello"))
        assert result is True, (
            "An uncertain type is accepted (next step re-checks live) — subtask "
            "should not fail outright"
        )
        assert orch.reflector.verify.call_count == 1, (
            "An uncertain type must be accepted on the first verdict, never "
            "re-typed (no double-typing)"
        )

    def test_uncertain_enter_accepts_without_resubmitting(self):
        """key_press Enter, reflector UNCERTAIN (success=False, conf=0.60).

        Enter is non-idempotent (it submits/launches). On an uncertain verdict
        it is accepted rather than retried, so the agent never double-submits.
        Launch/command correctness is still backstopped by _verify_launch and
        the on-disk command check at the subtask level.
        """
        step = _make_step(action_type="key_press", target=None)
        step.key = "enter"
        step.description = "press enter to launch app"
        reflections = [_make_reflection(success=False, confidence=0.60)] * 3
        orch = _make_orchestrator(reflections, step_override=step)
        result = orch._execute_subtask(_make_subtask("press enter"))
        assert result is True
        assert orch.reflector.verify.call_count == 1, (
            "An uncertain Enter must be accepted on the first verdict, never "
            "re-submitted"
        )

    def test_failure_high_conf_fails(self):
        """Reflector says success=False, conf=0.80 (above min_confidence=0.75).
        This is a confident failure — must not pass.
        """
        reflections = [
            _make_reflection(success=False, confidence=0.80, should_retry=True),
            _make_reflection(success=False, confidence=0.80, should_retry=True),
            _make_reflection(success=False, confidence=0.80, should_retry=True),
        ]
        orch = _make_orchestrator(reflections)
        result = orch._execute_subtask(_make_subtask())
        assert result is False

    def test_failure_high_conf_no_retry_stops_immediately(self):
        """Confident failure + should_retry=False → break out of retry loop immediately.
        Reflector is called exactly once (not retried).
        """
        reflections = [_make_reflection(success=False, confidence=0.80, should_retry=False)]
        orch = _make_orchestrator(reflections)
        result = orch._execute_subtask(_make_subtask())
        assert result is False
        assert orch.reflector.verify.call_count == 1, (
            "should_retry=False with confident failure must not trigger retries"
        )

    def test_uncertain_failure_retries_before_giving_up(self):
        """Low-confidence failure (uncertain) → the system retries max_retries_per_step
        times, then fails. Reflector is called exactly max_retries times.
        """
        n_retries = 3
        reflections = [_make_reflection(success=False, confidence=0.50)] * n_retries
        orch = _make_orchestrator(reflections)
        result = orch._execute_subtask(_make_subtask())
        assert result is False
        assert orch.reflector.verify.call_count == n_retries, (
            f"Uncertain failure must trigger exactly {n_retries} retries"
        )

    def test_success_after_retry_passes(self):
        """First attempt fails (uncertain), second attempt succeeds.
        Subtask must succeed overall.
        """
        reflections = [
            _make_reflection(success=False, confidence=0.50),
            _make_reflection(success=True, confidence=0.90),
        ]
        orch = _make_orchestrator(reflections)
        result = orch._execute_subtask(_make_subtask())
        assert result is True, "Success on retry must produce overall subtask success"
        assert orch.reflector.verify.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. VLM screenshot escalation on uncertain visual verdicts
#
# The OCR→LLM path is blind to visual-only changes (selection highlight, toggle,
# icon state). When it is uncertain about a *visual* action, the reflector takes a
# second look with the VLM on the real screenshot and reconciles the two verdicts.
# ═══════════════════════════════════════════════════════════════════════════

def _step_vlm(action_type="click", target="Item"):
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


class TestEscalation:
    def test_uncertain_click_escalates_and_vlm_rescues(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9))
        result = agent.verify(_step_vlm("click"), pre_hash=_PRE)
        assert client.query_vlm.called, "VLM should be consulted on uncertain click"
        assert result.success is True

    def test_uncertain_click_vlm_confirms_failure(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(False, 0.95))
        result = agent.verify(_step_vlm("click"), pre_hash=_PRE)
        assert client.query_vlm.called
        assert result.success is False
        assert result.confidence >= 0.75  # now a confident failure

    def test_confident_llm_does_not_escalate(self):
        agent, client = _make_agent(_verdict_json(True, 0.95), _verdict_json(False, 0.9))
        result = agent.verify(_step_vlm("click"), pre_hash=_PRE)
        assert not client.query_vlm.called, "confident verdict must not escalate"
        assert result.success is True

    def test_type_action_does_not_escalate(self):
        # type is a text action — the OCR path is appropriate; no VLM escalation.
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9))
        agent.verify(_step_vlm("type"))
        assert not client.query_vlm.called

    def test_escalation_disabled_flag(self):
        agent, client = _make_agent(_verdict_json(False, 0.5), _verdict_json(True, 0.9),
                                    escalate=False)
        agent.verify(_step_vlm("click"), pre_hash=_PRE)
        assert not client.query_vlm.called
