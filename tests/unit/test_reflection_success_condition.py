# tests/unit/test_reflection_success_condition.py
"""
Unit tests for Fix 0.1 — the broken success condition in _execute_subtask.

Regression guard: ensures that a low-confidence failure from the reflector
can never be silently converted into step_success=True.

Each test constructs a minimal orchestrator with mocked collaborators, runs
_execute_subtask with a controlled reflection result, and asserts the outcome.
"""
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from core.protocols.a2a import ActionStep, SubTask
from agents.reflection.reflection_agent import ReflectionResult


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """
    Build a TaskOrchestrator with all collaborators mocked.

    reflection_results: list of ReflectionResult objects returned by reflector.verify()
    in sequence (one per call).

    Config uses consecutive_failures_limit=1 so the subtask aborts after the
    first outer-loop planning step exhausts all its retries and still fails.
    This prevents the planner's second call (which returns None/"goal achieved")
    from masking a step failure — isolating Fix 0.1's logic cleanly.
    """
    from core.orchestrator import TaskOrchestrator, OrchestratorConfig

    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(side_effect=reflection_results)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    from agents.grounding.grounding_agent import GroundingResult
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
    planner.plan_next_step = MagicMock(side_effect=[step, None])

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


# ── tests ─────────────────────────────────────────────────────────────────────

class TestFix01_SuccessCondition:
    """
    Core invariant: only reflection.success=True can produce step_success=True.
    Low-confidence failure must NOT be treated as success.
    """

    def test_success_true_high_conf_passes(self):
        """Baseline: reflector says success → step succeeds."""
        orch = _make_orchestrator([_make_reflection(success=True, confidence=0.9)])
        result = orch._execute_subtask(_make_subtask())
        assert result is True

    def test_failure_low_conf_does_not_pass_click(self):
        """
        Click step. Reflector says success=False, conf=0.60.
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

    def test_failure_low_conf_does_not_pass_type(self):
        """
        Type step. Reflector says success=False, conf=0.60.
        Old code: 0.60 < 0.95 (type fail_threshold) → treated as success.
        New code: success=False → step fails → retried → subtask fails.
        """
        step = _make_step(action_type="type", target=None)
        step.value = "hello"
        reflections = [_make_reflection(success=False, confidence=0.60)] * 3
        orch = _make_orchestrator(reflections, step_override=step)
        result = orch._execute_subtask(_make_subtask("type hello"))
        assert result is False, (
            "A low-confidence failure on a type step must NOT produce subtask success"
        )

    def test_failure_low_conf_does_not_pass_enter_key(self):
        """
        key_press enter (launcher key). fail_threshold=0.95.
        Old code: conf=0.60 < 0.95 → success.
        New code: success=False → step fails.
        """
        step = _make_step(action_type="key_press", target=None)
        step.key = "enter"
        step.description = "press enter to launch app"
        reflections = [_make_reflection(success=False, confidence=0.60)] * 3
        orch = _make_orchestrator(reflections, step_override=step)
        result = orch._execute_subtask(_make_subtask("press enter"))
        assert result is False

    def test_failure_high_conf_fails(self):
        """
        Reflector says success=False, conf=0.80 (above min_confidence=0.75).
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
        """
        Confident failure + should_retry=False → break out of retry loop immediately.
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
        """
        Low-confidence failure (uncertain) → the system retries max_retries_per_step
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
        """
        First attempt fails (uncertain), second attempt succeeds.
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
