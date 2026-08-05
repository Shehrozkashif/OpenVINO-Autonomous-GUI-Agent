# tests/unit/conftest.py
"""Shared test doubles.

Every orchestrator test needs the same four fakes — an LLM that answers goal
checks, a grounder that finds things, a reflector with a verdict, and an actor
that succeeds. Building them ad hoc in each test file is how the suite drifted
out of sync with the code: a bare MagicMock grounder returns a MagicMock from
`min_confidence`, and comparing that to a float raises TypeError inside the
orchestrator — a crash that looks like a real bug but is only a bad double.

Build doubles from these helpers so a change in an agent's contract is fixed in
one place.
"""
import json
from unittest.mock import MagicMock

from agents.grounding import GroundingResult
from agents.reflection import ReflectionResult


def llm_reply(content: str) -> MagicMock:
    """One inference response with the given text content."""
    return MagicMock(content=content)


def goal_check_reply(satisfied: bool = True, confidence: float = 0.95,
                     evidence: str = "the goal state is visible") -> MagicMock:
    """A reply in the JSON shape `_goal_already_satisfied` parses."""
    return llm_reply(json.dumps({
        "satisfied": satisfied, "confidence": confidence, "evidence": evidence,
    }))


def make_llm(default_reply: MagicMock | None = None) -> MagicMock:
    """An inference client whose LLM answers every goal check with "satisfied".

    Needed because a subtask only ends when the planner returns no steps AND
    the goal check confirms it on screen — a fake that cannot answer that
    question leaves the loop escalating to the visual planner forever.
    """
    client = MagicMock()
    client.query_llm = MagicMock(return_value=default_reply or goal_check_reply())
    client.query_vlm = MagicMock(return_value=llm_reply(""))
    return client


def make_grounder(result: GroundingResult | None = None,
                  min_confidence: float = 0.5) -> MagicMock:
    """A grounder that finds its target at (100, 200) unless told otherwise."""
    found = result if result is not None else GroundingResult(
        found=True, confidence=0.9, x=100, y=200, latency_ms=5.0,
        target="Button", method="uia", element_type="foreground_interactive",
    )
    grounder = MagicMock()
    grounder.min_confidence = min_confidence      # a real float: the code compares it
    grounder.ground = MagicMock(return_value=found)
    grounder.ground_fast = MagicMock(return_value=found)
    grounder.is_dead_point = MagicMock(return_value=False)
    grounder.mark_dead = MagicMock()
    grounder.clear_dead_points = MagicMock()
    return grounder


def make_reflector(verdicts=None, client: MagicMock | None = None) -> MagicMock:
    """A verifier returning `verdicts` in order, then repeating the last one."""
    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.client = client or make_llm()
    if verdicts is None:
        verdicts = [success_verdict()]
    if isinstance(verdicts, ReflectionResult):
        verdicts = [verdicts]
    seq = list(verdicts)
    reflector.verify = MagicMock(side_effect=lambda *a, **k: seq.pop(0) if len(seq) > 1 else seq[0])
    return reflector


def success_verdict(confidence: float = 0.95) -> ReflectionResult:
    return ReflectionResult(
        success=True, confidence=confidence, observation="ok",
        error_description="", should_retry=False, recovery_hint="", ocr_text="",
    )


def failure_verdict(confidence: float = 0.95, error: str = "no change") -> ReflectionResult:
    return ReflectionResult(
        success=False, confidence=confidence, observation="nothing happened",
        error_description=error, should_retry=True, recovery_hint="", ocr_text="",
    )


def uncertain_verdict(confidence: float = 0.5) -> ReflectionResult:
    return ReflectionResult(
        success=False, confidence=confidence, observation="unclear",
        error_description="", should_retry=True, recovery_hint="", ocr_text="",
    )


def make_history() -> MagicMock:
    """A TaskHistory double. The loop only ever writes to it."""
    history = MagicMock()
    history.store_successful_task = MagicMock()
    history.get_recent_tasks = MagicMock(return_value=[])
    return history
