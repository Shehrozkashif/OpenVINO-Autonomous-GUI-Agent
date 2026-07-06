# core/protocols.py
"""Shared data models and the InferenceClient Protocol."""
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from core.ovms_client import InferenceResponse

# ── Task decomposition models ─────────────────────────────────────────────────

class SubTask(BaseModel):
    id: int
    description: str
    depends_on: list[int] = []
    burst: Any | None = Field(default=None, exclude=True)  # ActionBurst — excluded from model_dump (Fix C)


class ActionStep(BaseModel):
    id: int
    subtask_id: int
    # click | double_click | right_click | type | key_press | hotkey | scroll |
    # wait | screenshot | extract | drag |
    # set_value | select | invoke   (UIA structured-control actions)
    action_type: str
    target: str | None = None    # natural-language UI element description (for grounding)
    value: str | None = None     # text to type, scroll direction, or wait duration in seconds
    key: str | None = None       # key name for key_press / hotkey e.g. "ctrl+s"
    description: str = ""
    verification: str = ""          # what to observe to confirm success


# ── Burst execution primitives ────────────────────────────────────────────────

@dataclass
class ActionBurst:
    """A sequence of 2–5 actions executed without intermediate LLM calls.

    All targets are pre-grounded before step 0 fires.  If any grounding fails
    the burst is aborted and the orchestrator falls back to the planning loop.
    """

    steps: list[ActionStep]
    verify_at_end: bool = True              # run ONE reflection on the final step only
    timeout_ms: int = 5000                  # abort if the whole burst exceeds this
    rollback_steps: list[ActionStep] = field(default_factory=list)   # optional recovery


@dataclass
class BurstResult:
    """Outcome returned by BurstExecutor.run()."""

    success: bool
    failed_at_step: int | None           # None on success
    reason: str                             # human-readable summary


# ── Inference client Protocol ─────────────────────────────────────────────────

@runtime_checkable
class InferenceClient(Protocol):
    """Interface that OVMSClient satisfies. Agents type-hint against this."""

    def query_llm(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_schema: dict = None,
    ) -> InferenceResponse: ...

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        system_prompt: str = None,
    ) -> InferenceResponse: ...

    def check_health(self) -> dict: ...
