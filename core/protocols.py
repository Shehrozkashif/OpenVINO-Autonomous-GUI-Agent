# core/protocols.py
"""Shared data models and the InferenceClient Protocol."""
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from core.ovms_client import InferenceResponse

# ── Task decomposition models ─────────────────────────────────────────────────

class SubTask(BaseModel):
    """One unit of work the Router splits an instruction into (e.g. "open Teams").

    The orchestrator runs subtasks in dependency order; each is planned one step
    at a time against the live screen until its goal is satisfied.
    """

    id: int
    description: str
    depends_on: list[int] = []              # ids that must finish before this one


class ActionStep(BaseModel):
    """A single atomic action the planner emits (one click, one keypress, …).

    Only the fields relevant to `action_type` are set; the rest stay None.
    """

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
    ) -> InferenceResponse:
        """Chat-complete against the text model (routing, planning, verification).

        response_schema, when given, constrains the reply to valid JSON.
        """
        ...

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
        system_prompt: str = None,
    ) -> InferenceResponse:
        """Ask the vision model about one screenshot (grounding, visual verify)."""
        ...

    def check_health(self) -> dict:
        """Return per-model readiness, e.g. {"qwen3-8b-int4-ov": "OK", ...}."""
        ...
