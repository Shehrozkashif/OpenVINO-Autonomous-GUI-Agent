# core/protocols/a2a.py
"""
Shared data models and the InferenceClient Protocol.

InferenceClient is the interface every backend (OVMSClient, OllamaClient,
DirectOpenVINOClient) satisfies — agents depend on this Protocol, not on
any concrete class.
"""
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from core.pipeline.ovms_client import OVMSResponse


# ── Agent communication enums / models ───────────────────────────────────────

class AgentRole(str, Enum):
    ROUTER = "router"
    PLANNING = "planning"
    GROUNDING = "grounding"
    ACTION = "action"
    REFLECTION = "reflection"


class A2AMessage(BaseModel):
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str       # "task" | "result" | "error" | "status"
    payload: dict
    task_id: str
    step_id: Optional[int] = None


class TaskRequest(BaseModel):
    instruction: str
    task_id: str
    context: Optional[dict] = None


class SubTask(BaseModel):
    id: int
    description: str
    depends_on: List[int] = []


class ActionStep(BaseModel):
    id: int
    subtask_id: int
    action_type: str     # click | double_click | right_click | type | key_press | hotkey | scroll | wait | screenshot
    target: Optional[str] = None    # natural-language UI element description (for grounding)
    value: Optional[str] = None     # text to type, scroll direction, or wait duration in seconds
    key: Optional[str] = None       # key name for key_press / hotkey e.g. "ctrl+s"
    description: str = ""
    verification: str = ""          # what to observe to confirm success


# ── Inference client Protocol ─────────────────────────────────────────────────

@runtime_checkable
class InferenceClient(Protocol):
    """
    Structural interface satisfied by OVMSClient, OllamaClient, and
    DirectOpenVINOClient.  Agents type-hint against this Protocol so they
    remain decoupled from any specific backend.
    """

    def query_llm(
        self,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> OVMSResponse: ...

    def query_vlm(
        self,
        prompt: str,
        image_base64: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> OVMSResponse: ...

    def check_health(self) -> dict: ...
