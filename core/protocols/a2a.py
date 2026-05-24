# core/protocols/a2a.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


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
    action_type: str     # click | double_click | type | key_press | hotkey | scroll | wait | screenshot
    target: Optional[str] = None    # Natural language UI element (for grounding)
    value: Optional[str] = None     # Text to type, or wait duration in seconds
    key: Optional[str] = None       # Key name for key_press / hotkey (e.g. "ctrl+s")
    description: str = ""
    verification: str = ""          # What to check to confirm success

