# core/types.py
"""The two data models every layer passes around.

`SubTask` is what the Router produces; `ActionStep` is what the Planner
produces and the Actor executes. Nothing here imports a backend, a UI or a
Windows API, so every other module can depend on it freely.
"""
from pydantic import BaseModel


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
    # click | double_click | right_click | type | key_press | hotkey |
    # scroll | wait | extract | set_value | select | invoke
    action_type: str
    target: str | None = None    # natural-language UI element description (for grounding)
    value: str | None = None     # text to type, scroll direction, or wait duration in seconds
    key: str | None = None       # key name for key_press / hotkey e.g. "ctrl+s"
    description: str = ""
    verification: str = ""          # what to observe to confirm success
