# agents/planning/planning_agent.py
"""
Planning Agent — generates precise step sequences for sub-tasks.
Uses Chain-of-Thought prompting for complex multi-step tasks.
"""
import json
import re
from typing import List

from loguru import logger

from core.pipeline.ovms_client import OVMSClient
from core.protocols.a2a import ActionStep, SubTask

PLANNING_SYSTEM_PROMPT = """You are a desktop automation planner.
Generate precise atomic steps for a given sub-task.

Available action_types:
- click: click a UI element (requires target)
- double_click: open a file/folder (requires target)
- type: type text (requires value)
- key_press: press a key (requires key, e.g. "enter", "escape", "f5")
- hotkey: key combination (requires key, e.g. "ctrl+s", "ctrl+shift+n")
- scroll: scroll wheel (requires target + direction in value: "up" or "down")
- wait: pause execution (requires value: seconds as string, e.g. "1.5")
- screenshot: capture screen state

Chain-of-Thought process:
1. What is the starting state?
2. What exact UI elements need interaction?
3. Correct order?
4. What verifies each step?

Output ONLY a JSON array, nothing else:
[
  {
    "id": 1,
    "action_type": "click",
    "target": "the File menu at the top left",
    "value": null,
    "key": null,
    "description": "Click File menu to open it",
    "verification": "File dropdown menu appears"
  }
]"""


class PlanningAgent:
    def __init__(self, ovms_client: OVMSClient):
        self.ovms = ovms_client

    def plan(self, subtask: SubTask, context: dict = None) -> List[ActionStep]:
        logger.info(f"[PLANNING] Planning: '{subtask.description}'")
        ctx = f"\nCurrent context: {json.dumps(context)}" if context else ""
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content":
             f"Generate steps for: {subtask.description}{ctx}"}
        ]
        resp = self.ovms.query_llm(messages, max_tokens=1024, temperature=0.3)
        steps = self._parse_steps(resp.content, subtask.id)
        logger.info(f"[PLANNING] {len(steps)} steps generated")
        for s in steps:
            logger.info(f"  [{s.id}] {s.action_type}: {s.description}")
        return steps

    def replan(self, failed_step: ActionStep, error: str,
               remaining: List[ActionStep]) -> List[ActionStep]:
        logger.warning(f"[PLANNING] Replanning after step {failed_step.id} failure")
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content":
             f"Step {failed_step.id} ('{failed_step.description}') failed.\n"
             f"Error: {error}\n"
             f"Remaining planned steps: {[s.description for s in remaining]}\n\n"
             f"Generate a corrected sequence to recover and continue."}
        ]
        resp = self.ovms.query_llm(messages, max_tokens=512, temperature=0.3)
        return self._parse_steps(resp.content, failed_step.subtask_id)

    def _parse_steps(self, text: str, subtask_id: int) -> List[ActionStep]:
        arr_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not arr_match:
            raise ValueError(f"No JSON array in planning response: {text[:200]}")
        data = json.loads(arr_match.group())
        steps = []
        for item in data:
            item["subtask_id"] = subtask_id
            item.setdefault("target", None)
            item.setdefault("value", None)
            item.setdefault("key", None)
            item.setdefault("description", "")
            item.setdefault("verification", "")
            steps.append(ActionStep(**item))
        return steps
