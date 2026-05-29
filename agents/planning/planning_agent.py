# agents/planning/planning_agent.py
"""
Planning Agent — generates precise step sequences for sub-tasks.
Uses Chain-of-Thought prompting for complex multi-step tasks.
"""

import json
import platform
import re
from typing import List

from loguru import logger

from core.protocols.a2a import ActionStep, InferenceClient, SubTask

_OS = platform.system()   # "Windows", "Linux", or "Darwin"
if _OS == "Windows":
    _OS_CONTEXT = "Microsoft Windows 11 desktop"
    _LAUNCHER_KEY = "winleft"
    _LAUNCHER_NAME = "Windows Start menu"
    _TERM_NAME = "PowerShell or Command Prompt"
elif _OS == "Darwin":
    _OS_CONTEXT = "macOS desktop"
    _LAUNCHER_KEY = "command+space"
    _LAUNCHER_NAME = "Spotlight search"
    _TERM_NAME = "Terminal"
else:
    _OS_CONTEXT = "Linux desktop (Ubuntu/GNOME)"
    _LAUNCHER_KEY = "super"
    _LAUNCHER_NAME = "GNOME Activities search"
    _TERM_NAME = "Terminal"

PLANNING_SYSTEM_PROMPT = f"""You are a desktop automation planner running on {_OS_CONTEXT}.
Generate the minimum steps needed to complete a task using mouse and keyboard.

Available action_types and their required fields (set all others to null):
- click / right_click / double_click  →  "target": text visible on screen or clear description
- type                                 →  "value": the text to type
- key_press                            →  "key": key name e.g. "enter", "escape", "tab"
- hotkey                               →  "key": combo e.g. "ctrl+s", "alt+f4"
- scroll                               →  "target": element, "value": "up" or "down"
- wait                                 →  "value": seconds as string e.g. "1.5"

CRITICAL — writing the "target" field:
When the user message includes "Text currently visible on screen", use those EXACT words as click targets.
A target that matches visible text is found instantly. A target invented from memory may be wrong.
Example: if screen shows "Compose" and you need to click the compose button → target = "Compose button"

Output ONLY a valid JSON array. "id" must be an integer. All fields required, use null for unused.
[
  {{"id": 1, "action_type": "click", "target": "...", "value": null, "key": null, "description": "...", "verification": ""}},
  ...
]"""


class PlanningAgent:
    def __init__(self, ovms_client: InferenceClient):
        self.ovms = ovms_client

    def plan(self, subtask: SubTask, context: dict = None,
             screen_context: str = None) -> List[ActionStep]:
        logger.info(f"[PLANNING] Planning: '{subtask.description}'")
        user_content = f"Generate steps for: {subtask.description}"
        if screen_context:
            user_content += (
                f"\n\nText currently visible on screen (use these exact labels as "
                f"click targets when relevant): {screen_context}"
            )
        if context:
            user_content += f"\nAdditional context: {json.dumps(context)}"
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3)
        steps = self._parse_steps(resp.content, subtask.id)
        logger.info(f"[PLANNING] {len(steps)} steps generated")
        for s in steps:
            logger.info(f"  [{s.id}] {s.action_type}: {s.description}")
        return steps

    def replan(
        self, failed_step: ActionStep, error: str, remaining: List[ActionStep]
    ) -> List[ActionStep]:
        logger.warning(f"[PLANNING] Replanning after step {failed_step.id} failure")
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Step {failed_step.id} ('{failed_step.description}') failed.\n"
                f"Error: {error}\n"
                f"Remaining planned steps: {[s.description for s in remaining]}\n\n"
                f"Generate a corrected sequence to recover and continue.",
            },
        ]
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3)
        return self._parse_steps(resp.content, failed_step.subtask_id)

    def _parse_steps(self, text: str, subtask_id: int) -> List[ActionStep]:
        # Remove reasoning block if </think> is present
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
            raise ValueError(f"No JSON array in planning response: {text[:200]}")

        json_str = text[start_idx:end_idx+1]
        # Fix trailing commas common in LLM output
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[PLANNING] JSON parse error: {e}\nRaw string: {json_str}")
            raise
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
