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

_STEP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":          {"type": "integer"},
            "action_type": {"type": "string", "enum": [
                "click", "right_click", "double_click",
                "type", "key_press", "hotkey", "scroll", "wait",
            ]},
            "target":      {"type": ["string", "null"],
                            "description": "UI element description for click/scroll (null otherwise)"},
            "value":       {"type": ["string", "null"],
                            "description": "Text to type, 'up'/'down' for scroll, seconds for wait (null otherwise)"},
            "key":         {"type": ["string", "null"],
                            "description": "REQUIRED for key_press and hotkey. Key name e.g. 'enter', 'escape', 'super', or combo e.g. 'ctrl+s', 'ctrl+alt+t'. null for other actions."},
            "description":  {"type": "string"},
            "verification": {"type": "string"},
        },
        "required": ["id", "action_type", "target", "value", "key", "description", "verification"],
    },
}

PLANNING_SYSTEM_PROMPT = f"""You are a desktop automation planner running on {_OS_CONTEXT}.
Generate the minimum steps needed to complete a task using mouse and keyboard.

Available action_types and their required fields (set all others to null):
- click / right_click / double_click  →  "target": exact visible text or clear element name
- type                                 →  "value": the text to type
- key_press                            →  "key": key name e.g. "enter", "escape", "super", "tab"
- hotkey                               →  "key": combo e.g. "ctrl+s", "ctrl+alt+t", "alt+f4"
- scroll                               →  "target": element, "value": "up" or "down"
- wait                                 →  "value": seconds as string e.g. "1.5"

LAUNCHING APPLICATIONS — only if the app is NOT already open.
Use this EXACT 4-step pattern every time:
  Step 1: key_press  key="super"              (opens {_LAUNCHER_NAME})
  Step 2: click      target="Type to search"  (click the search bar to lock keyboard focus)
  Step 3: type       value="<app name>"       (searches for the app)
  Step 4: key_press  key="enter"              (launches the app)

The click on "Type to search" is MANDATORY — without it the typed text misses the search bar.

If the task description says "in <app>" and the "currently visible" context already
shows that app is open (e.g. shows "Downloads", "Firefox", "gedit" etc.), do NOT
re-launch. Start directly with the interaction (click, type, etc.).

CLICKING — the golden rule:
If the target text appears EXACTLY in the "currently visible on screen" list → generate
ONE single step: click with that exact string as the target. No navigation steps before it.
Example: visible shows "Downloads" and task is "click Downloads" → just one step: click "Downloads".
Only add navigation steps (Super search, etc.) if the target is NOT in the visible list.

IGNORE UNRELATED WINDOWS: focus ONLY on the app needed for the current task.
If a file manager, terminal, or other unrelated window is visible on screen, ignore it completely.
Never click on unrelated windows or their contents.

WEB NAVIGATION (typing URLs or searches in a browser):
  Use hotkey key="ctrl+l" to focus the address bar — do NOT click on the address bar visually.
  Pattern: hotkey ctrl+l → type <URL> → key_press enter

SCREENSHOTS — use: hotkey key="print_screen"

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
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3,
                                   response_schema=_STEP_SCHEMA)
        try:
            steps = self._parse_steps(resp.content, subtask.id)
        except (ValueError, json.JSONDecodeError) as first_err:
            logger.warning(f"[PLANNING] Parse failed ({first_err}) — retrying")
            retry_messages = [
                {"role": "system", "content": (
                    "Output ONLY a JSON array of desktop action steps.\n"
                    "RULES:\n"
                    "- click/right_click/double_click: 'target' must be the visible text or element name (not null)\n"
                    "- type: 'value' must be the text to type (not null)\n"
                    "- key_press/hotkey: 'key' must be the key name or combo e.g. 'super', 'ctrl+alt+t' (not null)\n"
                    "- scroll: 'target' is the element to scroll on, 'value' is 'up' or 'down'\n"
                    "- All other fields: null"
                )},
                {"role": "user", "content": f"Steps to accomplish: {subtask.description}"},
            ]
            resp = self.ovms.query_llm(retry_messages, max_tokens=2048, temperature=0.0,
                                       response_schema=_STEP_SCHEMA)
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
        resp = self.ovms.query_llm(messages, max_tokens=2048, temperature=0.3,
                                   response_schema=_STEP_SCHEMA)
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
            step = ActionStep(**item)
            # Validate that action-specific required fields are present
            if step.action_type in ("hotkey", "key_press") and not step.key:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'key' is missing. "
                    f"Description: {step.description}"
                )
            if step.action_type == "type" and not step.value:
                raise ValueError(
                    f"Step {step.id} is 'type' but 'value' is missing."
                )
            if step.action_type in ("click", "right_click", "double_click") and not step.target:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'target' is missing."
                )
            steps.append(step)
        return steps
