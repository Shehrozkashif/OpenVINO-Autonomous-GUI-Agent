# agents/router/router_agent.py
"""Router Agent — decomposes user instructions into sub-tasks."""

import json
import re
import uuid
from typing import List, Optional, Tuple

from loguru import logger

from core.protocols.a2a import InferenceClient, SubTask

_SUBTASK_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id":          {"type": "integer"},
            "description": {"type": "string"},
            "depends_on":  {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["id", "description", "depends_on"],
    },
}

ROUTER_SYSTEM_PROMPT = """You are a desktop automation task coordinator.
Break down user instructions into sub-tasks — one distinct action per sub-task.

Rules:
1. Each sub-task must describe ONE distinct action (open app, click element, type text, navigate, etc.)
2. Sub-tasks in logical order; set depends_on correctly so later steps wait for earlier ones
3. NEVER merge two different actions into one sub-task (e.g. "open app and type text" = 2 sub-tasks)
4. NEVER add wait, verify, confirm, or close steps unless explicitly asked
5. Every object MUST have: id (integer), description (string), depends_on (integer array)

DECOMPOSITION EXAMPLES:
  "open Firefox and go to wikipedia.org"
    → [{"id":1,"description":"open Firefox","depends_on":[]},
       {"id":2,"description":"navigate to wikipedia.org in Firefox","depends_on":[1]}]

  "open Files and navigate to Downloads"
    → [{"id":1,"description":"open the Files file manager","depends_on":[]},
       {"id":2,"description":"click on the Downloads folder","depends_on":[1]}]

  "take a screenshot"
    → [{"id":1,"description":"take a screenshot","depends_on":[]}]"""


class RouterAgent:
    def __init__(self, ovms_client: InferenceClient):
        self.ovms = ovms_client

    def decompose(
        self,
        instruction: str,
        screen_context: Optional[str] = None,
    ) -> Tuple[str, List[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}"
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
            user_content += "\nOnly include sub-tasks for things NOT already done. If the target is already visible, a single 'click' sub-task is enough."

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        resp = self.ovms.query_llm(messages, max_tokens=512, temperature=0.1,
                                   response_schema=_SUBTASK_SCHEMA)
        try:
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("[ROUTER] Parse failed — retrying with schema-only prompt")
            retry_messages = [
                {"role": "system", "content": "Output ONLY a JSON array of sub-tasks."},
                {"role": "user", "content": f"Sub-tasks for: {instruction}"},
            ]
            resp = self.ovms.query_llm(retry_messages, max_tokens=512, temperature=0.0,
                                       response_schema=_SUBTASK_SCHEMA)
            subtasks = self._parse_subtasks(resp.content)

        logger.info(f"[ROUTER] Decomposed into {len(subtasks)} sub-tasks:")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")

        return task_id, subtasks

    def _parse_subtasks(self, text: str) -> List[SubTask]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"No JSON array in router response: {text[:200]}")

        json_str = text[start_idx:end_idx + 1]
        # Fix trailing commas
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        # Fix single-quoted strings (some models use ' instead of ")
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"[ROUTER] JSON parse error: {e}\nRaw: {json_str[:300]}")
            raise

        subtasks = []
        for item in data:
            if "description" not in item:
                logger.warning(f"[ROUTER] Skipping item with no description: {item}")
                continue
            subtasks.append(SubTask(**item))
        return subtasks

    def summarize_completion(self, task_id: str, completed: list, success: bool) -> str:
        messages = [
            {"role": "system", "content": "Write a brief one-line task summary. No JSON."},
            {"role": "user", "content": f"Task {'succeeded' if success else 'failed'}. Sub-tasks completed: {completed}."},
        ]
        resp = self.ovms.query_llm(messages, max_tokens=100, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
