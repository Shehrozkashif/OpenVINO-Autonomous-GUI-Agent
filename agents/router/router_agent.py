# agents/router/router_agent.py
"""
Router Agent — the task coordinator.
Receives user instruction, decomposes into sub-tasks, manages workflow.
Uses DeepSeek-R1 LLM. Does NOT look at the screen.
"""
import json
import re
import uuid
from typing import List, Tuple

from loguru import logger

from core.pipeline.ovms_client import OVMSClient
from core.protocols.a2a import SubTask

ROUTER_SYSTEM_PROMPT = """You are a desktop automation task coordinator.
Break down user instructions into logical sub-tasks.

Rules:
1. Each sub-task is one coherent action (e.g. "open VS Code", "enable autosave")
2. Sub-tasks must be in logical order
3. Set depends_on to IDs of sub-tasks that must complete first
4. Maximum 10 sub-tasks per instruction
5. Be specific: "find and click the Settings menu" not "go to settings"

Output ONLY a JSON array, nothing else:
[
  {"id": 1, "description": "...", "depends_on": []},
  {"id": 2, "description": "...", "depends_on": [1]}
]"""


class RouterAgent:
    def __init__(self, ovms_client: OVMSClient):
        self.ovms = ovms_client

    def decompose(self, instruction: str) -> Tuple[str, List[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Break down this instruction:\n\n{instruction}"}
        ]
        resp = self.ovms.query_llm(messages, max_tokens=512, temperature=0.3)
        subtasks = self._parse_subtasks(resp.content)

        logger.info(f"[ROUTER] Decomposed into {len(subtasks)} sub-tasks:")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")

        return task_id, subtasks

    def _parse_subtasks(self, text: str) -> List[SubTask]:
        arr_match = re.search(r'\[.*?\]', text, re.DOTALL)
        if not arr_match:
            raise ValueError(f"No JSON array in router response: {text[:200]}")
        data = json.loads(arr_match.group())
        return [SubTask(**item) for item in data]

    def summarize_completion(self, task_id: str, completed: list, success: bool) -> str:
        messages = [
            {"role": "system", "content": "Write brief, friendly one-line task summaries."},
            {"role": "user", "content":
             f"Task {'succeeded' if success else 'failed'}. "
             f"Completed sub-tasks: {completed}. Write one summary sentence."}
        ]
        resp = self.ovms.query_llm(messages, max_tokens=80, temperature=0.5)
        return resp.content.strip()

