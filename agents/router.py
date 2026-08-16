# agents/router.py
"""Router Agent — a user instruction becomes an ordered list of subtasks.

Also owns the two other task-level conversations with the LLM: asking the user
for details the instruction leaves out (a meeting time, a recipient) before any
work starts, and re-planning the remaining work when a subtask fails.

The prompts it sends live in agents/prompts.py.
"""
import json
import re
import uuid

from loguru import logger

from agents.prompts import (
    FUNCTION_WORDS,
    ROUTER_SYSTEM_PROMPT,
    SUBTASK_SCHEMA,
    today_line,
)
from core.inference import InferenceClient
from core.types import SubTask


class RouterAgent:
    def __init__(self, client: InferenceClient):
        self.client = client

    @staticmethod
    def _installed_app_hint(instruction: str) -> str:
        """Ground-truth hint: installed apps whose names appear in the instruction.

        General by construction — the app list comes from the OS
        (Get-StartApps: Win32 + Store apps) and the match comes from the
        user's own words. No app names are hardcoded. Stops the router from
        planning a browser/web route when the user has the real app installed
        (e.g. 'outlook meeting' → the installed Outlook, not outlook.live.com).
        """
        from desktop.system import installed_apps
        apps = installed_apps()
        if not apps:
            logger.warning(
                "[ROUTER] Installed-app list unavailable (Get-StartApps "
                "failed/timed out) — no app hint for the planner"
            )
            return ""
        # WHOLE-WORD matching only, minus closed-class function words. Loose
        # substring matching sent a live run into the wrong app entirely:
        # "schedule a meeting WITH..." matched 'Task SCHEDULEr' and 'Firewall
        # WITH Advanced Security', and the router built the whole plan around
        # Task Scheduler. An app is hinted only when the user actually NAMES
        # a word of it ("outlook", "zoom", "spotify").
        tokens = set(re.findall(r"[a-zA-Z]{4,}", instruction.lower()))
        tokens -= FUNCTION_WORDS
        hits = sorted({
            a for a in apps
            if tokens & set(re.findall(r"[a-zA-Z]{4,}", a.lower()))
        })
        if not hits:
            logger.info(
                f"[ROUTER] No installed app matches the instruction "
                f"({len(apps)} apps checked)"
            )
            return ""
        logger.info(f"[ROUTER] Installed-app hint: {hits[:8]}")
        return (
            "Ground truth — these apps mentioned in the instruction are "
            "INSTALLED on this machine: " + ", ".join(hits[:8]) + ". "
            "Use the installed app (open it via Start menu search), NOT a "
            "browser/web version of it."
        )

    def decompose(
        self,
        instruction: str,
        screen_context: str | None = None,
    ) -> tuple[str, list[SubTask]]:
        task_id = str(uuid.uuid4())[:8]
        logger.info(f"[ROUTER] Task {task_id}: '{instruction}'")

        user_content = f"Instruction: {instruction}\n{today_line()}"
        app_hint = self._installed_app_hint(instruction)
        if app_hint:
            user_content += f"\n\n{app_hint}"
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
            user_content += (
                "\nUse screen context to skip already-done steps and prefer "
                "clicking visible icons over searching."
            )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # 768 tokens truncated real 8-subtask decompositions mid-string (seen
        # live) — the parse failed and the retry produced junk. 1536 fits any
        # realistic plan; generation still stops at the closing bracket.
        resp = self.client.query_llm(messages, max_tokens=1536, temperature=0.1,
                                   response_schema=SUBTASK_SCHEMA)
        try:
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError):
            logger.warning("[ROUTER] Parse failed — retrying with JSON-only reminder")
            subtasks = []
        else:
            # A syntactically valid but EMPTY array parses cleanly and then
            # sails through every backstop below, because they only inspect
            # sub-tasks that exist. Retry it like a parse failure: an 8B router
            # answering '[]' to a real instruction is a bad generation, not a
            # verdict that the task needs no work.
            if not subtasks:
                logger.warning(
                    "[ROUTER] Empty plan returned — retrying with JSON-only reminder"
                )

        if not subtasks:
            # Keep the FULL router system prompt: a bare "output JSON" retry
            # (no decomposition rules) returns conversational to-do items
            # ("Confirm the date with X") that no planner can act on.
            retry_messages = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT
                 + "\nOutput ONLY the JSON array — no prose, no explanation."},
                {"role": "user", "content": user_content},
            ]
            resp = self.client.query_llm(retry_messages, max_tokens=1536, temperature=0.0,
                                       response_schema=SUBTASK_SCHEMA)
            subtasks = self._parse_subtasks(resp.content)

        # Completeness backstop. The prompt rule helps but an 8B router
        # intermittently drops a trailing requested action. Detection is
        # deterministic and general (a list of action verbs); the FIX is handed
        # back to the LLM (re-prompt) so filenames/steps stay model-chosen, not
        # hardcoded.
        subtasks = self._ensure_complete(instruction, user_content, subtasks)
        # Second backstop: a plan that references sub-task ids it never defined
        # was truncated (the router dropped the opening launch/navigation
        # steps). Re-prompt for the complete plan, then sanitize any remaining
        # dangling refs so _topological_sort doesn't false-cycle.
        subtasks = self._ensure_connected(instruction, user_content, subtasks)

        logger.info(f"[ROUTER] Decomposed into {len(subtasks)} sub-tasks:")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")

        return task_id, subtasks

    # ── Task-level replanning (mid-task recovery) ────────────────────────────────

    def replan(
        self,
        instruction: str,
        completed_descs: list[str],
        failed_desc: str,
        screen_context: str | None = None,
        pending_descs: list[str] | None = None,
    ) -> list[SubTask]:
        """Produce a fresh sub-task list for the work of the FAILED subtask.

        The router sees what already succeeded (never repeated), which subtask
        failed (a different approach is required), which subtasks are still
        queued (the orchestrator preserves those verbatim — re-planning them
        here would duplicate work, and on a live run the model re-deriving
        "everything remaining" silently dropped the final 'click Save' step),
        and the live screen. Returns [] when re-decomposition fails.
        """
        done_block = (
            "\n".join(f"  - {d}" for d in completed_descs)
            if completed_descs else "  (none)"
        )
        pending_block = ""
        if pending_descs:
            listed = "\n".join(f"  - {d}" for d in pending_descs)
            pending_block = (
                f"\nThese sub-tasks are still QUEUED and will run AFTER your "
                f"new plan, unchanged — do NOT include them or their work in "
                f"your answer:\n{listed}\n"
            )
        user_content = (
            f"Instruction: {instruction}\n{today_line()}\n\n"
            f"This task is ALREADY IN PROGRESS. Sub-tasks completed successfully "
            f"(do NOT repeat them, their effects are already on the machine):\n"
            f"{done_block}\n\n"
            f"This sub-task FAILED: '{failed_desc}'\n"
            f"{pending_block}"
            f"Re-plan the FAILED sub-task's work as a fresh JSON array of "
            f"sub-tasks (you may split it into several). Use a DIFFERENT "
            f"approach (a different method or route — e.g. GUI instead of "
            f"terminal, a keyboard shortcut or menu path instead of the "
            f"button that failed, or navigating to the right view first). "
            f"The Windows search launcher opens APPLICATIONS only — NEVER "
            f"plan it to find a button, menu, or form inside an app that is "
            f"already open (searching a button name opens a WEB SEARCH in "
            f"the browser and derails the task). "
            f"depends_on may only reference ids inside this new array.\n"
            f"THE OBJECTIVE NEVER CHANGES: your plan plus the completed and "
            f"queued sub-tasks must still accomplish the user's original goal "
            f"(re-read the Instruction above). If the current screen is a dead "
            f"end for that goal, plan to NAVIGATE AWAY (a different section, "
            f"tab, or app) — do NOT reinterpret the goal as whatever the "
            f"current screen happens to offer."
        )
        if screen_context:
            user_content += f"\n\nCurrently visible on screen: {screen_context}"
        app_hint = self._installed_app_hint(instruction)
        if app_hint:
            user_content += f"\n\n{app_hint}"

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.2,
                response_schema=SUBTASK_SCHEMA,
            )
            subtasks = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Replan parse failed: {e}")
            return []

        logger.info(f"[ROUTER] Replanned remaining work into {len(subtasks)} sub-task(s):")
        for st in subtasks:
            logger.info(f"  [{st.id}] {st.description} (depends on: {st.depends_on})")
        return subtasks

    # ── Missing-parameter detection (asked BEFORE execution starts) ─────────────

    # Instructions that involve composing/scheduling something for other people
    # commonly omit details the task cannot proceed without (a time, a recipient).
    # Deterministic trigger — the LLM is only consulted for instructions in this
    # class, so simple tasks pay zero extra latency.
    _PARAM_TRIGGER = re.compile(
        r"\b(meeting|schedule|invite|appointment|event|remind(?:er)?|"
        r"e-?mail|send|compose|book|call)\b",
        re.IGNORECASE,
    )

    def missing_parameters(self, instruction: str) -> list[str]:
        """Return up to 3 short questions about details the instruction omits
        but the task cannot be completed without. [] when nothing is missing.
        """
        if not self._PARAM_TRIGGER.search(instruction):
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "You check a desktop-automation instruction for missing "
                    "REQUIRED details before an agent executes it. Output ONLY a "
                    "JSON array of short questions (strings), at most 3. Ask only "
                    "about details the task genuinely cannot be completed without "
                    "(a meeting needs a date/time; an email needs a recipient). "
                    "Never ask about details already given, optional settings, or "
                    "anything the agent can sensibly default. Output [] when "
                    "nothing essential is missing."
                ),
            },
            {"role": "user", "content": f"Instruction: {instruction}"},
        ]
        try:
            resp = self.client.query_llm(messages, max_tokens=200, temperature=0.0)
            text = re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL)
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                return []
            questions = json.loads(text[start:end + 1])
            return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:3]
        except Exception as e:
            logger.debug(f"[ROUTER] missing_parameters check skipped: {e}")
            return []

    # ── Completeness backstop (deterministic detection, LLM correction) ─────────

    # Trailing/finalizing actions a user commonly appends ("...and save it",
    # "...then close it"). General, not per-task: detection only — the fix is the
    # model's job. Each entry: canonical name → regex matching the user's verb.
    _FINALIZING_ACTIONS = {
        "save":     r"\bsave\b",
        "close":    r"\bclose\b",
        "send":     r"\b(?:send|e-?mail)\b",
        "print":    r"\bprint\b",
        "download": r"\bdownload\b",
        "delete":   r"\b(?:delete|remove)\b",
        "rename":   r"\brename\b",
    }

    @classmethod
    def _missing_actions(cls, instruction: str, subtasks: list[SubTask]) -> list[str]:
        """Return finalizing actions the user requested but no sub-task covers.

        Deterministic and general — it flags an OMISSION, it does not invent the
        missing step (that is the model's job on re-prompt).
        """
        instr = instruction.lower()
        covered = " ".join((st.description or "").lower() for st in subtasks)
        missing = []
        for name, pat in cls._FINALIZING_ACTIONS.items():
            if re.search(pat, instr) and not re.search(pat, covered):
                missing.append(name)
        return missing

    def _ensure_complete(
        self, instruction: str, user_content: str, subtasks: list[SubTask]
    ) -> list[SubTask]:
        """Re-prompt the router ONCE if it dropped an explicitly requested action.

        Keeps the model in charge of the fix (filenames, exact steps) — code only
        detects the gap. Accepts the retry only if it actually closes the gap;
        otherwise keeps the original so we never make things worse.
        """
        missing = self._missing_actions(instruction, subtasks)
        if not missing or not subtasks:
            return subtasks

        logger.warning(
            f"[ROUTER] Decomposition dropped requested action(s) {missing} — "
            f"re-prompting once to complete it"
        )
        prior = json.dumps([
            {"id": st.id, "description": st.description, "depends_on": st.depends_on}
            for st in subtasks
        ])
        correction = (
            f"Your sub-task list dropped these actions the user explicitly asked "
            f"for: {', '.join(missing)}. Re-output the COMPLETE ordered JSON array, "
            f"keeping every existing sub-task and adding one sub-task for each "
            f"missing action as the final step(s), with a concrete filename/target "
            f"where relevant. JSON array only."
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prior},
            {"role": "user", "content": correction},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.1,
                response_schema=SUBTASK_SCHEMA,
            )
            fixed = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Re-prompt parse failed ({e}) — keeping original")
            return subtasks

        # Only take the retry if it genuinely covers more of what was missing.
        if len(self._missing_actions(instruction, fixed)) < len(missing) and fixed:
            return fixed
        logger.warning("[ROUTER] Re-prompt did not close the gap — keeping original")
        return subtasks

    @staticmethod
    def _dangling_deps(subtasks: list[SubTask]) -> bool:
        """True if any sub-task depends on an id not present in the list.

        The router prompt requires depends_on to reference only ids inside the
        array, so a dangling reference is a deterministic signal the model
        dropped earlier sub-tasks (live: a Teams task came back as ONLY the
        final 'fill the schedule-meeting form' sub-task with depends_on:[3];
        the three opening steps — launch, open Calendar, click New meeting —
        were gone, so the agent ran the form-fill on the bare desktop and
        clicked random taskbar buttons looking for a Title field).
        """
        ids = {st.id for st in subtasks}
        return any(dep not in ids for st in subtasks for dep in (st.depends_on or []))

    @staticmethod
    def _sanitize_deps(subtasks: list[SubTask]) -> list[SubTask]:
        """Drop depends_on references to ids that aren't in the list.

        A dangling reference makes _topological_sort report a false
        'dependency cycle' and fall back to ID order silently. Stripping the
        bad refs lets the remaining plan run in a well-defined order instead.
        """
        ids = {st.id for st in subtasks}
        for st in subtasks:
            if st.depends_on:
                st.depends_on = [d for d in st.depends_on if d in ids]
        return subtasks

    def _ensure_connected(
        self, instruction: str, user_content: str, subtasks: list[SubTask]
    ) -> list[SubTask]:
        """Re-decompose ONCE if the plan is not self-contained.

        A dangling depends_on means the router truncated the plan and dropped
        the leading steps. Re-prompt for the COMPLETE ordered plan; accept the
        retry only if it is connected and no shorter. If the retry can't fix
        it, at least sanitize the dangling refs so the plan runs in order
        rather than tripping a misleading 'dependency cycle'.
        """
        if not subtasks or not self._dangling_deps(subtasks):
            return subtasks

        logger.warning(
            "[ROUTER] Decomposition references missing sub-task ids "
            "(dangling depends_on) — earlier steps were dropped; re-prompting "
            "for the COMPLETE plan"
        )
        prior = json.dumps([
            {"id": st.id, "description": st.description, "depends_on": st.depends_on}
            for st in subtasks
        ])
        correction = (
            "Your sub-task list is INCOMPLETE: a depends_on points at an id "
            "that is not in the list, which means you dropped the earlier "
            "steps. Re-output the COMPLETE ordered plan from the FIRST action "
            "to the last — launch/open the app, navigate to the right screen, "
            "THEN act — with ids starting at 1 and every depends_on referencing "
            "an id that exists in THIS array. JSON array only."
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": prior},
            {"role": "user", "content": correction},
        ]
        try:
            resp = self.client.query_llm(
                messages, max_tokens=1536, temperature=0.1,
                response_schema=SUBTASK_SCHEMA,
            )
            fixed = self._parse_subtasks(resp.content)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"[ROUTER] Re-prompt parse failed ({e}) — sanitizing deps")
            return self._sanitize_deps(subtasks)

        if fixed and not self._dangling_deps(fixed) and len(fixed) >= len(subtasks):
            return fixed
        logger.warning(
            "[ROUTER] Re-prompt did not produce a connected plan — sanitizing deps"
        )
        return self._sanitize_deps(subtasks)

    def _parse_subtasks(self, text: str) -> list[SubTask]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1:
            raise ValueError(f"No JSON array in router response: {text[:200]}")
        # A truncated generation (hit max_tokens mid-object) may have no
        # closing ']' at all — take everything and let the salvage below
        # recover the complete prefix.
        json_str = text[start_idx:end_idx + 1] if end_idx > start_idx else text[start_idx:]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Salvage a truncated array: cut back to the last complete object
            # and close the bracket. The complete prefix is real work the model
            # planned; anything dropped off the tail is restored by the
            # _ensure_complete() backstop, which re-prompts for missing actions.
            # Scan the FULL tail, not json_str — the rfind(']') trim above may
            # have cut at an inner bracket (e.g. "depends_on":[1]), dropping
            # the closing brace of the last complete object.
            tail = text[start_idx:]
            tail = re.sub(r",\s*([\]}])", r"\1", tail)
            tail = re.sub(r":\s*'([^']*)'", r': "\1"', tail)
            data = None
            for m in reversed(list(re.finditer(r"\}", tail))):
                try:
                    candidate = json.loads(tail[:m.end()] + "]")
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, list) and candidate:
                    data = candidate
                    logger.warning(
                        f"[ROUTER] Truncated JSON — salvaged "
                        f"{len(data)} complete sub-task(s)"
                    )
                    break
            if data is None:
                logger.error(f"[ROUTER] JSON parse error: {e}\nRaw: {json_str[:300]}")
                raise

        subtasks = []
        for item in data:
            if "description" not in item:
                logger.warning(f"[ROUTER] Skipping item with no description: {item}")
                continue
            subtasks.append(SubTask(**item))
        # depends_on may only reference ids inside this very array — replans
        # routinely emit a dangling 0 ("depends on the completed work"), which
        # the topological sort reads as a cycle and noisily falls back on.
        ids = {s.id for s in subtasks}
        for s in subtasks:
            s.depends_on = [d for d in s.depends_on if d in ids and d != s.id]
        return subtasks

    def summarize_completion(
        self, task_id: str, completed: list, success: bool, blocker: str = "",
        skipped: list | None = None,
    ) -> str:
        content = f"Task {'succeeded' if success else 'failed'}. Sub-tasks completed: {completed}."
        if skipped:
            content += (
                f" These sub-tasks could NOT be completed and were skipped: "
                f"{skipped}. The summary MUST say the task partially "
                f"succeeded and name what was skipped."
            )
        if blocker and not success:
            content += (
                f" Screen evidence of what blocked it: {blocker}. "
                "State the blocker plainly so the user knows what to fix."
            )
        messages = [
            {"role": "system", "content": "Write a brief one-line task summary. No JSON."},
            {"role": "user", "content": content},
        ]
        resp = self.client.query_llm(messages, max_tokens=120, temperature=0.3)
        return re.sub(r"<think>.*?</think>", "", resp.content, flags=re.DOTALL).strip()
