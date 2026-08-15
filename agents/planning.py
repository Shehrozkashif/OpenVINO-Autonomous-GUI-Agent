# agents/planning.py
"""Planning Agent — subtask + live screen → the next action step(s).

Two paths:
  text   OCR/UIA screen context → LLM → JSON steps. The normal route.
  visual a screenshot → UI-TARS → one action with pixel coordinates. The
         recovery route, used when the text path has stalled: it sees the
         icons and layout that text context is blind to.

The prompts both paths send live in agents/prompts.py.
"""
import json
import re

from loguru import logger

from agents.prompts import (
    COMPLETION_RULES,
    OS_CONTEXT,
    PLANNING_SYSTEM_PROMPT,
    STEP_SCHEMA,
    VISUAL_PLAN_PROMPT,
    VISUAL_PLAN_SYSTEM,
)
from core.inference import InferenceClient
from core.types import ActionStep, SubTask

# The completion rules used to be appended to the END of the user prompt, after
# the screen text. That put 490 fixed tokens behind a block that changes every
# step, so OVMS's prefix cache could never reach them and the GPU re-prefilled
# them on every planning call. They are constant instructions, so they belong in
# the constant half of the prompt — same words to the model, now cacheable.
_PLANNING_SYSTEM = PLANNING_SYSTEM_PROMPT + COMPLETION_RULES


class PlanningParseError(Exception):
    """Planner LLM output could not be parsed into steps, even after a retry.

    Distinct from "goal achieved" (an empty step array): the orchestrator must
    treat this as a FAILED planning attempt, never as subtask completion.
    """

# Imported inside functions to avoid a hard circular-import at module load time.
# TYPE_CHECKING guard is sufficient for type hints only; we need the class at
# runtime for isinstance checks, so the deferred import pattern is used instead.
_ScreenSnapshot = None  # resolved on first use


def _get_snapshot_class():
    global _ScreenSnapshot
    if _ScreenSnapshot is None:
        from desktop.snapshot import ScreenSnapshot  # noqa: PLC0415
        _ScreenSnapshot = ScreenSnapshot
    return _ScreenSnapshot


def _resize28(dim: int) -> int:
    """qwen2.5-VL smart_resize rounds each image side to a multiple of 28
    (patch 14 × merge 2). Our screenshots stay well under the model's max_pixels,
    so the model sees the sent image rounded to /28 and emits absolute pixel
    coordinates in that space. Mirror of grounding._qwen_resize_dim.
    """
    return max(28, int(round(dim / 28)) * 28)


def _parse_visual_action(
    text: str, subtask_id: int, screen_w: int, screen_h: int,
    display_w: int = 0, display_h: int = 0,
) -> ActionStep | None:
    """Parse a UI-TARS action line into an ActionStep.

    Click-family steps carry explicit screen-pixel coordinates in `value`
    (as "x,y"), so the orchestrator executes them directly without grounding.

    display_w/display_h: pixel size of the screenshot actually sent to the VLM.
    UI-TARS-1.5 (qwen2.5-VL) returns ABSOLUTE pixels in the /28 smart-resized
    image space, so coordinates are mapped screen = raw / resize28(sent) * screen
    (config.VLM_COORD_SPACE). When the sent size is unknown, or the convention is
    pinned to "norm1000", the legacy 0-1000 mapping is used instead.

    Returns None for finished(). Raises PlanningParseError when nothing parses.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if re.search(r"\bfinished\s*\(", text):
        return None

    try:
        import config as _cfg  # noqa: PLC0415
        _mode = str(getattr(_cfg, "VLM_COORD_SPACE", "pixels")).lower()
    except Exception:
        _mode = "pixels"

    def _map(raw: float, sent_dim: int, screen_dim: int) -> int:
        if _mode != "norm1000" and sent_dim and sent_dim > 1:
            return int(raw / _resize28(sent_dim) * screen_dim)
        return int(raw / 1000 * screen_dim)

    def _step(action_type, *, target=None, value=None, key=None, desc=""):
        return ActionStep(
            id=1, subtask_id=subtask_id, action_type=action_type,
            target=target, value=value, key=key,
            description=desc, verification="",
        )

    # Quantised UI-TARS builds vary the coordinate wrapper freely: '[[x,y]]',
    # '[[x,y)', '(x,y)', '[x, y]'… — accept brackets AND parens on both sides.
    # A coordinate slightly past 1000 (seen live: '[[1001, 104)') must clamp
    # to the screen edge, not click one pixel off-screen (delta=0 forever).
    def _clamp(px: int, py: int) -> tuple[int, int]:
        return (max(0, min(px, screen_w - 1)), max(0, min(py, screen_h - 1)))

    _OPEN = r"['\"]?\s*[\[\(]{0,4}\s*"

    # Click family with a 0-1000 bounding box
    m = re.search(
        r"(click|left_double|right_single)\s*\(\s*start_box\s*=\s*" + _OPEN +
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",
        text,
    )
    if m:
        kind = {"click": "click", "left_double": "double_click",
                "right_single": "right_click"}[m.group(1)]
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(2, 6))
        px, py = _clamp(_map((x1 + x2) / 2, display_w, screen_w),
                        _map((y1 + y2) / 2, display_h, screen_h))
        return _step(kind, value=f"{px},{py}",
                     desc=f"[visual] {kind} at ({px},{py})")

    # Click family with a 0-1000 center point
    m = re.search(
        r"(click|left_double|right_single)\s*\(\s*start_box\s*=\s*" + _OPEN +
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\]\)']",
        text,
    )
    if m:
        kind = {"click": "click", "left_double": "double_click",
                "right_single": "right_click"}[m.group(1)]
        px, py = _clamp(_map(float(m.group(2)), display_w, screen_w),
                        _map(float(m.group(3)), display_h, screen_h))
        return _step(kind, value=f"{px},{py}",
                     desc=f"[visual] {kind} at ({px},{py})")

    m = re.search(r"type\s*\(\s*content\s*=\s*'(.*?)'\s*\)", text, re.DOTALL)
    if m:
        content = m.group(1).replace("\\'", "'").replace("\\n", "\n")
        return _step("type", value=content, desc=f"[visual] type '{content[:40]}'")

    m = re.search(r"hotkey\s*\(\s*key\s*=\s*'([^']+)'\s*\)", text)
    if m:
        combo = "+".join(m.group(1).replace("+", " ").split())
        return _step("hotkey", key=combo, desc=f"[visual] hotkey {combo}")

    m = re.search(r"press\s*\(\s*key\s*=\s*'([^']+)'\s*\)", text)
    if m:
        return _step("key_press", key=m.group(1).strip(),
                     desc=f"[visual] press {m.group(1).strip()}")

    m = re.search(r"scroll\s*\(\s*direction\s*=\s*'([^']+)'\s*\)", text)
    if m:
        return _step("scroll", value=m.group(1).strip().lower(),
                     desc=f"[visual] scroll {m.group(1).strip()}")

    if re.search(r"\bwait\s*\(", text):
        return _step("wait", value="1.0", desc="[visual] wait for screen")

    raise PlanningParseError(f"unrecognised visual action: {text[:120]}")



class PlanningAgent:
    def __init__(self, client: InferenceClient):
        self.client = client

    def plan_next_step_visual(
        self,
        subtask: SubTask,
        image_base64: str,
        completed: list[str] = None,
        screen_w: int = 1920,
        screen_h: int = 1080,
        display_w: int = 0,
        display_h: int = 0,
    ) -> ActionStep | None:
        """Visual recovery planning: send the actual screenshot to the VLM (UI-TARS)
        and get the next action directly, with pixel coordinates.

        Used by the orchestrator when text-based planning has failed repeatedly —
        the text path is blind to icons, images, and layout, which is usually why
        it got stuck. Returns None when the VLM says finished().
        Raises PlanningParseError when the output is unusable.
        """
        history = ""
        if completed:
            lines = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(completed[-8:]))
            history = f"\nSteps attempted so far (FAILED ones are marked):\n{lines}\n"

        prompt = VISUAL_PLAN_PROMPT.format(
            os_name=OS_CONTEXT, goal=subtask.description, history=history,
        )
        resp = self.client.query_vlm(
            prompt=prompt,
            image_base64=image_base64,
            max_tokens=150,
            temperature=0.0,
            system_prompt=VISUAL_PLAN_SYSTEM,
        )
        step = _parse_visual_action(resp.content, subtask.id, screen_w, screen_h,
                                    display_w, display_h)
        if step is not None:
            step.id = len(completed or []) + 1
            logger.info(f"[PLANNING/VISUAL] Next: [{step.action_type}] {step.description}")
        else:
            logger.info("[PLANNING/VISUAL] VLM reports goal achieved (finished())")
        return step

    def plan_next_step(
        self,
        subtask: SubTask,
        screen_context: str = None,
        completed: list[str] = None,
        task_context: list[str] = None,
        snapshot=None,   # Optional[ScreenSnapshot] — when provided overrides screen_context
    ) -> ActionStep | None:
        """Compatibility wrapper around plan_steps(): first remaining step or None."""
        steps = self.plan_steps(
            subtask, screen_context, completed,
            task_context=task_context, snapshot=snapshot,
        )
        return steps[0] if steps else None

    def plan_steps(
        self,
        subtask: SubTask,
        screen_context: str = None,
        completed: list[str] = None,
        task_context: list[str] = None,
        snapshot=None,   # Optional[ScreenSnapshot] — when provided overrides screen_context
    ) -> list[ActionStep] | None:
        """Dynamic planning: return ALL remaining action steps toward the subtask goal.
        Returns None when the goal is already achieved (planner returns empty array).

        One LLM call plans the whole remaining sequence; the orchestrator executes
        it step by step and re-plans (a fresh call, live screen) only when a step
        fails or its outcome is uncertain. This is the main latency lever: the
        planning call is ~7-10 s on the target hardware, so paying it once per
        sequence instead of once per step cuts most of the per-step cost.

        task_context:   descriptions of subtasks already completed in this overall task.
        snapshot:       ScreenSnapshot from capture_snapshot(); when provided its
                        format_for_planner() output replaces the raw screen_context string.
        """
        # Use structured snapshot context when available (Fix 1.2)
        SnapshotClass = _get_snapshot_class()
        if snapshot is not None and isinstance(snapshot, SnapshotClass):
            screen_context = snapshot.format_for_planner()

        messages = [
            {"role": "system", "content": _PLANNING_SYSTEM},
            {"role": "user", "content": self._build_planning_prompt(
                subtask, screen_context, completed, task_context,
            )},
        ]
        steps = self._query_with_retry(messages, subtask.id)
        if not steps:
            logger.info("[PLANNING] Goal achieved — no more steps needed")
            return None
        return self._finalize_steps(steps, subtask, completed)

    @staticmethod
    def _build_planning_prompt(
        subtask: SubTask,
        screen_context: str,
        completed: list[str],
        task_context: list[str],
    ) -> str:
        """Assemble the user prompt: goal + context blocks.

        Everything here VARIES from step to step. Anything constant belongs in
        the system prompt instead (see _PLANNING_SYSTEM), so the server's prefix
        cache can keep it.
        """
        # Inter-subtask context — what was done before this subtask
        ctx_block = ""
        if task_context:
            ctx_lines = "\n".join(f"  - {d}" for d in task_context)
            ctx_block = f"\nSubtasks already completed in this task:\n{ctx_lines}\n"

        # Within-subtask step history
        history = ""
        if completed:
            lines = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(completed))
            history = f"\nSteps completed so far toward this goal:\n{lines}\n"

        user_content = f"Goal: {subtask.description}{ctx_block}{history}"

        # Reinforce the right_click rule in the user prompt when relevant
        _sd_lower = subtask.description.lower()
        if "right click" in _sd_lower or "right-click" in _sd_lower:
            user_content += (
                '\nRULE: The goal contains "right click" — you MUST output '
                'action_type: "right_click", NEVER "click".\n'
            )

        if screen_context:
            user_content += f"\nText currently visible on screen: {screen_context}"

        return user_content

    def _query_with_retry(self, messages: list[dict], subtask_id: int) -> list[ActionStep]:
        """One planning call, with a single temperature-0 retry on parse errors."""
        # 1152: the 14B pretty-prints JSON (spaces/newlines) and blew through
        # 768 on multi-field form plans — truncation mid-object. The salvage
        # in _parse_steps recovers the prefix, but headroom beats salvage.
        resp = self.client.query_llm(
            messages, max_tokens=1152, temperature=0.2,
            response_schema=STEP_SCHEMA,
        )
        try:
            return self._parse_steps(resp.content, subtask_id)
        except (ValueError, json.JSONDecodeError) as e:
            # A parse error is NOT "goal achieved" — returning None here made the
            # orchestrator mark the subtask complete on garbage output. Retry once
            # at temperature 0; if still unparseable, raise so the orchestrator
            # counts a planning failure and can recover.
            logger.warning(f"[PLANNING] parse error: {e} — retrying once at temperature 0")
            resp = self.client.query_llm(
                messages, max_tokens=1152, temperature=0.0,
                response_schema=STEP_SCHEMA,
            )
            try:
                return self._parse_steps(resp.content, subtask_id)
            except (ValueError, json.JSONDecodeError) as e2:
                raise PlanningParseError(
                    f"planner output unparseable after retry: {e2}"
                ) from e2

    @staticmethod
    def _finalize_steps(
        steps: list[ActionStep], subtask: SubTask, completed: list[str],
    ) -> list[ActionStep]:
        """Renumber steps after the completed history and apply overrides."""
        # Deterministic right_click override — the LLM occasionally outputs "click"
        # even when the subtask description clearly says "right click".
        _sub_lower = subtask.description.lower()
        _wants_right_click = (
            _sub_lower.startswith("right click")
            or "right click on" in _sub_lower
            or _sub_lower.startswith("right-click")
            or "right-click on" in _sub_lower
        )
        _base_id = len(completed or []) + 1
        for i, step in enumerate(steps):
            step.id = _base_id + i
            if _wants_right_click and step.action_type == "click":
                step.action_type = "right_click"
                logger.info(
                    "[PLANNING] Overrode action_type to right_click "
                    "(subtask says 'right click')"
                )

        _queued = f" (+{len(steps) - 1} queued)" if len(steps) > 1 else ""
        logger.info(
            f"[PLANNING] Next: [{steps[0].action_type}] {steps[0].description}{_queued}"
        )
        return steps

    def _parse_steps(self, text: str, subtask_id: int) -> list[ActionStep]:
        if "</think>" in text:
            text = text.split("</think>")[-1]
        else:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1:
            raise ValueError(f"No JSON array in planning response: {text[:200]}")

        # A truncated generation (hit max_tokens mid-object) has no closing
        # ']' — keep the whole tail and let the salvage below recover the
        # complete prefix (same approach as the router's _parse_subtasks).
        json_str = text[start_idx:end_idx + 1] if end_idx > start_idx else text[start_idx:]
        json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Salvage a truncated array: cut back to the last complete object
            # and close the bracket. The complete prefix is real planned work
            # — executing it beats discarding a 45-100 s planning call, and
            # the orchestrator re-plans against the live screen once the
            # queue drains. Live (14B): two plans truncated mid-string at the
            # 768-token cap; parse+retry burned ~200 s each and aborted the
            # subtask, when 5+ complete steps sat in the prefix both times.
            tail = re.sub(r",\s*([\]}])", r"\1", text[start_idx:])
            data = None
            for m in reversed(list(re.finditer(r"\}", tail))):
                try:
                    candidate = json.loads(tail[:m.end()] + "]")
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, list) and candidate:
                    data = candidate
                    logger.warning(
                        f"[PLANNING] Plan truncated mid-generation — salvaged "
                        f"{len(candidate)} complete step(s) from the prefix"
                    )
                    break
            if data is None:
                logger.error(f"[PLANNING] JSON parse error: {e}\nRaw: {json_str[:300]}")
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
            if step.action_type in ("hotkey", "key_press") and not step.key:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'key' is missing."
                )
            if step.action_type == "type" and step.value is None:
                raise ValueError(f"Step {step.id} is 'type' but 'value' is missing.")
            if step.action_type in ("click", "right_click", "double_click") and not step.target:
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'target' is missing."
                )
            if step.action_type in ("set_value", "select") and (
                not step.target or step.value is None
            ):
                raise ValueError(
                    f"Step {step.id} is '{step.action_type}' but 'target' or 'value' is missing."
                )
            if step.action_type == "invoke" and not step.target:
                raise ValueError(f"Step {step.id} is 'invoke' but 'target' is missing.")
            steps.append(step)
        return steps
