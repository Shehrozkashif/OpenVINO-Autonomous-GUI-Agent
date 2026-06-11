# ui/events.py
"""
AgentEventBus — turns the orchestrator's log stream into structured Qt signals.

The orchestrator reports progress through `self.log(str)` with stable prefixes
([TASK START], [ROUTER], [SUBTASK n], "Step N: [type] …", "Verified (conf=…)",
[FIREWALL], …). Parsing that stream here gives the UI a typed event model and
a live agent state machine WITHOUT touching core/ — the pipeline stays
UI-agnostic and the old plain-text log remains available as a raw console.
"""
import re
from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


class AgentState(Enum):
    IDLE = "IDLE"
    ROUTING = "ROUTING"
    PLANNING = "PLANNING"
    GROUNDING = "GROUNDING"
    ACTING = "ACTING"
    VERIFYING = "VERIFYING"
    RECOVERING = "RECOVERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

BUSY_STATES = frozenset({
    AgentState.ROUTING, AgentState.PLANNING, AgentState.GROUNDING,
    AgentState.ACTING, AgentState.VERIFYING, AgentState.RECOVERING,
})


# ── Regex table (mirrors core/orchestrator.py log strings) ───────────────────

_RX_TASK_START   = re.compile(r"\[TASK START\] '(.*)'")
_RX_ROUTER       = re.compile(r"\[ROUTER\] (\d+) sub-task")
_RX_BURST        = re.compile(r"\[BURST\] .*?(\d+) steps?\)")
_RX_SUBTASK      = re.compile(r"\[SUBTASK (\d+)\] (.+)")
_RX_STEP         = re.compile(r"^\s*Step (\d+): \[(\w+)\] (.*)")
_RX_VERIFIED     = re.compile(r"^\s*Verified \(conf=([\d.]+)\)")
_RX_VERIFY_FAIL  = re.compile(r"^\s*Verification failed: (.*?) \(conf=([\d.]+)\)")
_RX_UNCERTAIN    = re.compile(r"^\s*Uncertain result — retrying")
_RX_RETRY        = re.compile(r"^\s*Retry (\d+)/(\d+)")
_RX_STEP_FAILED  = re.compile(r"^\s*Step failed — re-evaluating")
_RX_EXTRACT      = re.compile(r"\[EXTRACT\] '(.+?)' = '(.*)'")
_RX_MEMORY       = re.compile(r"\[MEMORY\] Similar past task found \(sim=([\d.]+)\)")
_RX_TASK_DONE    = re.compile(r"\[TASK DONE\] (.*) \(([\d.]+)s\)")
_RX_STOPPED      = re.compile(r"\[TASK\] Stopped by user")
_RX_FIREWALL     = re.compile(r"\[FIREWALL\] (.*)")
_RX_GUARD        = re.compile(r"\[(GUARD|LOOP-GUARD|CMD-CHECK|GOAL-CHECK[\w-]*|CHECK|PRE-CHECK)\] (.*)")
_RX_VISUAL       = re.compile(r"\[VISUAL-REPLAN\] (.*)")
_RX_SCROLL_FIND  = re.compile(r"\[SCROLL-FIND\] (.*)")
_RX_SETTLE       = re.compile(r"\[SETTLE\]")

# ── Deep pipeline events (loguru bridge — agents/, tools/, core/) ────────────
_RX_PLANNED      = re.compile(r"\[PLANNING\] Next: \[(\w+)\] (.*)")
_RX_LOCATED      = re.compile(
    r"\[GROUNDING\] '(.+?)' → \((\d+),(\d+)\) conf=([\d.]+) method=(\w+)")
_RX_EXECUTED     = re.compile(r"\[ACTION\] (\w+)")
_RX_VLM_VERIFY   = re.compile(r"\[REFLECTION\] .*escalating to VLM")
_RX_KILLSWITCH   = re.compile(r"\[KILL-SWITCH\] Armed — (.*)")


class LoguruBridge(QObject):
    """Forwards loguru records (from any thread) into the UI thread.

    The deepest pipeline events — planner decisions, grounding results with
    coordinates/confidence, reflection escalating to the VLM — are loguru
    logs, not orchestrator.log() lines. This sink emits them as a queued Qt
    signal so the event bus can parse them too.
    """

    line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sink_id = None

    def install(self):
        if self._sink_id is not None:
            return
        try:
            from loguru import logger
            self._sink_id = logger.add(
                lambda msg: self.line.emit(str(msg).rstrip()),
                level="INFO", format="{message}")
        except Exception:
            pass

    def uninstall(self):
        if self._sink_id is not None:
            try:
                from loguru import logger
                logger.remove(self._sink_id)
            except Exception:
                pass
            self._sink_id = None


class AgentEventBus(QObject):
    """Feed raw orchestrator log lines into feed(); consume typed signals."""

    state_changed    = pyqtSignal(object)            # AgentState
    task_started     = pyqtSignal(str)               # instruction
    plan_ready       = pyqtSignal(int)               # n subtasks
    subtask_started  = pyqtSignal(int, str)          # id, description
    subtask_finished = pyqtSignal(int, bool)         # id, success
    step_started     = pyqtSignal(int, str, str)     # n, action_type, description
    step_verified    = pyqtSignal(float)             # confidence
    step_failed      = pyqtSignal(str, float)        # reason, confidence
    retrying         = pyqtSignal(int, int)          # attempt, max
    guard_event      = pyqtSignal(str, str)          # kind, message
    extracted        = pyqtSignal(str, str)          # key, value
    memory_hint      = pyqtSignal(float)             # similarity
    task_done        = pyqtSignal(str, float)        # summary, elapsed_s
    raw_line         = pyqtSignal(str)
    # Deep pipeline events (via the loguru bridge)
    detail           = pyqtSignal(str)               # "what is happening right now"
    element_located  = pyqtSignal(str, int, int, float, str)  # target,x,y,conf,method

    def __init__(self):
        super().__init__()
        self.state = AgentState.IDLE
        self.instruction = ""
        self.current_subtask = ""
        self.current_step = ""
        self.last_confidence = -1.0
        self.steps_total = 0
        self.steps_failed = 0
        self.retries = 0

    # ── state machine ─────────────────────────────────────────────────────────

    def _set_state(self, state: AgentState):
        if state != self.state:
            self.state = state
            self.state_changed.emit(state)

    def reset(self):
        self.instruction = ""
        self.current_subtask = ""
        self.current_step = ""
        self.last_confidence = -1.0
        self.steps_total = 0
        self.steps_failed = 0
        self.retries = 0
        self._set_state(AgentState.IDLE)

    def finish(self, success: bool, stopped: bool = False):
        """Called by the window when the worker thread returns."""
        if stopped:
            self._set_state(AgentState.STOPPED)
        else:
            self._set_state(AgentState.COMPLETE if success else AgentState.FAILED)

    # ── parser ────────────────────────────────────────────────────────────────

    def feed(self, line: str):
        self.raw_line.emit(line)
        for sub in line.splitlines():
            if sub.strip():
                self._parse(sub)

    def _parse(self, line: str):
        m = _RX_TASK_START.search(line)
        if m:
            self.reset()
            self.instruction = m.group(1)
            self._set_state(AgentState.ROUTING)
            self.task_started.emit(self.instruction)
            self.detail.emit("Breaking the mission into subtasks…")
            return

        m = _RX_ROUTER.search(line) or _RX_BURST.search(line)
        if m:
            self._set_state(AgentState.PLANNING)
            self.plan_ready.emit(int(m.group(1)))
            return

        m = _RX_SUBTASK.search(line)
        if m:
            sid, rest = int(m.group(1)), m.group(2).strip()
            if rest == "Complete":
                self.subtask_finished.emit(sid, True)
            elif rest.startswith("Failed"):
                self.subtask_finished.emit(sid, False)
            else:
                self.current_subtask = rest
                self._set_state(AgentState.PLANNING)
                self.subtask_started.emit(sid, rest)
            return

        m = _RX_STEP.search(line)
        if m:
            n, action, desc = int(m.group(1)), m.group(2), m.group(3).strip()
            self.current_step = desc
            self.steps_total += 1
            # Click-family steps locate their target on screen first
            if action in ("click", "right_click", "double_click", "drag"):
                self._set_state(AgentState.GROUNDING)
                self.detail.emit(f"Looking for the target on screen — {desc}")
            else:
                self._set_state(AgentState.ACTING)
                self.detail.emit(desc)
            self.step_started.emit(n, action, desc)
            return

        m = _RX_VERIFIED.search(line)
        if m:
            self.last_confidence = float(m.group(1))
            self._set_state(AgentState.PLANNING)
            self.step_verified.emit(self.last_confidence)
            self.detail.emit("Step confirmed — planning the next move…")
            return

        m = _RX_VERIFY_FAIL.search(line)
        if m:
            self.last_confidence = float(m.group(2))
            self._set_state(AgentState.RECOVERING)
            self.step_failed.emit(m.group(1), self.last_confidence)
            return

        m = _RX_RETRY.search(line)
        if m:
            self.retries += 1
            self._set_state(AgentState.RECOVERING)
            self.retrying.emit(int(m.group(1)), int(m.group(2)))
            return

        if _RX_UNCERTAIN.search(line):
            self._set_state(AgentState.VERIFYING)
            self.guard_event.emit("VERIFY", "Outcome uncertain — re-checking")
            return

        if _RX_STEP_FAILED.search(line):
            self.steps_failed += 1
            self._set_state(AgentState.RECOVERING)
            return

        m = _RX_EXTRACT.search(line)
        if m:
            self.extracted.emit(m.group(1), m.group(2))
            return

        m = _RX_MEMORY.search(line)
        if m:
            self.memory_hint.emit(float(m.group(1)))
            return

        m = _RX_FIREWALL.search(line)
        if m:
            self.guard_event.emit("FIREWALL", m.group(1).strip())
            return

        m = _RX_VISUAL.search(line)
        if m:
            self._set_state(AgentState.RECOVERING)
            self.guard_event.emit("VISION", m.group(1).strip())
            return

        m = _RX_SCROLL_FIND.search(line)
        if m:
            self.guard_event.emit("SEARCH", m.group(1).strip())
            return

        m = _RX_GUARD.search(line)
        if m:
            self.guard_event.emit(m.group(1), m.group(2).strip())
            return

        m = _RX_TASK_DONE.search(line)
        if m:
            self.task_done.emit(m.group(1).strip(), float(m.group(2)))
            return

        if _RX_STOPPED.search(line):
            self._set_state(AgentState.STOPPED)
            return

        # ── Deep pipeline events (loguru bridge) ──────────────────────────────

        m = _RX_PLANNED.search(line)
        if m:
            self.detail.emit(f"Planned: {m.group(2).strip()}")
            return

        m = _RX_LOCATED.search(line)
        if m:
            target, x, y = m.group(1), int(m.group(2)), int(m.group(3))
            conf, method = float(m.group(4)), m.group(5)
            self._set_state(AgentState.ACTING)
            self.detail.emit(f"Located “{target}” ({conf:.0%} · {method})")
            self.element_located.emit(target, x, y, conf, method)
            return

        m = _RX_EXECUTED.search(line)
        if m:
            if self.state in (AgentState.ACTING, AgentState.GROUNDING):
                self._set_state(AgentState.VERIFYING)
                self.detail.emit("Action sent — verifying the result…")
            return

        if _RX_VLM_VERIFY.search(line):
            self._set_state(AgentState.VERIFYING)
            self.detail.emit(
                "Double-checking with the vision model — this can take a "
                "minute while it loads…")
            return

        m = _RX_KILLSWITCH.search(line)
        if m:
            self.guard_event.emit("KILL-SWITCH", f"Armed — {m.group(1)}")
            return
