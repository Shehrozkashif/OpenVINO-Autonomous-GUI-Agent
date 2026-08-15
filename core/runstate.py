# core/runstate.py
"""Budgets, limits, and the mutable state of one subtask run.

Everything that bounds the agent lives here, in one readable place: no loop in
this project may run unbounded. The numbers are tuned to the 8B planner's
latency (~7-15 s per planning call); swapping in a slower model means scaling
the deadlines with it, or long subtasks die mid-progress with the budget spent
on planning rather than on failures.
"""
from dataclasses import dataclass, field


@dataclass
class OrchestratorConfig:
    """Every safety budget the orchestrator obeys. 0 disables a budget."""

    max_retries_per_step: int = 3
    reflection_wait_s: float = 0.5
    max_steps_per_subtask: int = 20
    max_scroll_find_attempts: int = 6      # scrolls before giving up on an off-screen element
    consecutive_failures_limit: int = 3

    # Wall-clock budgets. task_deadline_s is a FLOOR: the effective budget
    # scales with plan size (n_subtasks × subtask_deadline_s), so a legitimate
    # 8-subtask task is not killed by a cap tuned for 2-subtask tasks, while a
    # stuck 2-subtask task still dies quickly.
    task_deadline_s: float = 600.0         # minimum budget for a whole instruction
    subtask_deadline_s: float = 240.0      # hard cap on a single subtask

    # Consecutive step failures before planning escalates from the text path
    # (OCR context → LLM) to the visual path (screenshot → UI-TARS), which sees
    # the icons and layout the text path is blind to.
    visual_replan_after: int = 2

    # Times the router may be re-consulted for a fresh plan of the remaining
    # work after a subtask fails, before the task is declared failed.
    max_task_replans: int = 2


# Per-action-type limit on identical successful steps within ONE subtask,
# counted as a TOTAL rather than consecutively — so A-B-A-B ping-pong loops
# (a hallucinated verifier alternating between two clicks) trip it too.
# Stricter for high-signal actions, lenient for navigation keys.
DEDUP_LIMIT_BY_ACTION_TYPE: dict[str, int] = {
    "type":        1,   # same text typed twice → almost certainly a loop
    "click":       2,   # allow up to 2 repeats (e.g. re-focusing a window)
    "key_press":   3,   # allow up to 3 repeats (e.g. several Escape/arrow presses)
    "right_click": 1,   # a context menu should open once
    "invoke":      1,   # a button pressed twice → double submit
    "set_value":   1,   # same field set to the same value twice → a loop
    "select":      2,   # re-selecting is harmless but twice is enough
}


@dataclass
class SubtaskRun:
    """Mutable state threaded through one subtask's step loop.

    Bundling it in one object lets each phase of the loop live in its own named
    method instead of one interleaved 500-line body.
    """

    started_at: float

    # ── Ground-truth facts about the subtask, fixed at setup time ────────────
    is_launch_goal: bool = False    # "open <app>" — verified via the process list
    goal_proc: str = ""             # expected process name for that launch
    baseline_windows: int = 0       # window count before the launch attempt
    is_cmd_subtask: bool = False    # "run: <cmd>" — verified on disk/terminal
    save_target: str = ""           # "save as <path>" — verified on disk
    type_payload: str = ""          # "type: <text>" — done once typed and verified

    # ── Rolling execution state ──────────────────────────────────────────────
    completed: list = field(default_factory=list)        # step history shown to the planner
    consecutive_failures: int = 0
    cached_ocr: str = ""            # reflection's OCR, reused by the next plan
    screen_context: str = ""        # the context the current step was planned on
    step_sig_counts: dict = field(default_factory=dict)  # step signature → successes
    goal_check_cache: dict = field(default_factory=dict)  # screen-context hash → (ok, evidence)
    typed_ok: bool = False          # a type step succeeded in this subtask
    last_error: str = ""            # most recent step failure reason

    # Steps planned ahead by the last plan_steps() call. Executing from this
    # queue skips the ~7-10 s planning call per step; it is flushed whenever a
    # step fails or its outcome is uncertain, forcing a fresh plan against the
    # live screen.
    step_queue: list = field(default_factory=list)

    @property
    def acted(self) -> bool:
        """True once any step in this subtask has succeeded."""
        return any(not c.startswith("[FAILED") for c in self.completed)

    @property
    def last_success(self) -> str:
        """Description of the most recent successful step, "" if none."""
        return next((c for c in reversed(self.completed)
                     if not c.startswith("[FAILED")), "")

    @property
    def has_deterministic_check(self) -> bool:
        """True when this subtask completes by a check the OS can prove, so the
        LLM goal check must not be consulted.
        """
        return bool(self.is_launch_goal or self.is_cmd_subtask
                    or self.save_target or self.type_payload)
