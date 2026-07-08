# Code Tour — how to read this codebase

A guided path through the project for someone seeing it for the first time.
Read this top to bottom once; afterwards you should know where any behaviour
lives and why it was built that way.

---

## 1. The one-paragraph mental model

The agent runs a **See → Plan → Act → Verify** loop against a real Windows
desktop. A user instruction ("schedule a Teams meeting tomorrow at 3pm") is
decomposed by the **Router** into subtasks; each subtask is executed by
planning **one step at a time** against the live screen; every step is
**verified** before the next one is planned — preferably by ground truth
(a file on disk, a process in the task list, a control's value read back
through the accessibility tree) and only by LLM judgment when no ground
truth exists. Both models (an LLM for reasoning, a VLM for vision) run
locally on one OpenVINO Model Server endpoint.

## 2. Where things live

| Path | What it is |
|------|------------|
| `config.py` | Single source of truth: model names, endpoint, device, KV-cache sizes |
| `start.py` | One-command bootstrap: GPU check → export models → launch OVMS → launch UI |
| `main.py` | Wires all agents together (`build_orchestrator()`) and starts the Qt app |
| `core/orchestrator.py` | **The heart.** Task loop, verification policy, budgets, replanning, checkpoints |
| `core/protocols.py` | Shared dataclasses (`SubTask`, `ActionStep`) and the `InferenceClient` protocol |
| `core/ovms_client.py` | HTTP client for OVMS; `query_llm()` / `query_vlm()` select the model per request |
| `core/windows_uia.py` | Windows UI Automation: element search + structured actions (`set_value`, `select`, `invoke`) |
| `core/controller.py` | Low-level mouse/keyboard via Win32 `SendInput` |
| `core/capture/` | Screenshots (GDI), OCR thumbnails, perceptual hashing |
| `agents/router.py` | Instruction → subtasks; task-level **replanning**; missing-parameter questions |
| `agents/planning.py` | Subtask + screen context → next `ActionStep`(s) |
| `agents/grounding.py` | "Save button" → (x, y): Stage 0 UIA tree → Stage 1 OCR fuzzy match → Stage 2 VLM |
| `agents/action.py` | Executes one `ActionStep` (dispatch to controller / UIA / shell) |
| `agents/reflection.py` | LLM/OCR verification when no deterministic check applies |
| `memory/task_memory.py` | SQLite: success plans, failure patterns, per-subtask **checkpoints** |
| `ui/` | PyQt6 Mission Control. `events.py` parses orchestrator log lines into typed signals |
| `tests/unit/` | 546 tests, run anywhere: `venv` python + `pytest tests/unit` |
| `tests/live/` | Real-desktop suites (Windows + OVMS required); verified by disk/process ground truth |

## 3. Reading order

1. **`core/protocols.py`** — the two dataclasses everything passes around.
2. **`agents/router.py`** — read `ROUTER_SYSTEM_PROMPT` first; it defines what
   a "good decomposition" means. Then `decompose()`, `replan()`,
   `missing_parameters()`.
3. **`core/orchestrator.py`** — in this order:
   - `OrchestratorConfig` — every safety budget and limit, documented inline.
   - `execute()` — the task-level story: elicit missing details → resume from
     checkpoint → decompose → run subtasks → replan on failure → summarize.
   - `_execute_subtask()` — the step-level skeleton. Each phase is a named
     method: `_plan_next_step` (queue / text / visual escalation),
     `_run_step_attempts` (retry policy), `_judge_reflection` (verification
     verdicts), `_record_step_success` / `_record_step_failure` (bookkeeping,
     early exits, loop guard). `_SubtaskRun` holds the loop state.
   - `_execute_step()` — grounding + firewall + dispatch for a single action.
4. **`agents/grounding.py`** — the 3-stage fallback that turns element names
   into coordinates.
5. **`ui/events.py`** — how log lines become UI state (regex → signals).

## 4. Design principles (the "why" behind recurring patterns)

**Ground truth beats model judgment.** Wherever the real world can be read
directly, the code never asks an LLM. Files are checked on disk with
freshness (`_file_saved_fresh`), app launches against the process list
(`_goal_confirmed`), typed text by reading the focused control back through
UIA (`_typed_text_in_focused_control`), form fields by `set_value` read-back.
LLM reflection is the fallback, not the default.

**Idempotency decides retry policy.** Repeating a click is harmless;
repeating a `type`, `Enter`, `invoke`, or paste double-executes. Non-idempotent
actions are never blind-retried after they physically fire — on an uncertain
verdict the loop accepts and lets the *next* planning step read the live
screen and correct course. See `_run_step_attempts` / `_judge_reflection`.

**Every loop has a budget.** Steps per subtask, retries per step, consecutive
failures, wall-clock deadlines (task deadline scales with plan size —
`_effective_task_deadline`), loop-guard limits per action type
(`DEDUP_LIMIT_BY_ACTION_TYPE`), replans per task. Nothing can run unbounded.

**Failures are recoverable state, not fatal.** A failed subtask triggers a
router `replan()` with the completed work held constant; progress is
checkpointed to SQLite after every subtask so an interrupted task resumes
instead of restarting; degraded runs (loops, VLM-declared finishes) are
quarantined from success memory so bad plans are never reused.

**Deterministic security boundaries.** The action firewall
(`_execute_step`) is a regex over typed/shell text, not an LLM — it cannot
be prompt-injected. Destructive commands require human confirmation, and the
confirmation dialog times out to *deny*.

**The prompt is part of the code.** Router and planner behaviour is largely
defined in their system prompts (`ROUTER_SYSTEM_PROMPT`, the planner's
ACTION REFERENCE). If the agent decomposes or plans badly, fix the prompt
next to the parsing code that enforces it.

## 5. How a Teams meeting actually happens (end to end)

This is the flagship flow — trace it once and the whole pipeline clicks.

```
"schedule a teams meeting with the team tomorrow at 3pm"
  → orchestrator.execute()
      → router.missing_parameters()  → dialog asks for topic/duration if absent
      → memory.load_checkpoint()     → resume hint if a recent run was interrupted
      → router.decompose()           → [launch Teams, open the calendar & New meeting form,
                                        fill all fields, save & confirm]
      → per subtask: _execute_subtask()
          launch    → verified by ms-teams.exe in the process list
          form fill → planner emits set_value/select/invoke;
                      windows_uia sets each control and reads it back
                      (Teams is a WebView2 app — the accessibility tree, not OCR,
                       is what proves the title/attendees/time actually landed)
          save      → invoke "Save"; verified in the Teams calendar via UIA
      → on a subtask failure: router.replan(completed, failed) → new queue
      → memory.save_checkpoint() after each subtask; cleared on success
```

## 6. Testing philosophy

Unit tests (`tests/unit/`, no display needed) pin the *policies*: dispatch
routing, retry/idempotency rules, budget math, replanning ID renumbering,
checkpoint lifecycle, firewall coverage, prompt-output parsing. Live tests
(`tests/live/`, Windows + OVMS) prove the loop against real apps and verify
outcomes by **ground truth only** — files on disk, processes running, UIA
reads — never by trusting the model's own success claim.
