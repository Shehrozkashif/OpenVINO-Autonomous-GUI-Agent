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

## 2. The four layers

Dependencies point one way: `ui → core → agents → desktop`.

```
ui/        PyQt6 command center. Reads the orchestrator's log stream.
core/      The loop and its policy. Decides what to do and whether it worked.
agents/    One model-facing job each: route, plan, ground, act, verify.
desktop/   Windows facts and effects. Reports; never decides.
```

The boundary that matters most is `core` ↔ `desktop`:

```python
desktop.system.count_process_windows("ms-teams.exe")   # a fact:  3
core.groundtruth.new_window_appeared(exe, baseline)    # a policy: did the launch count?
```

Facts are Windows-only and hard to test; policy is pure and covered by unit
tests. Keeping them apart is why the suite runs on Linux with no GPU.

## 3. Where things live

| Path | What it is |
|------|------------|
| `config.py` | Single source of truth: model names, endpoint, device, KV-cache sizes, `FORCE_MOUSE` |
| `start.py` | One-command bootstrap: GPU check → export models → launch OVMS → launch UI |
| `main.py` | Wires all agents together (`build_orchestrator()`) and starts the Qt app |
| `core/orchestrator.py` | **The heart.** Task loop, verification policy, budgets, replanning |
| `core/runstate.py` | `OrchestratorConfig` (every budget) and `SubtaskRun` (per-subtask state) |
| `core/groundtruth.py` | Deterministic checks: file on disk, launch confirmed, command effect, view switch, unfilled form fields |
| `core/subtasks.py` | Pure parsers over a subtask's text: save target, typed payload, commit button, required values |
| `core/apps.py` | App keyword → executable name / on-screen signal words |
| `core/anchor.py` | The task's app window: adopt it, refocus it, refuse clicks outside it |
| `core/firewall.py` | Regex classifier for destructive typed text (never an LLM) |
| `core/inference.py` | `InferenceClient` protocol + `OVMSClient`; `query_llm()` / `query_vlm()` pick the model |
| `core/types.py` | `SubTask` and `ActionStep` |
| `agents/router.py` | Instruction → subtasks; task-level **replanning**; missing-parameter questions |
| `agents/planning.py` | Subtask + screen context → next `ActionStep`(s) |
| `agents/grounding.py` | "Save button" → (x, y): Stage 0 UIA tree → Stage 1 OCR fuzzy match → Stage 2 VLM |
| `agents/coords.py` | Parsing a VLM's answer into a screen pixel (formats and value scales) |
| `agents/action.py` | Executes one `ActionStep` (dispatch to controller / UIA patterns) |
| `agents/reflection.py` | LLM/OCR verification when no deterministic check applies |
| `agents/prompts.py` | Every system prompt, in one file |
| `desktop/uia.py` | Windows UI Automation: element search + structured actions (`set_value`, `select`, `invoke`) |
| `desktop/input.py` | Low-level mouse/keyboard via Win32 `SendInput`, plus the kill switch |
| `desktop/capture.py` | Screenshots (GDI), frame hashing, and the mask hiding the agent's own window |
| `desktop/ocr.py` | RapidOCR engine and fuzzy text search |
| `desktop/system.py` | DPI, foreground window, process/window queries, installed apps, GPUs |
| `core/history.py` | SQLite record of tasks that completed cleanly — read by the UI, never by the loop |
| `ui/` | PyQt6 Mission Control. `events.py` parses orchestrator log lines into typed signals |
| `tests/unit/` | 453 tests, run anywhere: `pytest tests/unit` |
| `tests/live/` | Real-desktop suites (Windows + OVMS required); verified by disk/process ground truth |

## 4. Reading order

1. **`core/types.py`** — the two dataclasses everything passes around.
2. **`agents/prompts.py`** — read `ROUTER_SYSTEM_PROMPT` first; it defines what
   a "good decomposition" means, then `PLANNING_SYSTEM_PROMPT`'s ACTION
   REFERENCE for what a step may be.
3. **`core/runstate.py`** — every budget and limit in the system, on one page.
4. **`core/orchestrator.py`** — in this order:
   - `execute()` — the task-level story: elicit missing details → decompose →
     run subtasks → replan on failure → summarize.
   - `_execute_subtask()` — the step-level skeleton. Each phase is a named
     method: `_plan_next_step` (queue / text / visual escalation),
     `_run_step_attempts` (retry policy), `_judge_reflection` (verification
     verdicts), `_record_step_success` / `_record_step_failure` (bookkeeping,
     early exits, loop guard).
   - `_execute_step()` — grounding, gates and dispatch for a single action.
5. **`core/groundtruth.py`** — read alongside the loop: every "is it really
   done?" question the loop asks resolves here.
6. **`agents/grounding.py`** — the 3-stage fallback that turns element names
   into coordinates.
7. **`ui/events.py`** — how log lines become UI state (regex → signals).

## 5. Design principles (the "why" behind recurring patterns)

**Ground truth beats model judgment.** Wherever the real world can be read
directly, the code never asks an LLM. Files are checked on disk with
freshness, app launches against the process list, typed text by reading the
focused control back through UIA, form fields by `set_value` read-back. LLM
reflection is the fallback, not the default. All of it lives in
`core/groundtruth.py`.

**Facts and decisions are separate modules.** `desktop/` answers questions
about the machine and takes no position; `core/` decides what the answers
mean. That split is what makes the policy unit-testable off Windows.

**Idempotency decides retry policy.** Repeating a click is harmless;
repeating a `type`, `Enter`, `invoke`, or paste double-executes. Non-idempotent
actions are never blind-retried after they physically fire — on an uncertain
verdict the loop accepts and lets the *next* planning step read the live
screen and correct course. See `_run_step_attempts` / `_judge_reflection`.

**Every loop has a budget.** Steps per subtask, retries per step, consecutive
failures, wall-clock deadlines (the task deadline scales with plan size),
loop-guard limits per action type, replans per task. All of them are in
`core/runstate.py`. Nothing can run unbounded.

**Failures are recoverable state, not fatal.** A failed subtask triggers a
router `replan()` with the completed work held constant, and a subtask that
cannot be finished is skipped rather than killing queued work behind it. A run
that only finished through such a path is marked degraded and is not recorded
as a clean success.

**The agent decides from the live screen, not from its own past.** Nothing in
the loop reads `core/history.py` back. Earlier versions fed past runs into
planning — a "similar task" hint to the router and "this target failed before"
hints to the planner — and both changed behaviour based on state the user
could not see. The router hint was observed replacing a correct decomposition
with a stale plan. History is now write-only: a record for the operator.

**Deterministic security boundaries.** The action firewall
(`core/firewall.py`) is a regex over typed/shell text, not an LLM — it cannot
be prompt-injected. Destructive commands require human confirmation, and the
confirmation dialog times out to *deny*.

**The prompt is part of the code.** Router and planner behaviour is largely
defined in `agents/prompts.py`. If the agent decomposes or plans badly, fix
the prompt — but keep it next to the parsing code that enforces it.

**Log lines are an interface.** `ui/events.py` parses the orchestrator's log
stream to drive the mission timeline. Changing a log string changes the UI, so
treat those formats as API.

## 6. How a Teams meeting actually happens (end to end)

This is the flagship flow — trace it once and the whole pipeline clicks.

```
"schedule a teams meeting with the team tomorrow at 3pm"
  → orchestrator.execute()
      → router.missing_parameters()  → dialog asks for topic/duration if absent
      → router.decompose()           → [launch Teams, open the calendar & New meeting form,
                                        fill all fields, save & confirm]
      → per subtask: _execute_subtask()
          launch    → groundtruth.verify_launch(): ms-teams.exe in the process list
          navigate  → ctrl+4, confirmed by the window title ("Calendar | Microsoft Teams")
          form fill → planner emits set_value/select/invoke;
                      desktop/uia sets each control and reads it back
                      (Teams is a WebView2 app — the accessibility tree, not OCR,
                       is what proves the title/attendees/time actually landed)
          save      → the commit guard checks every required value is really on the
                      form, then the Save click completes the subtask
      → on a subtask failure: router.replan(completed, failed) → new queue
      → history.store_successful_task() once the whole task completes cleanly
```

## 7. Testing philosophy

Unit tests (`tests/unit/`, no display needed) pin the *policies*: dispatch
routing, retry/idempotency rules, budget math, replanning ID renumbering,
firewall coverage, prompt-output parsing. Live tests
(`tests/live/`, Windows + OVMS) prove the loop against real apps and verify
outcomes by **ground truth only** — files on disk, processes running, UIA
reads — never by trusting the model's own success claim.

Test doubles come from `tests/unit/conftest.py` (`make_grounder`,
`make_reflector`, `make_llm`, …). Build fakes from those helpers rather than
raw `MagicMock`s: a bare mock returns a mock from `min_confidence`, and
comparing that to a float raises inside the orchestrator — a crash that reads
like a real bug but is only a bad double.
