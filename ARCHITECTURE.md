# Architecture

A desktop GUI agent that runs **entirely on local models** (served by OpenVINO™
Model Server) and drives real Windows applications: it plans with an LLM,
locates UI elements through the accessibility tree / OCR / a grounding VLM,
acts through raw Win32 input, and verifies every step against ground truth.

```
                ┌────────────────────────────── OVMS (localhost:8000) ─┐
                │  qwen3-8b-int4-ov (LLM)   ui-tars-1.5-7b-int8 (VLM) │
                └───▲───────────▲───────────▲──────────────▲───────────┘
                    │           │           │              │
 instruction ─► Router ─► Orchestrator ─► Planner      Grounding S2 / verify
                (decompose)     │        (next step)
                                ▼
              per-subtask loop (core/orchestrator.py):
              goal check → plan → ground → act → verify → (replan)
                                │
                                ▼
              DesktopController (SendInput)  +  windows_uia (UIA patterns)
```

## The pipeline, end to end

1. **Elicit** (`RouterAgent.detect_missing_parameters`) — missing details
   (meeting time, recipient…) are asked from the user *before* execution.
2. **Decompose** (`agents/router.py`) — the LLM splits the instruction into
   `SubTask`s. An installed-app hint (`Get-StartApps` × whole-word match)
   steers it toward apps that actually exist on the machine.
3. **Per-subtask loop** (`core/orchestrator.py`) — plans ONE step at a time
   against the live screen:
   - **Screen context** = OCR text + `CLICKABLE CONTROLS` (exact names from
     the accessibility tree, disabled ones labeled) + `SYSTEM CLOCK` +
     goal-check evidence. The planner may only pick click targets that exist.
   - **Goal check** — one focused LLM yes/no per cycle: *is this subtask's end
     state already on screen?* Stops "click X to open Y" from re-clicking
     after Y opened. Cached per screen state.
   - **Plan** (`agents/planning.py`) — text path (OCR context → LLM steps);
     escalates to the visual path (screenshot → UI-TARS) when stuck.
   - **Ground** (`agents/grounding.py`) — 3 stages, first hit wins:
     S0 accessibility tree → S1 OCR fuzzy text → S2 VLM coordinates; then
     LLM rephrasings and a scroll-to-find hunt.
   - **Act** (`agents/action.py` → `core/controller.py`) — UIA-grounded clicks
     try the control's **pattern invoke** first (Invoke→Select→Toggle→
     DoDefaultAction), pixel click as fallback. `set_value` uses ValuePattern
     with read-back, else focus-and-type. Typed text passes the
     **action firewall** (`core/action_firewall.py`) first.
   - **Verify** (`agents/reflection.py`) — perceptual-hash delta (delta=0 ⇒
     the click provably did nothing) → LLM over OCR text plus the *focused
     control* read from the accessibility tree → VLM screenshot check only
     when uncertain.
4. **Recovery** — failures feed blacklists and a replanner whose prompt pins
   the original objective ("navigate away from dead ends, never reinterpret
   the goal").
5. **Report** (`Router.summarize_completion`) — on failure the last
   goal-check evidence is logged as `[BLOCKER] …` and put in the summary, so
   the user learns *why* ("Teams is at a sign-in screen"), not just "failed".

## Ground-truth mechanisms (the heart of the design)

Every recovery decision is anchored to something the OS can prove — never to
a model's opinion alone:

| Mechanism | Keyed by | Protects against |
|---|---|---|
| Dead-point blacklist (`grounding.mark_dead`) | (target, screen phash) | re-clicking a coordinate a phash delta=0 proved inert (pixel attempts only — a failed pattern invoke never dead-marks the pixel it didn't touch) |
| Invoke-dead blacklist (`orchestrator._invoke_dead`) | target, per task | WebView2 providers that accept Invoke/Toggle but do nothing |
| Occlusion hit-test (`windows_uia.covering_element`) | ElementFromPoint ancestor chain | clicking/invoking a control a dialog covers — the blocker's name is handed to the planner so it dismisses the overlay |
| Field-value read-back (controls list `= '…'` suffix) | ValuePattern on Edit/ComboBox | planner/goal-check blindness to form state (premature Save, re-filling already-set fields) |
| Controls-diff verification (`reflection._controls_delta_note`) | appeared/disappeared control labels | verifier misjudging actions whose only effect lives in the tree (an added attendee pill on an OCR-invisible WebView2 screen) |
| Negative grounding cache (`grounding._no_find`) | (target, screen phash) | re-running the 15-20 s find cascade for a known miss on an unchanged screen |
| Goal-check cache (`run.goal_check_cache`) | screen context hash | re-paying an LLM call to re-judge identical pixels |
| App anchor (`orchestrator._app_anchor`) | (hwnd, pid, exe) | operating on the wrong window; clicks that escape to another process are dead-marked and focus restored |
| Launch-skip / launch signals | process list + foreground | re-launching an app that is already up |
| Disabled-control filter (UIA `IsEnabled`) | accessibility tree | clicking/invoking grayed buttons; planner sees `[Button (disabled)]` |
| Focused-control read-back | `GetFocusedControl` | verifying typing/caret effects that are invisible to pixels |
| Action firewall | regex, no model | screen-injected destructive commands (`rm -rf`, format, …) |
| Kill switch (`controller.KillSwitch`) | OS input state | triple-Esc or mouse in top-left corner halts everything, mid-step |

## File map

| Path | Role |
|---|---|
| `main.py` | Wires everything, DPI awareness, warmups, starts the Qt app |
| `start.py` | Environment check, model download/export (venv-move-proof CLI shims), OVMS launch |
| `core/orchestrator.py` | The loop described above; all recovery policy |
| `core/windows_uia.py` | Accessibility tree: native FindAll batch search (reaches WebView2/Electron content), find/invoke/set_value/select/focus, interactive-elements list, focused-control info |
| `core/controller.py` | Raw SendInput mouse/keyboard, clipboard typing, kill switch |
| `core/ovms_client.py` | OpenAI-compatible chat client for OVMS, retry with backoff |
| `core/burst_executor.py` | Pre-grounded short action sequences with one verification at the end |
| `core/action_firewall.py` | Deterministic destructive-command guard for typed text |
| `core/capture/` | Screen capture (GDI), perceptual hashing, foreground-window-aware OCR snapshot |
| `core/protocols.py` | `SubTask`, `ActionStep`, burst primitives, `InferenceClient` protocol |
| `agents/router.py` | Decompose / replan / missing-parameter elicitation / completion summary |
| `agents/planning.py` | Next-step planning prompts (text + visual paths) |
| `agents/grounding.py` | 3-stage grounding, caches and blacklists, OCR engine |
| `agents/action.py` | ActionStep dispatch to controller / UIA patterns |
| `agents/reflection.py` | Step verification (phash → LLM → VLM) |
| `memory/task_memory.py` | SQLite: past successes (difflib similarity), failure hints, resume checkpoints |
| `utils/` | DPI awareness, GPU/app detection, clipboard, keyring credentials |
| `ui/` | PyQt6 command center: pages, HUD, event bus parsing the log stream, on-screen click pulse (excluded from the agent's own captures) |

## Models

| Role | Model | Why |
|---|---|---|
| Reasoning: decompose, plan, verify text, goal check, rephrase | `qwen3-8b-int4-ov` | fast JSON/instruction following on the iGPU; frees VRAM for an INT8 VLM |
| Visual: grounding coordinates, screenshot verification, visual planning | `ui-tars-1.5-7b-int8-ov` | purpose-trained GUI grounding; INT8 for more accurate coordinates |

Both stay resident in OVMS simultaneously; all calls go through one
OpenAI-compatible endpoint (`core/ovms_client.py`).

## Running and testing

```bash
python start.py          # environment check, models, OVMS, then the app
pytest tests/unit        # 500+ tests, no Windows/GPU needed (UIA is mocked)
ruff check .             # style baseline
```

Development loop: code on Linux → push → pull on the Windows AI PC → run a
real task → paste the log. Every mechanism above was added in response to a
specific failure visible in one of those logs, and each has regression tests
quoting that log.
