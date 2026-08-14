# Architecture

A desktop GUI agent that runs **entirely on local models** (served by OpenVINO™
Model Server) and drives real Windows applications: it plans with an LLM,
locates UI elements through the accessibility tree / OCR / a grounding VLM,
acts through raw Win32 input, and verifies every step against ground truth.

```
                ┌────────────────────────────── OVMS (localhost:8000) ─┐
                │  qwen3-8b-int4-ov (LLM)   ui-tars-1.5-7b-int8 (VLM)  │
                └───▲───────────▲───────────▲──────────────▲───────────┘
                    │           │           │              │
 instruction ─► Router ─► Orchestrator ─► Planner      Grounding S2 / verify
                (decompose)     │        (next step)
                                ▼
              per-subtask loop (core/orchestrator.py):
              goal check → plan → ground → act → verify → (replan)
                                │
                                ▼
              DesktopController (SendInput)  +  desktop/uia (UIA patterns)
```

## The four layers

Dependencies point one way only: `ui → core → agents → desktop`. Nothing in
`desktop/` knows about agents; nothing in `agents/` knows about the loop.

| Layer | Package | Answers |
|---|---|---|
| Interface | `ui/` | what the operator sees and controls |
| Decision | `core/` | what to do next, and whether it worked |
| Reasoning | `agents/` | what a model thinks (plan, target, verdict) |
| World | `desktop/` | what is actually on the machine |

The key rule lives at the `core` ↔ `desktop` boundary: **`desktop/` reports
facts and never decides**. `desktop/system.count_process_windows()` answers "how
many windows does this exe own"; `core/groundtruth.new_window_appeared()`
decides whether that counts as a launch. Policy is testable without Windows,
facts are swappable without touching policy.

## The pipeline, end to end

1. **Elicit** (`RouterAgent.missing_parameters`) — missing details (meeting
   time, recipient…) are asked from the user *before* execution.
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
   - **Act** (`agents/action.py` → `desktop/input.py`) — clicks are delivered
     with the real, gliding mouse at UIA-exact coordinates (`config.FORCE_MOUSE`);
     set `FORCE_MOUSE=False` to prefer the UIA pattern invoke instead. `set_value`
     uses ValuePattern with read-back, else focus-and-type. Typed text passes the
     **action firewall** (`core/firewall.py`) first.
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
| Occlusion hit-test (`desktop.uia.covering_element`) | ElementFromPoint ancestor chain | clicking/invoking a control a dialog covers — the blocker's name is handed to the planner so it dismisses the overlay |
| App anchor (`core/anchor.py`) | (hwnd, pid, exe) | operating on the wrong window; clicks that would escape to another process are rejected *before* they fire, by `WindowFromPoint` |
| Field-value read-back (controls list `= '…'` suffix) | ValuePattern on Edit/ComboBox | planner/goal-check blindness to form state (premature Save, re-filling already-set fields) |
| Commit guard (`groundtruth.unfilled_form_values`) | required values vs. live controls | a Save firing on a half-filled form — it "verifies" as a good click while the calendar stays empty |
| Controls-diff verification (`reflection._controls_delta_note`) | appeared/disappeared control labels | verifier misjudging actions whose only effect lives in the tree (an added attendee pill on an OCR-invisible WebView2 screen) |
| Negative grounding cache (`grounding._no_find`) | (target, screen phash) | re-running the 15-20 s find cascade for a known miss on an unchanged screen |
| Goal-check cache (`run.goal_check_cache`) | screen context hash | re-paying an LLM call to re-judge identical pixels |
| Launch verification (`groundtruth.verify_launch`) | process list + window count | re-launching an app that is already up; a focused old window passing as a new launch |
| Own-window mask (`desktop.capture.OwnWindowMask`) | per-cell `WindowFromPoint` grid | the agent reading its own log panel through OCR and "verifying" its own output |
| Disabled-control filter (UIA `IsEnabled`) | accessibility tree | clicking/invoking grayed buttons; planner sees `[Button (disabled)]` |
| Focused-control read-back | `GetFocusedControl` | verifying typing/caret effects that are invisible to pixels |
| Action firewall (`core/firewall.py`) | regex, no model | screen-injected destructive commands (`rm -rf`, format, …) |
| Kill switch (`desktop.input.KillSwitch`) | OS input state | triple-Esc or mouse in top-left corner halts everything, mid-step |

## File map

**`core/` — the loop and its policy**

| Path | Role |
|---|---|
| `core/orchestrator.py` | The See → Plan → Act → Verify loop; all recovery policy |
| `core/runstate.py` | Every budget and limit; the per-subtask run state |
| `core/groundtruth.py` | Checks the OS can prove (file on disk, launch, command effect, view switch, form fields) |
| `core/subtasks.py` | What a subtask's own words ask for (save target, typed payload, commit button, required values) |
| `core/apps.py` | App knowledge: description keyword → executable / on-screen signals |
| `core/anchor.py` | Which window the task owns, and keeping every click inside it |
| `core/firewall.py` | Deterministic destructive-command guard for typed text |
| `core/inference.py` | `InferenceClient` protocol + the OVMS client implementing it |
| `core/history.py` | A record of tasks that completed cleanly — written for the UI, never read back by the loop |
| `core/types.py` | `SubTask` and `ActionStep` — the two models every layer passes around |

**`agents/` — one job each, all behind `InferenceClient`**

| Path | Role |
|---|---|
| `agents/router.py` | Decompose / replan / missing-parameter elicitation / completion summary |
| `agents/planning.py` | Next-step planning, text and visual paths |
| `agents/grounding.py` | 3-stage grounding cascade, caches and blacklists |
| `agents/coords.py` | VLM answer → screen pixel (every format and value scale) |
| `agents/action.py` | `ActionStep` dispatch to the controller / UIA patterns |
| `agents/reflection.py` | Step verification (phash → LLM → VLM) |
| `agents/prompts.py` | Every instruction sent to a model, in one file |

**`desktop/` — Windows facts and effects**

| Path | Role |
|---|---|
| `desktop/uia.py` | Accessibility tree: native FindAll batch search (reaches WebView2/Electron content), find/invoke/set_value/select/focus, interactive-elements list |
| `desktop/input.py` | Raw SendInput mouse/keyboard, clipboard typing, kill switch |
| `desktop/capture.py` | Screen capture (GDI), perceptual hashing, own-window mask |
| `desktop/ocr.py` | RapidOCR engine + fuzzy label matching |
| `desktop/snapshot.py` | Foreground-window-aware OCR snapshot |
| `desktop/system.py` | DPI, windows, processes, installed apps, GPU detection |
| `desktop/clipboard.py`, `desktop/credentials.py` | Clipboard paste-typing, OS-keyring credentials |

**Everything else**

| Path | Role |
|---|---|
| `main.py` | Wires everything, DPI awareness, warmups, starts the Qt app |
| `start.py` | Setup in one command: installs packages, fetches the OVMS binary, downloads/exports models, launches OVMS |
| `config.py` | Model ids, device, KV-cache sizes, interaction flags |
| `ui/` | PyQt6 command center: pages, HUD, event bus parsing the log stream, on-screen click pulse |

## A note on logs

`core/orchestrator.py`'s log lines are a **public interface**: `ui/events.py`
parses them with regexes (`[SUBTASK n]`, `Step N: [type] …`, `Verified (conf=…)`)
to drive the live mission timeline. Changing a log format changes the UI.

## Models

| Role | Model | Why |
|---|---|---|
| Reasoning: decompose, plan, verify text, goal check, rephrase | `qwen3-8b-int4-ov` | fast JSON/instruction following on the iGPU; frees VRAM for an INT8 VLM |
| Visual: grounding coordinates, screenshot verification, visual planning | `ui-tars-1.5-7b-int8-ov` | purpose-trained GUI grounding; INT8 for more accurate coordinates |

Both stay resident in OVMS simultaneously; all calls go through one
OpenAI-compatible endpoint (`core/inference.py`).

## Running and testing

```bash
python start.py          # environment check, models, OVMS, then the app
pytest tests/unit        # 491 tests, no Windows or GPU needed (UIA is mocked)
ruff check .             # style baseline
```

Development loop: code on Linux → push → pull on the Windows AI PC → run a
real task → paste the log. Every mechanism above was added in response to a
specific failure visible in one of those logs, and each has regression tests
quoting that log.
