<div align="center">

<a href="https://github.com/openvinotoolkit/openvino"><img src="docs/assets/openvino-logo.svg" alt="OpenVINO" height="72"></a>

# Desktop GUI Agent

**Tell your computer what to do — in plain English.**
An autonomous desktop agent that observes your screen, plans, clicks, types, and
**verifies every single step** — running entirely on your own machine.
No cloud. No API keys. No data ever leaves your desk.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](#installation)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Backend](https://img.shields.io/badge/inference-Ollama%20%2B%20vLLM-orange)](https://ollama.com)
[![GUI](https://img.shields.io/badge/GUI-PyQt6-41cd52)](https://www.riverbankcomputing.com/software/pyqt/)
[![Tests](https://img.shields.io/badge/tests-355%20passing-brightgreen)](#running-tests)

[Quick Start](#quick-start) •
[How It Works](#how-it-works) •
[Architecture](#architecture) •
[Safety](#safety) •
[Installation](#installation) •
[Contributing](CONTRIBUTING.md)

</div>

---

## Demo

```
User:  "Open Firefox and navigate to wikipedia.org"

Agent: [ROUTER]   2 sub-tasks: open Firefox → navigate to URL
       [PLAN]     Super → click search bar → type firefox → Enter
       [GROUND]   OCR found "Type to search" at (850, 78)
       [ACTION]   key_press super
       [VERIFY]   ✓ Activities overlay appeared (conf=1.00)
       [ACTION]   type firefox
       [VERIFY]   ✓ Firefox search result visible (conf=1.00)
       [ACTION]   key_press enter
       [VERIFY]   ✓ Firefox window opened (conf=0.95)
       [PLAN]     hotkey ctrl+l → type wikipedia.org → Enter
       [ACTION]   hotkey ctrl+l
       [VERIFY]   ✓ Address bar focused (conf=1.00)
       [ACTION]   type wikipedia.org
       [ACTION]   key_press enter
       [DONE]     Task completed in 29s
```

---

## Highlights

|     | Feature | What it means |
|-----|---------|---------------|
| 🔒 | **100 % local** | All inference runs on your GPU via Ollama / vLLM — nothing is sent to the cloud |
| 👁 | **Verifies every step** | A reflection agent checks the screen after each action; failures trigger automatic replanning |
| 🎯 | **3-stage grounding** | UIA accessibility tree → OCR fuzzy-match → vision model, fastest path first |
| 🛡 | **Prompt-injection-proof firewall** | Destructive shell commands are blocked by a deterministic classifier that never calls a model |
| 🚨 | **Hardware-style kill switch** | Triple-Esc or slam the mouse into the top-left corner — the agent stops instantly |
| 🧠 | **Learns from experience** | Successful task plans and known failure patterns are stored in SQLite and reused |
| ⚡ | **Burst execution** | Recognised multi-step patterns (context menus, rename dialogs) run with zero LLM calls |
| 🖥 | **Cross-platform** | Windows 10/11 and Linux (X11), with per-platform input and capture backends |
| 💾 | **Fits 6 GB VRAM** | Quantised 7–8 B models with automatic model swapping — no datacenter GPU required |

---

## How It Works

The agent runs a closed-loop **See → Plan → Act → Verify** cycle at every step:

```
User Instruction
      │
      ▼
┌─────────────┐   decompose    ┌──────────────────────┐
│ Router Agent│ ──────────────▶│  Sub-tasks (ordered) │
└─────────────┘                └──────────────────────┘
                                          │
                          ┌───────────────┘  for each sub-task
                          ▼
                ┌──────────────────┐
                │  Planning Agent  │  plans ONE step at a time
                └──────────────────┘  from the LIVE screen state
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Grounding Agent  │    │  Action Agent    │
    │  UIA / OCR / VLM │    │  clicks, types,  │
    │  → screen (x, y) │    │  presses keys    │
    └──────────────────┘    └──────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Reflection Agent    │  OCR→LLM verifies outcome
                          └──────────────────────┘  (VLM check for visual steps)
                                     │
                          confirmed? ─▶ next step
                             failed? ─▶ retry / replan
```

Planning is **dynamic**: the planner sees the live screen before every step, so
it recovers from popups, focus changes, and failed actions instead of blindly
executing a stale plan.

### Grounding Pipeline (fastest → most robust)

| Stage | Method | When it fires |
|-------|--------|---------------|
| 0 | Windows UIA accessibility tree | Windows only; exact coordinates in ~20–50 ms |
| 1 | OCR fuzzy-match (RapidOCR) | Target text is visible on screen — all platforms |
| 2 | VLM direct (x, y) coordinates (UI-TARS) | Icon, field, or unlabelled element |

If all stages miss, the grounder asks the LLM for alternative label phrasings
and retries the pipeline.

---

## Architecture

The system is organised into five layers. Every agent depends only on the
`InferenceClient` protocol — never on a concrete backend — so inference
engines are drop-in replaceable.

```mermaid
flowchart TB
    USER(["💬 &nbsp;User instruction — <i>'Open Firefox and go to wikipedia.org'</i>"])

    subgraph UI_LAYER["&nbsp;UI Layer · ui/&nbsp;"]
        direction LR
        GUI["PyQt6<br/>Command Center"]
        EVBUS["AgentEventBus<br/>log stream → timeline"]
        HUD["Mission HUD<br/>always-on-top"]
        GUI --- EVBUS --- HUD
    end

    subgraph ORCH_LAYER["&nbsp;Orchestration Layer · core/&nbsp;"]
        direction LR
        ORCH["TaskOrchestrator<br/>retries · replanning ·<br/>deadlines · loop guard"]
        BURST["BurstExecutor<br/>zero-LLM<br/>fast paths"]
        FIREWALL["Action Firewall<br/>injection-proof"]
        KILL["Kill Switch<br/>triple-Esc"]
        MEMORY[("Task Memory<br/>SQLite")]
        ORCH --- BURST --- FIREWALL --- KILL --- MEMORY
    end

    subgraph AGENT_LAYER["&nbsp;Agent Layer · agents/ — every agent depends only on the InferenceClient protocol&nbsp;"]
        direction LR
        ROUTER["Router<br/>instruction →<br/>subtasks"]
        PLANNER["Planner<br/>one step at a time,<br/>from the live screen"]
        GROUND["Grounding<br/>UIA → OCR → VLM<br/>target → (x, y)"]
        ACTION["Action<br/>click · type ·<br/>keys · drag"]
        REFLECT["Reflection<br/>OCR→LLM verify ·<br/>VLM for visual steps"]
        ROUTER --- PLANNER --- GROUND --- ACTION --- REFLECT
    end

    subgraph PLATFORM_LAYER["&nbsp;Platform Layer · core/capture · tools/ · utils/&nbsp;"]
        direction LR
        PERCEPTION["👁 Perception — ScreenCapture (Xlib / GDI) · RapidOCR · Windows UIA"]
        INPUT["⌨️ Input — DesktopController (XTest / pynput) · credential vault (OS keyring)"]
    end

    subgraph INFER_LAYER["&nbsp;Inference Layer · core/pipeline — 100% local&nbsp;"]
        direction LR
        OLLAMA["Ollama<br/>qwen3:8b LLM ·<br/>UI-TARS VLM fallback"]
        CLIENT["OllamaClient<br/>implements<br/>InferenceClient"]
        VLLM["vLLM · optional<br/>UI-TARS-1.5-7B<br/>primary VLM"]
        CLIENT --> OLLAMA
        CLIENT -- "auto-detected<br/>on :8000" --> VLLM
    end

    USER --> UI_LAYER
    UI_LAYER -- "run task ↓ &nbsp;·&nbsp; ↑ live step events" --> ORCH_LAYER
    ORCH_LAYER -- "See → Plan → Act → Verify loop" --> AGENT_LAYER
    AGENT_LAYER -- "capture · OCR · clicks · keys" --> PLATFORM_LAYER
    AGENT_LAYER -- "plan · ground · verify queries" --> INFER_LAYER
```

**How to read it:** the orchestrator owns the loop — it consults memory before
routing, screens every typed command through the firewall, and arms the kill
switch for the duration of a task. Agents do one job each and touch the world
only through the platform layer. All model calls funnel through a single
client behind the `InferenceClient` protocol (`core/protocols/a2a.py`) — which
is what makes the planned OpenVINO backend a drop-in addition, not a rewrite.

| Agent | Consumes | Produces |
|-------|----------|----------|
| Router | instruction, screen context, memory hints | ordered `SubTask` list |
| Planner | subtask, live OCR context, step history | next `ActionStep` (or *done*) |
| Grounding | target description, screen | `(x, y)` + confidence |
| Action | grounded step | real mouse / keyboard events |
| Reflection | post-action screen | verdict: success · fail · uncertain |

### Reliability Engineering

Real desktops are messy. The orchestrator defends against the failure modes
that actually happen in live runs:

- **Loop guard** — per-action limits on identical repeated steps; a plan stuck
  in a loop is detected and stopped instead of clicking forever.
- **Idempotency protection** — non-repeatable actions (typing, Enter, paste)
  are never blind-retried after an uncertain verdict; the planner re-evaluates
  the live screen instead, so text is never typed twice.
- **Deterministic command verification** — terminal commands are verified
  against the real filesystem (file created / deleted / fresh mtime), because
  a successful shell command prints nothing and OCR would misread that
  silence as failure.
- **Launch verification** — "open X" subtasks are confirmed by process and
  window checks; focusing an *existing* window does not count as launching a
  new one.
- **Visual replanning** — when text-based planning stalls, the agent escalates
  to the vision model with a full screenshot to see what OCR can't.
- **Degraded-run quarantine** — tasks that finish through a recovery path are
  never stored as reusable successes, so broken plans cannot poison future
  routing.
- **Memory with failure patterns** — known-bad target/action combinations are
  fed to the planner as warnings before it repeats them.

---

## Models

Model ids live in [`config.py`](config.py) — the single source of truth.

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| **LLM** | `qwen3:8b` via Ollama | ~5 GB | Routing, planning, reflection reasoning |
| **VLM (preferred)** | `ByteDance-Seed/UI-TARS-1.5-7B` via vLLM on port 8000 | ~16 GB download | GUI grounding, visual verification |
| **VLM (Ollama)** | `hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q4_K_S` via Ollama | ~5 GB | Used when vLLM is unavailable |

> **OpenVINO™ roadmap** — inference currently runs through Ollama (with
> optional vLLM for the VLM). An OpenVINO execution backend is planned work:
> the `InferenceClient` protocol in `core/protocols/a2a.py` keeps every agent
> backend-agnostic, and `models/OpenVINO/` holds candidate INT4/INT8 model
> packages for that integration, targeting Intel® CPU / iGPU / NPU execution.

---

## Quick Start

```bash
git clone https://github.com/Shehrozkashif/intel-openvino-desktop-agent.git
cd intel-openvino-desktop-agent
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python start.py
```

`start.py` does the rest: detects your GPU, starts Ollama with the right
device assignment, starts vLLM when it is installed, prepares any missing
Ollama models, and opens the agent GUI.

```bash
# Pre-fill the instruction box
python start.py --prompt "Open VS Code and enable autosave"

# Pre-fill and run immediately on startup
python start.py --prompt "Search for OpenVINO documentation" --auto-run
```

---

## Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 24.04 / Windows 11 |
| Python | 3.10 | 3.12 |
| RAM | 16 GB | 32 GB |
| VRAM | 6 GB (models swap) | 24 GB (both models resident) |
| Disk | 20 GB free | 30 GB free |
| Display | X11 (Linux) | X11 or Windows desktop |

---

## Installation

### Linux

```bash
# 1. Clone the repository
git clone https://github.com/Shehrozkashif/intel-openvino-desktop-agent.git
cd intel-openvino-desktop-agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Ollama  (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh
```

> **No sudo required.** `start.py` extracts the missing Qt system library
> (`libxcb-cursor0`) to `~/.local_xcb` automatically on first run.

### Windows

```powershell
# 1. Clone the repository
git clone https://github.com/Shehrozkashif/intel-openvino-desktop-agent.git
cd intel-openvino-desktop-agent

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Ollama
# Download from https://ollama.com/download/windows and run the installer
```

> **Note:** On Windows, `pynput` uses `win32 SendInput` for keyboard injection —
> no extra drivers or admin rights needed.

---

## Running the Agent

```bash
# Linux
source venv/bin/activate
python start.py

# Windows
venv\Scripts\activate
python start.py
```

`start.py` handles everything automatically:

1. **Environment setup** — configures `LD_LIBRARY_PATH` on Linux (no sudo)
2. **GPU detection** — finds AMD ROCm / NVIDIA CUDA GPUs and assigns devices
3. **Ollama check** — verifies Ollama is running; starts it with the right GPU env if not
4. **vLLM check** — starts vLLM for UI-TARS when installed (optional, better accuracy)
5. **Model check** — pulls the LLM and configured Ollama VLM when needed
6. **Launches** `main.py` — the agent GUI opens

<details>
<summary><b>First-run output</b></summary>

```
╔══════════════════════════════════════════════╗
║       Desktop GUI Agent — Startup Check      ║
╚══════════════════════════════════════════════╝

Platform: Linux
  [OK] Linux environment configured

Ollama (LLM):
  [OK] Ollama already running on localhost:11434

vLLM (VLM — UI-TARS):
  [OK] vLLM already running — UI-TARS active

Models:
  [OK] qwen3:8b                       LLM — planning, routing, reflection
  [OK] VLM served by vLLM (UI-TARS)

Starting Desktop GUI Agent...
```

</details>

### Using the GUI

1. **Type** your instruction in the command dock (e.g. `"Open Firefox and go to wikipedia.org"`)
2. **Run** — the window minimises, an always-on-top mission HUD appears, and the agent takes over
3. **Watch** Mission Control: a live timeline of every subtask, step, grounding hit, and verification verdict
4. **Stop** any time — from the HUD, the GUI, or the keyboard kill switch

Other pages: **Agent Sessions** (task history & re-run), **Workflows**,
**Memory** (learned tasks & failure patterns), **Screen History** (frames
recorded during missions), and **Settings**.

---

## Safety

- **Action firewall** — every `type` step is screened by a deterministic
  classifier before execution; destructive shell commands (`rm -rf /`, `mkfs`,
  fork bombs, …) are blocked. It never calls a model, so it is immune to
  prompt injection.
- **Kill switch** — press Esc three times, or slam the mouse into the
  top-left corner, to stop the agent instantly and release all held keys.
- **Wall-clock budgets** — a stuck task aborts (default 10 min/task,
  4 min/subtask) instead of running unbounded.
- **Credential safety** — `{{cred:site:field}}` values live in the OS keyring,
  are redacted from all logs, and are cleared from the clipboard after paste.
- **Keyboard injection** uses `XTest` (Linux) or `win32 SendInput` (Windows) —
  standard OS-level events, same as a real keyboard.
- **Agent window minimises** before executing tasks so the agent never clicks
  its own UI.
- **Stop button** in the GUI interrupts execution after the current step
  completes.
- **Max retries** — each step retries at most 3 times before the task is
  marked failed.

---

## Platform Differences

| Feature | Linux | Windows |
|---------|-------|---------|
| Keyboard backend | **XTest** (Xlib) — injects at X11 server level; reaches GNOME Shell global capture | **pynput** — uses win32 `SendInput`; works with all Windows apps |
| Screenshot backend | **Xlib `get_image`** — focus-neutral, does not dismiss overlays | **PIL `ImageGrab`** — GDI BitBlt |
| Grounding Stage 0 | — | **UIA accessibility tree** (~20–50 ms, exact) |
| App launcher key | `Super` (GNOME Activities) | `Win` (Start Menu) |
| libxcb-cursor | Extracted automatically to `~/.local_xcb` (no sudo) | Not needed |
| Wayland | ❌ X11 session required (`GDK_BACKEND=x11`) | N/A |

---

## Project Structure

```
intel-openvino-desktop-agent/
├── start.py                       ← single entry point (run this)
├── main.py                        ← Qt app + orchestrator wiring
├── config.py                      ← model ids & server settings (single source of truth)
│
├── agents/
│   ├── action/action_agent.py        # ActionExecutionAgent — executes steps
│   ├── grounding/grounding_agent.py  # UIGroundingAgent — text → (x, y), OCR engine
│   ├── planning/planning_agent.py    # PlanningAgent — plans one step at a time
│   ├── reflection/reflection_agent.py# ReflectionAgent — OCR→LLM / VLM verification
│   └── router/router_agent.py        # RouterAgent — decomposes instructions
│
├── core/
│   ├── capture/
│   │   ├── screenshot.py          # Cross-platform screen capture (Xlib/PIL)
│   │   └── screen_snapshot.py     # Foreground/background-aware OCR snapshot
│   ├── executor/burst_executor.py # Fast multi-action sequences (no per-step LLM)
│   ├── grounding/windows_uia.py   # Stage 0: Windows UIA accessibility tree
│   ├── pipeline/ollama_client.py  # Dual-backend client (Ollama LLM + vLLM/Ollama VLM)
│   ├── protocols/a2a.py           # Shared data models + InferenceClient protocol
│   ├── safety/action_firewall.py  # Deterministic destructive-command classifier
│   └── orchestrator.py            # Central coordinator — runs the full loop
│
├── memory/task/task_memory.py     # SQLite task + failure-pattern memory
├── tools/desktop_control/controller.py  # Keyboard/mouse (XTest + pynput) + kill switch
├── utils/                         # Platform detection, clipboard, credentials
├── ui/                            # PyQt6 command-center GUI
├── tests/unit/                    # Unit tests (no backend required)
├── e2e_test.py                    # End-to-end pipeline check
├── requirements.txt
└── run.sh                         # Linux convenience wrapper (sets LD_LIBRARY_PATH)
```

---

<details>
<summary><h2>Manual Setup (without start.py)</h2></summary>

If you prefer to control each step manually:

### Linux

```bash
# Set Qt library path (one-time, no sudo)
apt-get download libxcb-cursor0
dpkg-deb -x libxcb-cursor0_*.deb ~/.local_xcb/
export LD_LIBRARY_PATH="$HOME/.local_xcb/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Pull models
ollama pull qwen3:8b
ollama pull hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q4_K_S

# Launch
source venv/bin/activate
python main.py
```

### Windows

```powershell
# Pull models
ollama pull qwen3:8b
ollama pull hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q4_K_S

# Launch
venv\Scripts\activate
python main.py
```

### Optional: UI-TARS via vLLM (best grounding accuracy)

```bash
pip install vllm
vllm serve ByteDance-Seed/UI-TARS-1.5-7B --port 8000
# The agent auto-detects vLLM on port 8000 and uses it as the VLM backend
```

### Checking GPU usage

```bash
ollama ps
```

```
NAME        SIZE     PROCESSOR    CONTEXT
qwen3:8b    6.4 GB   100% GPU     40960
```

If a model shows `100% CPU`, your GPU may not have enough VRAM.
Try closing other GPU applications and restarting Ollama.

</details>

---

## Running Tests

```bash
source venv/bin/activate        # Linux
# venv\Scripts\activate         # Windows

# Unit tests — fast, no backend or desktop required
pytest

# Lint
ruff check .

# End-to-end pipeline check (requires Ollama running + a live desktop)
python e2e_test.py
```

---

## Performance Reference

Measured on a 6 GB-VRAM GPU with Ollama (models swap as needed):

| Operation | Latency |
|-----------|---------|
| Screen capture (Xlib) | < 20 ms |
| Windows UIA grounding | 20 – 50 ms |
| OCR (RapidOCR) | < 150 ms |
| VLM grounding (UI-TARS) | 1 – 3 s (plus one-time model swap on small VRAM) |
| LLM planning (`qwen3:8b`) | 1 – 3 s |
| Full task (3–5 steps) | 15 s – 60 s |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `qt.qpa.plugin: could not load xcb` | Run `start.py` — it auto-extracts `libxcb-cursor0` |
| `Ollama not running` | Run `ollama serve` in a separate terminal |
| Model on CPU instead of GPU | Check VRAM with `ollama ps`; close other apps; ensure ROCm/CUDA is installed |
| `No JSON array in router response` | Rare LLM format issue; retry the task |
| Agent clicks wrong place | Lower screen scaling or check `DISPLAY` env var points to your active session |
| Wayland session (Linux) | Log out, select "GNOME on Xorg" at login screen, log back in |

---

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, code style, and architecture constraints.

## Acknowledgements

Built on the shoulders of excellent open-source work:
[OpenVINO™](https://github.com/openvinotoolkit/openvino) ·
[Ollama](https://ollama.com) ·
[vLLM](https://github.com/vllm-project/vllm) ·
[UI-TARS](https://github.com/bytedance/UI-TARS) ·
[Qwen](https://github.com/QwenLM) ·
[RapidOCR](https://github.com/RapidAI/RapidOCR) ·
[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

**Google Summer of Code 2026 — Intel® OpenVINO™ Desktop Agent**

</div>
