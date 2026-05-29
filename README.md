# Desktop GUI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](#installation)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Ollama](https://img.shields.io/badge/backend-Ollama-orange)](https://ollama.com)

An autonomous desktop agent that accepts natural language instructions, observes your screen, and executes mouse and keyboard actions to complete tasks — fully local, no cloud required.

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
                │  Planning Agent  │  generates atomic action steps
                └──────────────────┘
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Grounding Agent  │    │  Action Agent    │
    │  OCR + VLM  →    │    │  clicks, types,  │
    │  screen (x, y)   │    │  presses keys    │
    └──────────────────┘    └──────────────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │  Reflection Agent    │  VLM verifies outcome
                          └──────────────────────┘
                                     │
                          confirmed? ─▶ next step
                             failed? ─▶ retry / replan
```

### Grounding Pipeline (4 stages, fastest → most robust)

| Stage | Method | Confidence | When it fires |
|-------|--------|-----------|--------------|
| 1 | OCR direct fuzzy-match | 0.95 | Target text is visible on screen |
| 2 | VLM normalized coordinates | ~0.80 | Icon, field, or unlabelled element |
| 3 | VLM → text label → OCR | 0.85 | Semantically described element |
| 4 | VLM 3×3 zone estimation | 0.50 | Last resort fallback |

---

## Models

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| **LLM** | `qwen3:14b` via Ollama | 9 GB | Routing, planning, reflection reasoning |
| **VLM** | `qwen2.5vl-gui` via Ollama | 6 GB | Visual grounding, screen verification |

`qwen2.5vl-gui` is a custom Ollama build of `qwen2.5vl:7b` with a 4096-token context window so it fits in GPU VRAM alongside the LLM. `start.py` creates it automatically on first run.

---

## Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Ubuntu 22.04 / Windows 10 | Ubuntu 24.04 / Windows 11 |
| Python | 3.10 | 3.12 |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 24 GB (both models on GPU) |
| Disk | 20 GB free | 30 GB free |
| Display | X11 (Linux) | X11 or Windows desktop |

---

## Installation

### Linux

```bash
# 1. Clone the repository
git clone https://github.com/your-org/intel-openvino-desktop-agent.git
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
git clone https://github.com/your-org/intel-openvino-desktop-agent.git
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

### One command — works on both platforms

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
2. **Ollama check** — verifies Ollama is running; tries to start it if not
3. **Model check** — confirms `qwen3:14b` and `qwen2.5vl-gui` are pulled;
   downloads them on first run (~15 GB total, one-time only)
4. **Launches** `main.py` — the agent GUI opens

### First-run output

```
╔══════════════════════════════════════════════╗
║       Desktop GUI Agent — Startup Check      ║
╚══════════════════════════════════════════════╝

Platform: Linux
  [OK] Linux environment configured

Ollama:
  [OK] Ollama is running on localhost:11434

Models:
  [OK] qwen3:14b            LLM  — planning, routing, reflection
  [OK] qwen2.5vl-gui        VLM  — visual grounding & verification

Starting Desktop GUI Agent...
```

### Command-line options

```bash
# Pre-fill the instruction box
python start.py --prompt "Open VS Code and enable autosave"

# Pre-fill and run immediately on startup
python start.py --prompt "Search for OpenVINO documentation" --auto-run
```

---

## Using the GUI

1. **Type** your instruction in the text box (e.g. `"Open Firefox and go to wikipedia.org"`)
2. **Click Run Task** — the agent window minimises and the agent starts working
3. **Watch** the Agent Log panel for real-time step-by-step output
4. **Click Stop** at any time to interrupt

The **Live Screen** panel shows a 1 FPS preview of what the agent is seeing.
The **History** tab records all completed tasks.
The **Settings** tab lets you change the inference device (`AUTO / GPU / CPU`).

---

## Platform Differences

| Feature | Linux | Windows |
|---------|-------|---------|
| Keyboard backend | **XTest** (Xlib) — injects at X11 server level; reaches GNOME Shell global capture | **pynput** — uses win32 `SendInput`; works with all Windows apps |
| Screenshot backend | **Xlib `get_image`** — focus-neutral, does not dismiss overlays | **PIL `ImageGrab`** — GDI BitBlt |
| App launcher key | `Super` (GNOME Activities) | `Win` (Start Menu) |
| libxcb-cursor | Extracted automatically to `~/.local_xcb` (no sudo) | Not needed |
| Wayland | ❌ X11 session required (`GDK_BACKEND=x11`) | N/A |

---

## Project Structure

```
intel-openvino-desktop-agent/
├── start.py                     ← single entry point (run this)
├── main.py                      ← Qt app + orchestrator wiring
│
├── agents/
│   ├── action/                  # ActionExecutionAgent — executes steps
│   ├── grounding/               # UIGroundingAgent — text → (x, y)
│   ├── planning/                # PlanningAgent — generates action steps
│   ├── reflection/              # ReflectionAgent — VLM verifies each action
│   └── router/                  # RouterAgent — decomposes instructions
│
├── core/
│   ├── capture/screenshot.py    # Cross-platform screen capture (Xlib/PIL)
│   ├── grounding/
│   │   ├── hybrid_grounding.py  # 4-stage grounding engine
│   │   ├── ocr_engine.py        # RapidOCR wrapper
│   │   └── som_engine.py        # Set-of-Marks (available, not in pipeline)
│   ├── pipeline/
│   │   ├── ollama_client.py     # Dual-model Ollama client (LLM + VLM)
│   │   ├── ovms_client.py       # OpenVINO Model Server client
│   │   └── direct_client.py     # Direct OpenVINO inference
│   ├── protocols/a2a.py         # Shared data models + InferenceClient protocol
│   └── orchestrator.py          # Central coordinator — runs the full loop
│
├── memory/
│   ├── interaction/             # App-specific learned shortcuts
│   ├── screen/                  # Recent screen frame history
│   └── task/                    # SQLite task history
│
├── tools/desktop_control/
│   ├── controller.py            # Cross-platform keyboard/mouse (XTest + pynput)
│   └── server.py                # Optional FastAPI tool server (port 8015)
│
├── ui/main_window.py            # PyQt6 GUI
├── scripts/setup/               # Model download helpers
├── tests/                       # Unit tests
├── requirements.txt
└── run.sh                       # Linux convenience wrapper (sets LD_LIBRARY_PATH)
```

---

## Manual Setup (without start.py)

If you prefer to control each step manually:

### Linux

```bash
# Set Qt library path (one-time, no sudo)
apt-get download libxcb-cursor0
dpkg-deb -x libxcb-cursor0_*.deb ~/.local_xcb/
export LD_LIBRARY_PATH="$HOME/.local_xcb/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Pull models
ollama pull qwen3:14b
ollama pull qwen2.5vl:7b

# Create GPU-friendly VLM variant (fits in VRAM, 4096-token context)
printf 'FROM qwen2.5vl:7b\nPARAMETER num_ctx 4096\n' | ollama create qwen2.5vl-gui -f -

# Launch
source venv/bin/activate
python main.py
```

### Windows

```powershell
# Pull models
ollama pull qwen3:14b
ollama pull qwen2.5vl:7b

# Create GPU-friendly VLM variant
$modelfile = "FROM qwen2.5vl:7b`nPARAMETER num_ctx 4096"
$modelfile | Out-File -FilePath "$env:TEMP\qwen2.5vl-gui.Modelfile" -Encoding utf8
ollama create qwen2.5vl-gui -f "$env:TEMP\qwen2.5vl-gui.Modelfile"

# Launch
venv\Scripts\activate
python main.py
```

---

## Checking GPU Usage

```bash
# Verify both models loaded on GPU (not CPU)
ollama ps
```

Expected output:

```
NAME                    SIZE     PROCESSOR    CONTEXT
qwen3:14b               16 GB    100% GPU     40960
qwen2.5vl-gui:latest    14 GB    100% GPU     4096
```

If either model shows `100% CPU`, your GPU may not have enough VRAM.
Try closing other GPU applications and restarting Ollama.

---

## Running Tests

```bash
source venv/bin/activate        # Linux
# venv\Scripts\activate         # Windows

# Unit tests
PYTHONPATH=. pytest tests/ -v

# End-to-end pipeline check (requires Ollama running)
PYTHONPATH=. python e2e_test.py
```

---

## Performance Reference

Measured on AMD Radeon RX 7900 XTX (24 GB VRAM) with Ollama + ROCm:

| Operation | Latency |
|-----------|---------|
| Screen capture (Xlib) | < 20 ms |
| OCR (RapidOCR) | < 150 ms |
| VLM grounding `qwen2.5vl-gui` | 650 ms – 1.5 s |
| LLM planning `qwen3:14b` | 1 s – 3 s |
| Full task (3–5 steps) | 15 s – 60 s |

---

## Safety

- **Keyboard injection** uses `XTest` (Linux) or `win32 SendInput` (Windows) — standard OS-level events, same as a real keyboard.
- **Tool Server** (`server.py`) binds to `127.0.0.1` only — never accessible from the network.
- **Agent window minimises** before executing tasks so the agent never clicks its own UI.
- **Stop button** in the GUI interrupts execution after the current step completes.
- **Max retries** — each step retries at most 3 times before the task is marked failed.

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

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

*GSoC 2026 — Intel OpenVINO Desktop Agent*
