# Intel® OpenVINO™ Desktop Agent

[![OpenVINO](https://img.shields.io/badge/Powered%20By-OpenVINO-blue)](https://docs.openvino.ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)](https://github.com)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A privacy-first, local desktop automation agent powered by Vision-Language Models and Intel® OpenVINO™.
The agent sees your screen, reasons about it, and executes mouse/keyboard actions — no cloud required.

---

## How It Works

The agent runs a continuous **See → Think → Act → Verify** loop:

```
User Instruction
      │
      ▼
┌─────────────┐     decomposes     ┌──────────────────────┐
│ Router Agent│ ──────────────────▶│  Sub-tasks (ordered) │
└─────────────┘                    └──────────────────────┘
                                              │
                              ┌───────────────┘
                              │  for each sub-task
                              ▼
                    ┌──────────────────┐
                    │  Planning Agent  │  generates action steps
                    └──────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
          ┌──────────────────┐  ┌──────────────────┐
          │ Grounding Agent  │  │  Action Agent    │
          │  OCR + VLM →     │  │  clicks, types,  │
          │  screen (x, y)   │  │  presses keys    │
          └──────────────────┘  └──────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │  Reflection Agent    │  VLM verifies result
                              └──────────────────────┘
```

### Grounding Pipeline (4 stages, fastest first)

| Stage | Method | Confidence | When it fires |
|-------|--------|-----------|--------------|
| 1 | OCR direct fuzzy-match | 0.95 | Target text is visible on screen |
| 2 | VLM → text label → OCR | 0.85 | Target described semantically |
| 3 | VLM normalized coordinates | ~0.65 | Icon or unlabelled element |
| 4 | VLM 3×3 zone estimation | 0.50 | All other stages failed |

---

## Backends

| Backend | Best for | Models used |
|---------|----------|-------------|
| **Ollama** (default) | Development; any GPU via Ollama | `llama3.1:8b` (LLM) + `qwen2.5vl:3b` (VLM) |
| **Direct OpenVINO** | Intel AI PC / Arc GPU / NPU | DeepSeek-R1-Distill-Qwen-7B-int4-ov + Qwen2.5-VL-7B-Instruct-int4-ov |
| **OVMS** | Docker/server deployments | Same models via OpenVINO Model Server |

The app auto-detects the best available backend on startup (Ollama → OVMS → Direct OpenVINO).

---

## Requirements

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Windows 10 / Ubuntu 22.04 | Windows 11 / Ubuntu 24.04 |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB |
| GPU | Any (CPU fallback works) | Intel® Arc™ or NVIDIA RTX |
| Display session | X11 (Linux) | X11 |

> **Linux only:** pyautogui requires an X11 session. If you use Wayland, add
> `export GDK_BACKEND=x11` to your `~/.bashrc` and log out / back in.

---

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/your-org/intel-openvino-desktop-agent.git
cd intel-openvino-desktop-agent

# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Choose a backend and pull models

#### Option A — Ollama (recommended for development)

Install [Ollama](https://ollama.com) for your platform, then:

```bash
ollama pull llama3.1:8b     # LLM for planning / routing
ollama pull qwen2.5vl:3b    # VLM for visual grounding
```

#### Option B — Direct OpenVINO (Intel AI PC / Arc GPU)

```bash
python scripts/setup/pull_models.py   # downloads ~12 GB to models/OpenVINO/
```

### 4. Start the Tool Server

The Tool Server handles all mouse/keyboard actions and must run in a separate terminal.

**Linux:**
```bash
python -m tools.desktop_control.server
```

**Windows (PowerShell):**
```powershell
python -m tools.desktop_control.server
```

Verify it is running:
```bash
curl http://127.0.0.1:8015/health
# → {"status":"ok"}
```

### 5. Launch the agent

```bash
python main.py
```

Select your backend in **Settings → Pipeline Mode**, type an instruction, and click **Run Task**.

---

## One-command dev startup (Linux)

```bash
bash scripts/setup/start_dev.sh
```

This script checks for Ollama, pulls missing models, starts the Tool Server, and prints the launch command.

**Windows** — open two PowerShell windows:

```powershell
# Terminal 1 — Tool Server
python -m tools.desktop_control.server

# Terminal 2 — Agent UI
python main.py
```

---

## Command-line options

```bash
python main.py --prompt "Open VS Code and enable autosave"   # pre-fill instruction
python main.py --prompt "Search for OpenVINO" --auto-run     # run immediately on startup
```

---

## Project Structure

```
intel-openvino-desktop-agent/
├── agents/
│   ├── action/         # ActionExecutionAgent — translates steps to tool calls
│   ├── grounding/      # UIGroundingAgent — natural language → screen (x, y)
│   ├── planning/       # PlanningAgent — generates step sequences
│   ├── reflection/     # ReflectionAgent — VLM verifies each action succeeded
│   └── router/         # RouterAgent — decomposes instructions into sub-tasks
│
├── core/
│   ├── capture/        # ScreenCapture — mss-based screenshot + pHash change detection
│   ├── grounding/      # HybridGroundingEngine, OCREngine, SoMEngine
│   ├── pipeline/       # OllamaClient, DirectOpenVINOClient, OVMSClient, OptimizedPipeline
│   ├── protocols/      # Shared data models (SubTask, ActionStep) + InferenceClient Protocol
│   └── orchestrator.py # TaskOrchestrator — central coordinator
│
├── memory/
│   ├── interaction/    # App-specific shortcut knowledge
│   ├── screen/         # Sliding window of recent screen frames
│   └── task/           # SQLite-backed task history with semantic search
│
├── models/             # Downloaded OpenVINO model files (git-ignored)
│
├── scripts/
│   └── setup/
│       ├── pull_models.py   # Cross-platform model downloader
│       ├── start_dev.sh     # Linux dev startup script
│       └── start_dev.bat    # Windows dev startup script
│
├── tools/
│   └── desktop_control/
│       ├── controller.py    # HTTP client → Tool Server
│       └── server.py        # FastAPI server on port 8015 (pyautogui wrapper)
│
├── ui/
│   └── main_window.py       # PyQt6 GUI (Agent Control / History / Settings tabs)
│
├── e2e_test.py              # End-to-end pipeline verification
├── main.py                  # Application entry point
└── requirements.txt
```

---

## Running Tests

```bash
PYTHONPATH=. pytest                    # unit tests
PYTHONPATH=. python e2e_test.py        # end-to-end pipeline check (requires running backend)
```

---

## Performance Reference

Measured on Intel® Core™ Ultra 7 with Arc™ GPU (Direct OpenVINO backend):

| Operation | Latency |
|-----------|---------|
| Screen capture + pHash | < 50 ms |
| OCR (RapidOCR) | < 100 ms |
| VLM grounding (Qwen2.5-VL-7B-int4) | ~1–3 s |
| LLM planning (DeepSeek-R1-7B-int4) | ~1–2 s |

On a development machine without Intel GPU (Ollama, CPU VLM):
VLM calls take ~90–160 s. OCR-grounded tasks remain fast.

---

## Safety

- The Tool Server runs on `127.0.0.1` only — not accessible from the network.
- `pyautogui.FAILSAFE = True` — moving the mouse to the top-left corner stops the agent immediately.
- The agent window minimises before executing tasks to avoid the agent clicking its own UI.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

*GSoC 2026 project — Intel OpenVINO ecosystem.*
