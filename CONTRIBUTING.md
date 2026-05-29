# Contributing

Thank you for considering a contribution to the Intel® OpenVINO™ Desktop Agent.

---

## Development Setup

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/intel-openvino-desktop-agent.git
cd intel-openvino-desktop-agent
git checkout -b my-feature
```

### 2. Create a virtual environment

```bash
# Linux
python -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull a lightweight backend for testing

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5vl:3b
```

### 5. Start the Tool Server

```bash
# Linux / macOS
python -m tools.desktop_control.server

# Windows (PowerShell)
python -m tools.desktop_control.server
```

### 6. Run tests

```bash
PYTHONPATH=. pytest                  # unit tests
PYTHONPATH=. python e2e_test.py      # end-to-end (requires running backend + tool server)
```

---

## Code Style

- Follow **PEP 8**.
- Use **type hints** on all public function signatures.
- Agent constructors must accept `InferenceClient` (the Protocol in `core/protocols/a2a.py`), not a concrete class.
- Imports at module top — no `import` statements inside function bodies.
- No inline `time.sleep()` on the main Qt thread — use `QTimer.singleShot`.
- Log with `loguru` (`from loguru import logger`), not `print`.

---

## Architecture Constraints

| Rule | Reason |
|------|--------|
| Agents never import `pyautogui` directly | All OS interaction goes through the Tool Server |
| Grounding coordinates are always in screen pixels (not display pixels) | mss captures physical pixels; controller expects physical pixels |
| `DirectOpenVINOClient` must load both `self.llm` and `self.vlm_pipeline` in `__init__` | Lazy loading causes silent `AttributeError` at query time |
| `InferenceClient` Protocol must stay in sync with all three backend classes | OVMSClient, OllamaClient, DirectOpenVINOClient are all valid backends |

---

## Submitting Changes

1. Ensure `pytest` passes and `e2e_test.py` runs without errors.
2. Update docstrings and README if you change any public API.
3. Open a pull request against `main` with a clear description of what changed and why.

---

## Reporting Issues

Open a GitHub issue with:
- OS and Python version
- Which backend (Ollama / Direct OpenVINO / OVMS)
- The instruction that failed
- The full log output from the Agent Log panel
