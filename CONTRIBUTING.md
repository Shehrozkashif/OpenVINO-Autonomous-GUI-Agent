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

### 4. Prepare the models (only needed for live/e2e testing)

Model ids live in `config.py` — the single source of truth. `start.py` pulls the
LLM and converts UI-TARS into the OpenVINO Model Server repository, then launches
OVMS serving both on port 8000:

```bash
# The conversion toolchain (optimum-intel, nncf) is included in requirements.txt.
# On Windows: install the native ovms.exe and set OVMS_DIR (see README.md).
# On Linux:   docker pull openvino/model_server:latest-gpu
python start.py                              # prepares models + starts OVMS + UI
```

> **First run takes 30–60 minutes** for the UI-TARS INT4 conversion. Subsequent
> runs skip this step. If conversion produces files with broken permissions on
> Windows, delete the model folder from an elevated terminal and re-run `start.py`.

### 5. Run tests

```bash
pytest                    # unit tests — fast, no backend or desktop required
python e2e_test.py        # end-to-end — requires OVMS running + a live desktop
```

The unit suite must pass on a machine with no model server and no GPU; anything
that needs a live backend belongs in `e2e_test.py` or `tests/` (live tests), not
`tests/unit/`.

---

## Code Style

- Follow **PEP 8**; lint with `ruff check .` (configured in `pyproject.toml`).
- Use **type hints** on all public function signatures.
- Agent constructors must accept `InferenceClient` (the Protocol in
  `core/protocols/a2a.py`), not a concrete client class.
- Heavy or optional dependencies may be imported lazily inside functions
  (e.g. `sentence_transformers`, `ctypes` Windows calls) — everything else is
  imported at module top.
- No inline `time.sleep()` on the main Qt thread — use `QTimer.singleShot`.
- Log with `loguru` (`from loguru import logger`), not `print`.

---

## Architecture Constraints

| Rule | Reason |
|------|--------|
| All OS input goes through `tools/desktop_control/controller.py` | Single place for platform differences (XTest vs pynput) and the kill switch |
| Grounding coordinates are always physical screen pixels | Capture returns physical pixels; the controller expects physical pixels |
| Agents depend on the `InferenceClient` Protocol, never on `OVMSClient` directly | Keeps the inference backend (OVMS today, anything else tomorrow) drop-in replaceable |
| `type` steps must pass the action firewall (`core/safety/action_firewall.py`) | Deterministic protection against destructive shell commands |
| Tasks completed via degraded paths must not be stored in success memory | Broken plans would otherwise poison future routing hints |

---

## Submitting Changes

1. Ensure `pytest` passes and `ruff check .` is clean.
2. Update docstrings, `README.md`, and `CLAUDE.md` if you change any public API
   or workflow.
3. Open a pull request against `main` with a clear description of what changed
   and why.

---

## Reporting Issues

Open a GitHub issue with:
- OS and Python version
- Target device in use (`TARGET_DEVICE` in `config.py` — GPU / CPU / NPU)
- The instruction that failed
- The full log output from the Agent Log panel
