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

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt       # runtime
pip install -r requirements-dev.txt   # pytest + ruff (to run the test suite)
```

### 4. Prepare the models (only needed for live testing and real runs)

Model ids live in `config.py` — the single source of truth. `start.py` pulls the
LLM and converts UI-TARS into the OpenVINO Model Server repository, then launches
OVMS serving both on port 8000:

```bash
# The one-time conversion toolchain lives in requirements-export.txt:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements-export.txt
# Install the native ovms/ovms.exe binary and set OVMS_DIR (see README.md).
python start.py                              # prepares models + starts OVMS + UI
```

> **First run takes 30–60 minutes** for the UI-TARS INT8 conversion. Subsequent
> runs skip this step. If conversion produces files with broken permissions on
> Windows, delete the model folder from an elevated terminal and re-run `start.py`.

### 5. Run tests

```bash
pytest                                 # 453 unit tests — no backend or desktop needed
python tests/live/test_usecases.py     # real desktop — requires OVMS + Windows
```

The unit suite (`tests/unit/`) must pass on a machine with no model server,
no GPU and **no display**; anything that needs a live backend or a real screen
belongs in `tests/live/` instead. The no-display part is easy to break by
accident: `desktop/capture._screen_size()` once fell back to a real screen grab
and raised `X connection failed` on CI, taking `TaskOrchestrator.__init__` with
it — green on every developer desktop, red on every runner. Anything in
`desktop/` that a constructor calls has to degrade instead of raise.

### Run the CI checks before pushing

```bash
pre-commit install --install-hooks        # ruff + hygiene on every commit
pre-commit install --hook-type pre-push   # unit suite before each push
pre-commit run --all-files                # check everything right now
```

Build test doubles from `tests/unit/conftest.py` (`make_grounder`,
`make_reflector`, `make_llm`, `make_memory`) rather than bare `MagicMock`s. A
raw mock returns a mock from `min_confidence`, and comparing that to a float
raises inside the orchestrator — a failure that looks like a real bug but is
only a badly built double.

---

## Code Style

- Follow the **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**;
  lint with `ruff check .` (rules configured in `pyproject.toml`: pyflakes,
  import order, modern typing, bugbear, comprehensions, and Google-convention
  docstrings). It must pass clean before you submit.
- Use **type hints** on all public function signatures; prefer the modern forms
  (`list`/`dict`/`tuple` over `typing.List`/`Dict`/`Tuple`, and `X | None` over
  `Optional[X]`).
- Agent constructors must accept `InferenceClient` (the Protocol in
  `core/inference.py`), not a concrete client class.
- Heavy or optional dependencies may be imported lazily inside functions
  (e.g. `sentence_transformers`, `ctypes` Windows calls) — everything else is
  imported at module top.
- No inline `time.sleep()` on the main Qt thread — use `QTimer.singleShot`.
- Log with `loguru` (`from loguru import logger`), not `print`.

---

## Architecture Constraints

| Rule | Reason |
|------|--------|
| Dependencies point one way: `ui → core → agents → desktop` | Each layer stays testable and replaceable on its own |
| `desktop/` reports facts and never decides | Policy built on those facts lives in `core/` and is unit-testable without Windows |
| New "did it work?" checks belong in `core/groundtruth.py`, not inline in the loop | Ground truth beats model judgment — keeping the checks together is what makes that principle enforceable |
| All OS input goes through `desktop/input.py` | Single place for keyboard/mouse injection (raw Win32 SendInput via ctypes) and the kill switch |
| Grounding coordinates are always physical screen pixels | Capture returns physical pixels; the controller expects physical pixels |
| Agents depend on the `InferenceClient` Protocol, never on `OVMSClient` directly | Keeps the inference backend (OVMS today, anything else tomorrow) drop-in replaceable |
| `type` steps must pass the action firewall (`core/firewall.py`) | Deterministic protection against destructive shell commands |
| Every prompt lives in `agents/prompts.py` | The prompt is behaviour; keeping them together makes behaviour reviewable |
| Orchestrator log strings are an interface — `ui/events.py` parses them | Changing a log format silently changes the mission timeline |
| New budgets/limits go in `core/runstate.py` | Nothing may loop unbounded, and every bound must be findable in one place |
| Nothing in the loop may read `core/history.py` back | The agent must decide from the live screen. Feeding past runs into planning changes behaviour based on state the user cannot see — it once replaced a correct decomposition with a stale plan |

---

## Submitting Changes

1. Ensure `pytest` passes and `ruff check .` is clean (CI runs both on every
   push — see `.github/workflows/ci.yml`).
2. Update docstrings, `README.md` and `ARCHITECTURE.md` if you change any
   public API, module layout or workflow.
3. Open a pull request against `main` with a clear description of what changed
   and why.

---

## Reporting Issues

Open a GitHub issue with:
- OS and Python version
- Target device in use (`TARGET_DEVICE` in `config.py` — GPU / CPU / NPU)
- The instruction that failed
- The full log output from the Agent Log panel
