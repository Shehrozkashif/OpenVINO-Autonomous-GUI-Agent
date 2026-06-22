"""
Live end-to-end test: create a folder on the desktop via the GUI agent pipeline.

Task: "Right click on the desktop, click New, click Folder, type TestFolder, press Enter."

Run:
    python tests/live_folder_test.py

Requirements:
    - OpenVINO Model Server serving qwen3-8b-int4-ov (LLM) and ui-tars-1.5-7b-int4-ov (VLM)
    - Real Windows display (desktop visible, no fullscreen apps)
    - Write permission to %USERPROFILE%\\Desktop
"""
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

# Force UTF-8 output on Windows so print() never raises UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
# Remove default sink; add a timed, colourless sink so every line has a timestamp.
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    level="DEBUG",
    colorize=True,
)

TASK = (
    "Right click on the desktop, click New, click Folder, "
    "type TestFolder, press Enter."
)
def _resolve_desktop() -> Path:
    """Return the real Desktop path, handling OneDrive folder redirection."""
    userprofile = Path(os.environ.get("USERPROFILE", "C:/Users/user"))
    onedrive_desktop = userprofile / "OneDrive" / "Desktop"
    if onedrive_desktop.is_dir():
        return onedrive_desktop
    return userprofile / "Desktop"

DESKTOP = _resolve_desktop()
FOLDER_NAME = "TestFolder"
FOLDER_PATH = DESKTOP / FOLDER_NAME


def phase(label: str):
    print(f"\n{'-'*60}")
    print(f"  PHASE: {label}")
    print(f"{'-'*60}")


def _clean_existing():
    """Remove TestFolder if it already exists so the test is repeatable."""
    if FOLDER_PATH.exists():
        try:
            FOLDER_PATH.rmdir()
            print(f"  [PRE-CLEAN] Removed existing {FOLDER_PATH}")
        except Exception as e:
            print(f"  [PRE-CLEAN] Could not remove existing folder: {e}")
            print("  Please delete it manually and re-run.")
            sys.exit(1)


def main():
    print("\n" + "="*60)
    print("  GUI AGENT — LIVE FOLDER CREATION TEST")
    print("="*60)
    print(f"  Task   : {TASK}")
    print(f"  Desktop: {DESKTOP}")
    print(f"  Target : {FOLDER_PATH}")

    _clean_existing()

    # ── Phase 1: Build orchestrator ────────────────────────────────────────────
    phase("1. Build orchestrator (OVMS health-check + component init)")
    t0 = time.perf_counter()
    try:
        from main import build_orchestrator
        orch = build_orchestrator()
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  Orchestrator ready in {build_ms:.0f} ms")
    except Exception as e:
        print(f"  [FATAL] Could not build orchestrator: {e}")
        print("  Make sure OpenVINO Model Server is running (python start.py) with both models loaded.")
        raise

    # ── Phase 2: Instrument the orchestrator to capture per-phase timing ───────
    phase("2. Instrument for timing capture")

    timings = {
        "router_decompose":    None,
        "burst_detected":      False,
        "burst_time_ms":       None,
        "planning_calls":      0,
        "planning_time_ms":    0.0,
        "grounding_calls":     0,
        "grounding_time_ms":   0.0,
        "reflection_calls":    0,
        "reflection_time_ms":  0.0,
        "total_steps":         0,
    }

    # Wrap planner
    _orig_plan = orch.planner.plan_next_step
    def _timed_plan(*args, **kwargs):
        timings["planning_calls"] += 1
        t = time.perf_counter()
        result = _orig_plan(*args, **kwargs)
        timings["planning_time_ms"] += (time.perf_counter() - t) * 1000
        if result:
            timings["total_steps"] += 1
        return result
    orch.planner.plan_next_step = _timed_plan

    # Wrap grounder
    _orig_ground = orch.grounder.ground
    def _timed_ground(*args, **kwargs):
        timings["grounding_calls"] += 1
        t = time.perf_counter()
        result = _orig_ground(*args, **kwargs)
        timings["grounding_time_ms"] += (time.perf_counter() - t) * 1000
        return result
    orch.grounder.ground = _timed_ground

    # Wrap reflector
    _orig_verify = orch.reflector.verify
    def _timed_reflect(*args, **kwargs):
        timings["reflection_calls"] += 1
        t = time.perf_counter()
        result = _orig_verify(*args, **kwargs)
        timings["reflection_time_ms"] += (time.perf_counter() - t) * 1000
        return result
    orch.reflector.verify = _timed_reflect

    # Wrap burst executor
    _orig_burst_run = orch.burst_executor.run
    def _timed_burst(burst, *args, **kwargs):
        timings["burst_detected"] = True
        t = time.perf_counter()
        result = _orig_burst_run(burst, *args, **kwargs)
        timings["burst_time_ms"] = (time.perf_counter() - t) * 1000
        return result
    orch.burst_executor.run = _timed_burst

    print("  Wrappers installed.")

    # ── Phase 3: Wait 3 s so user can position cursor away from desktop ────────
    phase("3. Countdown (3 s) — move cursor to desktop, close other windows")
    for i in range(3, 0, -1):
        print(f"  Starting in {i}…", flush=True)
        time.sleep(1.0)

    # ── Phase 4: Run the task ──────────────────────────────────────────────────
    phase("4. Execute task")
    t_task_start = time.perf_counter()

    # Capture all log lines emitted during the run into a buffer as well
    log_lines = []
    _orig_log = orch.log
    def _capturing_log(msg):
        log_lines.append(f"[{time.perf_counter() - t_task_start:6.2f}s] {msg}")
        _orig_log(msg)
    orch.log = _capturing_log

    result = orch.execute(TASK)
    t_task_end = time.perf_counter()
    total_s = t_task_end - t_task_start

    # ── Phase 5: Verification ──────────────────────────────────────────────────
    phase("5. Verification")
    time.sleep(0.5)   # let filesystem settle
    folder_exists = FOLDER_PATH.exists() and FOLDER_PATH.is_dir()
    print(f"  Folder '{FOLDER_PATH}' exists: {folder_exists}")

    # Also check for partial matches (folder name mismatch / extra chars)
    desktop_folders = [
        p.name for p in DESKTOP.iterdir()
        if p.is_dir() and "TestFolder" in p.name
    ]
    if desktop_folders:
        print(f"  Matching folders on desktop: {desktop_folders}")

    # ── Phase 6: Report ────────────────────────────────────────────────────────
    phase("6. Full timing report")

    print(f"\n  Task succeeded (orchestrator): {result.get('success')}")
    print(f"  Subtasks completed           : {result.get('subtasks_completed')}")
    print(f"  Total wall time              : {total_s:.2f}s")
    print()
    SEP = "  " + "-"*51
    print(SEP)
    print("  Phase timing breakdown")
    print(SEP)
    print(f"  Burst detected                 : {'YES' if timings['burst_detected'] else 'NO'}")
    if timings["burst_time_ms"] is not None:
        print(f"  Burst execution                : {timings['burst_time_ms']:.0f} ms")
    print(f"  Planning calls                 : {timings['planning_calls']}")
    print(f"  Planning total time            : {timings['planning_time_ms']:.0f} ms")
    if timings["planning_calls"] > 0:
        avg_plan = timings["planning_time_ms"] / timings["planning_calls"]
        print(f"  Planning avg/call              : {avg_plan:.0f} ms")
    print(f"  Grounding calls                : {timings['grounding_calls']}")
    print(f"  Grounding total time           : {timings['grounding_time_ms']:.0f} ms")
    if timings["grounding_calls"] > 0:
        avg_ground = timings["grounding_time_ms"] / timings["grounding_calls"]
        print(f"  Grounding avg/call             : {avg_ground:.0f} ms")
    print(f"  Reflection calls               : {timings['reflection_calls']}")
    print(f"  Reflection total time          : {timings['reflection_time_ms']:.0f} ms")
    if timings["reflection_calls"] > 0:
        avg_ref = timings["reflection_time_ms"] / timings["reflection_calls"]
        print(f"  Reflection avg/call            : {avg_ref:.0f} ms")
    print(f"  Total steps planned+executed   : {timings['total_steps']}")
    print(SEP)

    print(f"\n  ── Orchestrator log ({len(log_lines)} entries) ──")
    for line in log_lines:
        print(f"    {line}")

    print()
    print("="*60)
    status = "PASSED" if folder_exists else "FAILED"
    print(f"  RESULT: {status}")
    if not folder_exists:
        print(f"  Folder '{FOLDER_NAME}' was NOT found on the desktop.")
        print(f"  Orchestrator reported success={result.get('success')}")
    print("="*60)

    return folder_exists


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
