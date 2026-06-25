"""Live end-to-end use-case test suite.

Runs real tasks through the full orchestrator pipeline (router → burst/planner →
grounder → actor → reflector) and verifies each outcome independently.

Use cases tested
----------------
1. Create a folder on the desktop          (burst path, no LLM)
2. Open Notepad                            (LLM planning path)
3. Open Calculator                         (LLM planning path)
4. Open Windows Terminal                   (LLM planning path)
5. Open Notepad and type + save a phrase   (multi-step LLM path)

Run:
    python tests/live_usecases_test.py

Requirements:
    - OpenVINO Model Server running: python start.py
    - qwen3-8b-int4-ov         served (LLM)
    - ui-tars-1.5-7b-int4-ov   served (VLM)
    - Real Windows display, desktop visible, no fullscreen app
"""
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, ".")

# Force UTF-8 so Windows console never raises UnicodeEncodeError
import io as _io

if hasattr(sys.stdout, "buffer") and (not sys.stdout.encoding or sys.stdout.encoding.lower() not in ("utf-8","utf8")):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and (not sys.stderr.encoding or sys.stderr.encoding.lower() not in ("utf-8","utf8")):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _desktop() -> Path:
    up = Path(os.environ.get("USERPROFILE", "C:/Users/user"))
    od = up / "OneDrive" / "Desktop"
    return od if od.is_dir() else up / "Desktop"


def _proc_running(exe: str) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return exe.lower() in out.lower()
    except Exception:
        return False


def _kill(exe: str):
    try:
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True, timeout=5)
        time.sleep(0.5)
    except Exception:
        pass


@dataclass
class UCResult:
    name: str
    passed: bool
    elapsed_s: float
    burst_used: bool = False
    planning_calls: int = 0
    grounding_calls: int = 0
    reflection_calls: int = 0
    notes: str = ""


# ── test runner ────────────────────────────────────────────────────────────────

class LiveUseCaseTester:
    def __init__(self):
        from main import build_orchestrator
        self.orch = build_orchestrator()
        self.desktop = _desktop()
        self.results: list[UCResult] = []

    def _instrument(self):
        """Attach lightweight timing wrappers. Returns a stats dict (reset each call)."""
        stats = {"burst": False, "plan": 0, "ground": 0, "reflect": 0}

        _orig_plan = self.orch.planner.plan_next_step
        def _p(*a, **kw):
            stats["plan"] += 1
            return _orig_plan(*a, **kw)
        self.orch.planner.plan_next_step = _p

        _orig_ground = self.orch.grounder.ground
        def _g(*a, **kw):
            stats["ground"] += 1
            return _orig_ground(*a, **kw)
        self.orch.grounder.ground = _g

        _orig_verify = self.orch.reflector.verify
        def _r(*a, **kw):
            stats["reflect"] += 1
            return _orig_verify(*a, **kw)
        self.orch.reflector.verify = _r

        _orig_burst = self.orch.burst_executor.run
        def _b(burst, *a, **kw):
            stats["burst"] = True
            return _orig_burst(burst, *a, **kw)
        self.orch.burst_executor.run = _b

        return stats

    def _run(self, name: str, task: str, verify_fn, cleanup_fn=None, pre_fn=None) -> UCResult:
        print(f"\n{'='*62}")
        print(f"  USE CASE: {name}")
        print(f"  Task    : {task}")
        print(f"{'='*62}")

        if pre_fn:
            pre_fn()

        stats = self._instrument()
        t0 = time.perf_counter()
        try:
            self.orch.execute(task)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  [EXCEPTION] {e}")
            return UCResult(name=name, passed=False, elapsed_s=elapsed, notes=str(e))

        elapsed = time.perf_counter() - t0

        time.sleep(1.0)  # let OS settle before verification
        passed, notes = verify_fn()

        if cleanup_fn:
            try:
                cleanup_fn()
            except Exception as e:
                print(f"  [CLEANUP WARN] {e}")

        r = UCResult(
            name=name,
            passed=passed,
            elapsed_s=elapsed,
            burst_used=stats["burst"],
            planning_calls=stats["plan"],
            grounding_calls=stats["ground"],
            reflection_calls=stats["reflect"],
            notes=notes,
        )
        status = "PASSED" if passed else "FAILED"
        print(f"\n  Result : {status} in {elapsed:.1f}s")
        print(f"  Burst  : {'yes' if stats['burst'] else 'no'}  |  "
              f"Plan calls: {stats['plan']}  |  "
              f"Ground calls: {stats['ground']}  |  "
              f"Reflect calls: {stats['reflect']}")
        if notes:
            print(f"  Notes  : {notes}")
        self.results.append(r)
        return r

    # ── use cases ──────────────────────────────────────────────────────────────

    def uc1_create_folder(self):
        folder = self.desktop / "LiveTestFolder"

        def pre():
            if folder.exists():
                try:
                    folder.rmdir()
                except OSError:
                    pass

        def verify():
            ok = folder.exists() and folder.is_dir()
            # Also accept any name containing LiveTestFolder (rename box quirks)
            if not ok:
                matches = [p.name for p in self.desktop.iterdir()
                           if p.is_dir() and "LiveTestFolder" in p.name]
                if matches:
                    return True, f"found as: {matches}"
            return ok, "" if ok else "folder not found on desktop"

        def cleanup():
            for p in self.desktop.iterdir():
                if p.is_dir() and "LiveTestFolder" in p.name:
                    try:
                        p.rmdir()
                    except OSError:
                        pass

        return self._run(
            "1. Create folder on desktop",
            "Right click on the desktop, click New, click Folder, type LiveTestFolder, press Enter.",
            verify, cleanup, pre,
        )

    def uc2_open_notepad(self):
        def pre():
            _kill("notepad.exe")

        def verify():
            ok = _proc_running("notepad.exe")
            return ok, "" if ok else "notepad.exe not found in process list"

        def cleanup():
            _kill("notepad.exe")

        return self._run(
            "2. Open Notepad",
            "open Notepad",
            verify, cleanup, pre,
        )

    def uc3_open_calculator(self):
        def pre():
            _kill("CalculatorApp.exe")

        def verify():
            ok = _proc_running("CalculatorApp.exe")
            return ok, "" if ok else "CalculatorApp.exe not found in process list"

        def cleanup():
            _kill("CalculatorApp.exe")

        return self._run(
            "3. Open Calculator",
            "open Calculator",
            verify, cleanup, pre,
        )

    def uc4_open_terminal(self):
        def pre():
            _kill("WindowsTerminal.exe")

        def verify():
            ok = _proc_running("WindowsTerminal.exe")
            return ok, "" if ok else "WindowsTerminal.exe not found in process list"

        def cleanup():
            _kill("WindowsTerminal.exe")

        return self._run(
            "4. Open Windows Terminal",
            "open Windows Terminal",
            verify, cleanup, pre,
        )

    def uc5_notepad_type_save(self):
        save_path = self.desktop / "agent_test_note.txt"

        def pre():
            _kill("notepad.exe")
            if save_path.exists():
                save_path.unlink()

        def verify():
            # Notepad must be running AND the file must exist on the desktop
            notepad_ok = _proc_running("notepad.exe")
            file_ok = save_path.exists()
            if notepad_ok and file_ok:
                return True, f"notepad running + file saved at {save_path}"
            notes = []
            if not notepad_ok:
                notes.append("notepad not running")
            if not file_ok:
                notes.append(f"file '{save_path.name}' not on desktop")
            return False, "; ".join(notes)

        def cleanup():
            _kill("notepad.exe")
            try:
                if save_path.exists():
                    save_path.unlink()
            except OSError:
                pass

        task = (
            "Open Notepad, then type the text 'Hello from the GUI Agent', "
            "then save the file as agent_test_note.txt on the Desktop."
        )
        return self._run(
            "5. Notepad: open, type, save file",
            task,
            verify, cleanup, pre,
        )

    # ── report ─────────────────────────────────────────────────────────────────

    def print_report(self):
        print(f"\n\n{'='*62}")
        print("  FINAL REPORT")
        print(f"{'='*62}")
        passed = sum(1 for r in self.results if r.passed)
        total  = len(self.results)
        print(f"  {passed}/{total} use cases PASSED\n")

        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name}")
            print(f"         {r.elapsed_s:.1f}s  |  burst={'yes' if r.burst_used else 'no '}  |  "
                  f"plan={r.planning_calls}  ground={r.grounding_calls}  reflect={r.reflection_calls}")
            if r.notes:
                print(f"         note: {r.notes}")

        print(f"\n{'='*62}")
        total_time = sum(r.elapsed_s for r in self.results)
        print(f"  Total wall time: {total_time:.1f}s")
        print(f"{'='*62}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  GUI AGENT — LIVE USE CASE TEST SUITE")
    print("="*62)
    print("  Building orchestrator…")

    try:
        tester = LiveUseCaseTester()
    except Exception as e:
        print(f"\n[FATAL] Could not build orchestrator: {e}")
        print("  Make sure OpenVINO Model Server is running (python start.py).")
        sys.exit(1)

    print(f"  Desktop  : {tester.desktop}")
    print("\n  Countdown: 4 seconds — make sure the desktop is visible.")
    for i in range(4, 0, -1):
        print(f"  {i}…", flush=True)
        time.sleep(1.0)

    # Run all use cases in sequence
    tester.uc1_create_folder()
    time.sleep(2.0)

    tester.uc2_open_notepad()
    time.sleep(2.0)

    tester.uc3_open_calculator()
    time.sleep(2.0)

    tester.uc4_open_terminal()
    time.sleep(2.0)

    tester.uc5_notepad_type_save()

    tester.print_report()

    passed = sum(1 for r in tester.results if r.passed)
    sys.exit(0 if passed == len(tester.results) else 1)


if __name__ == "__main__":
    main()
