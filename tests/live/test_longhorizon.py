"""Live long-horizon test suite — multi-subtask tasks with ground-truth verification.

Exercises the long-task machinery end to end on a real desktop:
  adaptive wall-clock budgets, per-subtask checkpointing, task-level
  replanning, and the UIA structured-control actions (set_value / select).

Use cases
---------
1. Long chain (6+ subtasks): create, transform, and verify files across
   Notepad and the terminal — every effect checked on disk.
2. Form filling via UIA structured actions: drive a native dialog's
   fields deterministically.
3. Zoom meeting smoke test — runs ONLY when Zoom is installed
   (set LIVE_ZOOM=1 to enable; it opens the schedule form, fills it, and
   verifies the meeting exists in the Zoom client's meeting list).

Run:
    python tests/live/test_longhorizon.py

Requirements:
    - OpenVINO Model Server running: python start.py
    - Real Windows display, desktop visible, no fullscreen app
"""
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Force UTF-8 so Windows console never raises UnicodeEncodeError
import io as _io

if hasattr(sys.stdout, "buffer") and (not sys.stdout.encoding or sys.stdout.encoding.lower() not in ("utf-8", "utf8")):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and (not sys.stderr.encoding or sys.stderr.encoding.lower() not in ("utf-8", "utf8")):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)


def _desktop() -> Path:
    up = Path(os.environ.get("USERPROFILE", "C:/Users/user"))
    od = up / "OneDrive" / "Desktop"
    return od if od.is_dir() else up / "Desktop"


def _kill(exe: str):
    try:
        subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True, timeout=5)
        time.sleep(0.5)
    except Exception:
        pass


def _zoom_installed() -> bool:
    candidates = [
        Path(os.environ.get("APPDATA", "")) / "Zoom" / "bin" / "Zoom.exe",
        Path("C:/Program Files/Zoom/bin/Zoom.exe"),
        Path("C:/Program Files (x86)/Zoom/bin/Zoom.exe"),
    ]
    return any(p.is_file() for p in candidates) or bool(shutil.which("Zoom.exe"))


@dataclass
class LHResult:
    name: str
    passed: bool
    elapsed_s: float
    subtasks_completed: int = 0
    replans: int = 0
    notes: str = ""


class LongHorizonTester:
    def __init__(self):
        from main import build_orchestrator
        self.orch = build_orchestrator()
        self.desktop = _desktop()
        self.results: list[LHResult] = []

    def _run(self, name: str, task: str, verify_fn, cleanup_fn=None, pre_fn=None):
        print(f"\n{'=' * 62}\n  LONG-HORIZON: {name}\n  Task: {task}\n{'=' * 62}")
        if pre_fn:
            pre_fn()

        replans = [0]
        _orig_replan = self.orch.router.replan
        def _rp(*a, **kw):
            replans[0] += 1
            return _orig_replan(*a, **kw)
        self.orch.router.replan = _rp

        t0 = time.perf_counter()
        try:
            result = self.orch.execute(task)
        except Exception as e:
            self.results.append(LHResult(
                name=name, passed=False,
                elapsed_s=time.perf_counter() - t0, notes=str(e),
            ))
            print(f"  [EXCEPTION] {e}")
            return
        elapsed = time.perf_counter() - t0

        time.sleep(1.0)
        passed, notes = verify_fn()
        if cleanup_fn:
            try:
                cleanup_fn()
            except Exception as e:
                print(f"  [CLEANUP WARN] {e}")

        r = LHResult(
            name=name, passed=passed, elapsed_s=elapsed,
            subtasks_completed=len(result.get("subtasks_completed", [])),
            replans=replans[0], notes=notes,
        )
        self.results.append(r)
        print(f"\n  Result   : {'PASSED' if passed else 'FAILED'} in {elapsed:.1f}s")
        print(f"  Subtasks : {r.subtasks_completed} completed | replans: {r.replans}")
        if notes:
            print(f"  Notes    : {notes}")

    # ── use cases ──────────────────────────────────────────────────────────────

    def lh1_long_file_chain(self):
        """6-subtask chain across two apps — the flat 600 s budget of the old
        orchestrator could not fit this; the adaptive budget must.
        Every effect is verified on disk (no OCR judgment involved).
        """
        d = str(self.desktop).replace("\\", "/")
        notes_txt = self.desktop / "lh_notes.txt"
        copy_txt = self.desktop / "lh_copy.txt"
        report_dir = self.desktop / "lh_report"

        task = (
            f"open notepad, type: long horizon check, "
            f"save the document as {d}/lh_notes.txt, "
            f"then open the terminal and run: copy {d}/lh_notes.txt {d}/lh_copy.txt, "
            f"then run: mkdir {d}/lh_report, "
            f"then run: copy {d}/lh_copy.txt {d}/lh_report/final.txt"
        )

        def pre():
            for p in (notes_txt, copy_txt):
                p.unlink(missing_ok=True)
            shutil.rmtree(report_dir, ignore_errors=True)

        def verify():
            missing = [str(p) for p in
                       (notes_txt, copy_txt, report_dir / "final.txt")
                       if not p.exists()]
            if missing:
                return False, f"missing on disk: {missing}"
            content = (report_dir / "final.txt").read_text(encoding="utf-8", errors="replace")
            if "long horizon" not in content.lower():
                return False, f"final.txt content wrong: {content[:60]!r}"
            return True, "all 3 artifacts on disk with correct content"

        def cleanup():
            _kill("notepad.exe")
            pre()

        self._run("6-subtask file chain", task, verify, cleanup, pre)

    def lh2_structured_save_dialog(self):
        """Drive a native Save dialog's filename field via set_value semantics:
        the planner should fill the labelled field rather than blind-typing.
        Ground truth: the file lands exactly where the field said.
        """
        d = str(self.desktop).replace("\\", "/")
        target = self.desktop / "lh_form.txt"
        task = (
            f"open notepad, type: structured control check, "
            f"save the document as {d}/lh_form.txt"
        )

        def pre():
            target.unlink(missing_ok=True)

        def verify():
            if not target.exists():
                return False, "file not created via the dialog"
            return True, "dialog form filled and confirmed"

        def cleanup():
            _kill("notepad.exe")
            target.unlink(missing_ok=True)

        self._run("form fill via structured actions", task, verify, cleanup, pre)

    def lh3_zoom_schedule(self):
        """Full meeting-arrangement flow. Requires Zoom installed AND signed in;
        gated behind LIVE_ZOOM=1 because it creates a real calendar entry.
        """
        if os.environ.get("LIVE_ZOOM") != "1":
            print("\n  [SKIP] Zoom test disabled — set LIVE_ZOOM=1 to enable")
            return
        if not _zoom_installed():
            print("\n  [SKIP] Zoom is not installed on this machine")
            return

        topic = f"Agent Live Test {int(time.time())}"
        task = (
            f"open Zoom, click the Schedule button, and schedule a meeting "
            f"titled {topic} for tomorrow at 3:00 PM lasting 30 minutes"
        )

        def verify():
            # Ground truth: the topic appears in Zoom's upcoming-meetings list,
            # read through the accessibility tree (not OCR).
            try:
                from core import windows_uia
                found = windows_uia.find_element(topic, timeout_s=4.0)
                if found:
                    return True, f"meeting '{topic}' visible in Zoom"
            except Exception as e:
                return False, f"UIA verification error: {e}"
            return False, f"meeting '{topic}' not found in Zoom's meeting list"

        self._run("zoom meeting scheduling", task, verify)

    # ── summary ────────────────────────────────────────────────────────────────

    def summary(self) -> bool:
        print(f"\n{'=' * 62}\n  LONG-HORIZON SUMMARY\n{'=' * 62}")
        all_ok = True
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.name:<38} {r.elapsed_s:7.1f}s  "
                  f"subtasks={r.subtasks_completed} replans={r.replans}")
            if r.notes:
                print(f"         {r.notes}")
            all_ok &= r.passed
        return all_ok


def main():
    tester = LongHorizonTester()
    tester.lh1_long_file_chain()
    tester.lh2_structured_save_dialog()
    tester.lh3_zoom_schedule()
    ok = tester.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
