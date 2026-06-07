# tests/unit/test_orchestrator_verify_launch.py
"""
Unit tests for Fix B — _verify_launch must NOT trigger on subtasks that mention
"open" as part of "open the context menu", "with the context menu open", etc.

It should still trigger correctly for genuine app-launch subtasks ("open Notepad",
"open Windows Terminal") and only fire if a known app keyword is present alongside
the word "open".

Tests also verify that the OCR-based check uses foreground-only snapshot regions
(not raw OCR) so the agent's own log window cannot cause false positives.
"""
import sys
sys.path.insert(0, ".")

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from core.orchestrator import TaskOrchestrator, OrchestratorConfig
from core.protocols.a2a import SubTask


# ── helpers ───────────────────────────────────────────────────────────────────

def _sub(description: str) -> SubTask:
    return SubTask(id=1, description=description, depends_on=[])


def _make_orch() -> TaskOrchestrator:
    """Build a minimal orchestrator with all collaborators mocked."""
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(),
        grounder=MagicMock(),
        actor=MagicMock(),
        reflector=MagicMock(),
        capturer=MagicMock(),
        task_memory=MagicMock(),
        config=OrchestratorConfig(),
        on_step_log=lambda _: None,
        ocr=MagicMock(),
    )
    return orch


# ── tests: trigger condition (Fix B part 1) ───────────────────────────────────

class TestVerifyLaunchTriggerCondition:

    def test_right_click_to_open_context_menu_skips_verification(self):
        """
        'right click on the desktop to open the context menu' contains 'open'
        but no known app keyword. _verify_launch must return True immediately
        (skip verification) and never try a process/OCR check.
        """
        orch = _make_orch()
        subtask = _sub("right click on the desktop to open the context menu")
        result = orch._verify_launch(subtask)
        # True = "no verification needed" (correctly skipped)
        assert result is True

    def test_context_menu_open_click_new_skips_verification(self):
        """
        'with the context menu open, click New' — 'open' in desc, no app keyword.
        Must skip verification.
        """
        orch = _make_orch()
        result = orch._verify_launch(_sub("with the context menu open, click New"))
        assert result is True

    def test_new_menu_open_click_folder_skips_verification(self):
        """'with the New menu open, click Folder' — same pattern, must skip."""
        orch = _make_orch()
        result = orch._verify_launch(_sub("with the New menu open, click Folder"))
        assert result is True

    def test_already_open_always_skips(self):
        """'already open' in desc → always True regardless of other words."""
        orch = _make_orch()
        result = orch._verify_launch(_sub("open notepad (already open from previous step)"))
        assert result is True

    def test_already_running_always_skips(self):
        """'already running' in desc → always True."""
        orch = _make_orch()
        result = orch._verify_launch(_sub("use the terminal already running"))
        assert result is True

    def test_no_launch_word_at_all_skips(self):
        """Subtask with no 'open'/'launch'/'search launcher' — skips immediately."""
        orch = _make_orch()
        result = orch._verify_launch(_sub("type TestFolder and press enter"))
        assert result is True

    def test_search_launcher_always_triggers(self):
        """
        'search launcher' explicitly → always runs verification path.
        Mock process check to return True so the call completes.
        """
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=False), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:
            snap = MagicMock()
            snap.ocr_regions = []
            mock_snap.return_value = snap
            # 'search launcher' but no known app — signals list empty → returns True
            result = orch._verify_launch(_sub("use search launcher to open an app"))
            # With no signals derived, _verify_launch returns True (nothing to check)
            assert result is True

    def test_launch_word_always_triggers_verification(self):
        """
        'launch' (not 'open') always triggers regardless of app keyword presence.
        Mock process check to fail so we can confirm it ran.
        """
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=False), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:
            snap = MagicMock()
            # Simulate no matching OCR text on screen
            snap.ocr_regions = []
            mock_snap.return_value = snap
            # 'launch Firefox' triggers. Process not found → False.
            result = orch._verify_launch(_sub("launch Firefox browser"))
            # Firefox is in _PROCESS_MAP_WINDOWS and _APP_SIGNALS; process not found → False
            assert result is False


# ── tests: "open + known app" correctly triggers verification ─────────────────

class TestVerifyLaunchAppKeywordCheck:

    def test_open_notepad_triggers_verification(self):
        """
        'open notepad' has 'open' + 'notepad' (known app) → triggers verification.
        Process check is mocked to fail → _verify_launch returns False.
        This confirms the verification path was entered (not skipped).
        """
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=False):
            result = orch._verify_launch(_sub("open notepad"))
        assert result is False, (
            "'open notepad' must trigger launch verification; "
            "with process not found it should return False"
        )

    def test_open_calculator_triggers_verification(self):
        """'open calculator' → triggers verification (process not found → False)."""
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=False):
            result = orch._verify_launch(_sub("open calculator"))
        assert result is False

    def test_open_terminal_triggers_verification(self):
        """'open windows terminal' → triggers verification."""
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=False):
            result = orch._verify_launch(_sub("open windows terminal"))
        assert result is False

    def test_open_notepad_process_found_returns_true(self):
        """'open notepad' with Notepad process running → verification passes → True."""
        orch = _make_orch()
        with patch.object(orch, '_is_process_running', return_value=True):
            result = orch._verify_launch(_sub("open notepad"))
        assert result is True


# ── tests: foreground-only OCR in the fallback check (Fix B part 3) ──────────

class TestVerifyLaunchForegroundOCR:

    def test_ocr_fallback_only_reads_foreground_regions(self):
        """
        For an app not in the process map (Linux path or unknown), the OCR fallback
        must read snapshot.ocr_regions filtered to is_in_foreground=True, not raw OCR.
        """
        orch = _make_orch()

        # Force Linux-like path by patching _OS
        with patch('core.orchestrator._OS', 'Linux'), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:

            # Two regions: one foreground with the signal, one background without
            fg_region = MagicMock()
            fg_region.text = "Calculator"
            fg_region.is_in_foreground = True

            bg_region = MagicMock()
            bg_region.text = "SomethingElse"
            bg_region.is_in_foreground = False

            snap = MagicMock()
            snap.ocr_regions = [fg_region, bg_region]
            mock_snap.return_value = snap

            result = orch._verify_launch(_sub("launch calculator"))
            # "Calculator" is in _APP_SIGNALS_LINUX signals; foreground region has it → True
            assert result is True
            # Confirm capture_snapshot was called (not raw self._ocr.extract)
            assert mock_snap.called

    def test_ocr_fallback_ignores_background_region_signal(self):
        """
        If the matching signal word appears only in a BACKGROUND region, it must
        NOT count as a confirmed launch.
        """
        orch = _make_orch()

        with patch('core.orchestrator._OS', 'Linux'), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:

            bg_region = MagicMock()
            bg_region.text = "Calculator"
            bg_region.is_in_foreground = False  # background — must be ignored

            fg_region = MagicMock()
            fg_region.text = "SomeOtherWord"
            fg_region.is_in_foreground = True

            snap = MagicMock()
            snap.ocr_regions = [bg_region, fg_region]
            mock_snap.return_value = snap

            result = orch._verify_launch(_sub("launch calculator"))
            # "Calculator" is only in background → not confirmed → False
            assert result is False
