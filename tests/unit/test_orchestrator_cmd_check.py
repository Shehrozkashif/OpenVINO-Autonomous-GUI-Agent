# tests/unit/test_orchestrator_cmd_check.py
"""Tests for deterministic terminal-command verification (_verify_command_effect).

Live failure this guards against: `echo 'x' > file` SUCCEEDED (file on disk),
but OCR reflection read the silent new prompt as "no change → failed" and
aborted a subtask whose goal was already achieved.
"""
import sys

sys.path.insert(0, ".")

import os
import time
from unittest.mock import MagicMock, patch

import pytest

from core.orchestrator import OrchestratorConfig, TaskOrchestrator
from core.protocols.a2a import ActionStep, SubTask


def _sub(desc):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_orch():
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(plan_next_step=MagicMock(return_value=None)),
        grounder=MagicMock(min_confidence=0.5),
        actor=MagicMock(execute=MagicMock(return_value=True)),
        reflector=MagicMock(min_confidence=0.75),
        capturer=MagicMock(),
        task_memory=MagicMock(
            get_failure_hints=MagicMock(return_value=[]),
            store_failure_pattern=MagicMock(),
        ),
        config=OrchestratorConfig(max_retries_per_step=1, max_steps_per_subtask=5),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value="")
    orch._firewall_allows = MagicMock(return_value=True)
    return orch


@pytest.fixture
def orch():
    with patch("core.orchestrator.time.sleep"):
        yield _make_orch()


class TestCreateCommands:

    def test_fresh_file_passes(self, orch, tmp_path):
        f = tmp_path / "notes.txt"
        started = time.time()
        f.write_text("hello agent")
        ok, why = orch._verify_command_effect(
            _sub(f"run: echo 'hello agent' > {f}"), started, typed_ok=True)
        assert ok is True
        assert str(f) in why

    def test_missing_file_fails(self, orch, tmp_path):
        f = tmp_path / "nope.txt"
        ok, why = orch._verify_command_effect(
            _sub(f"run: echo 'x' > {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "does not exist" in why

    def test_stale_file_fails(self, orch, tmp_path):
        """A file left over from an earlier run must not pass."""
        f = tmp_path / "old.txt"
        f.write_text("old content")
        old_mtime = time.time() - 3600
        os.utime(f, (old_mtime, old_mtime))
        ok, why = orch._verify_command_effect(
            _sub(f"run: echo 'x' > {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "stale" in why.lower() or "not modified" in why.lower()

    def test_mkdir_fresh_folder_passes(self, orch, tmp_path):
        d = tmp_path / "projects"
        started = time.time()
        d.mkdir()
        ok, _ = orch._verify_command_effect(
            _sub(f"run: mkdir {d}"), started, typed_ok=True)
        assert ok is True

    def test_quoted_path_with_spaces(self, orch, tmp_path):
        f = tmp_path / "my notes.txt"
        started = time.time()
        f.write_text("x")
        ok, _ = orch._verify_command_effect(
            _sub(f"run: echo 'x' > \"{f}\""), started, typed_ok=True)
        assert ok is True


class TestDeleteCommands:

    def test_deleted_file_passes(self, orch, tmp_path):
        f = tmp_path / "gone.txt"   # never created
        ok, _ = orch._verify_command_effect(
            _sub(f"run: del {f}"), time.time(), typed_ok=True)
        assert ok is True

    def test_surviving_file_fails(self, orch, tmp_path):
        f = tmp_path / "alive.txt"
        f.write_text("x")
        ok, why = orch._verify_command_effect(
            _sub(f"run: del {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "still exists" in why


class TestGenericCommands:

    def _with_ocr_text(self, orch, text):
        from agents.grounding.grounding_agent import OCRWord
        words = [OCRWord(t, 0, 0, 10, 10, 0.9) for t in text.split()]
        orch._ocr = MagicMock()
        orch._ocr.extract = MagicMock(return_value=words)
        img = MagicMock()
        orch.capturer.capture = MagicMock(return_value=img)
        return orch

    def test_silence_means_success(self, orch):
        orch = self._with_ocr_text(orch, "PS C: Users sharo")
        ok, why = orch._verify_command_effect(
            _sub("run: git status"), time.time(), typed_ok=True)
        assert ok is True
        assert "silence" in why

    def test_error_marker_fails(self, orch):
        orch = self._with_ocr_text(
            orch, "out-file Access to the path is denied CategoryInfo OpenError")
        ok, why = orch._verify_command_effect(
            _sub("run: git status"), time.time(), typed_ok=True)
        assert ok is False

    def test_enter_without_typed_command_fails(self, orch):
        ok, why = orch._verify_command_effect(
            _sub("run: git status"), time.time(), typed_ok=False)
        assert ok is False
        assert "no command was typed" in why


class TestSubtaskIntegration:

    def test_cmd_subtask_completes_on_verified_effect(self, tmp_path):
        """Type + enter with the file actually created → subtask returns True
        WITHOUT any reflection call for the Enter step.
        """
        f = tmp_path / "notes.txt"

        type_step = ActionStep(id=1, subtask_id=1, action_type="type",
                               target=None, value=f"echo 'x' > {f}", key=None,
                               description="Type command", verification="")
        enter_step = ActionStep(id=2, subtask_id=1, action_type="key_press",
                                target=None, value=None, key="enter",
                                description="Execute command", verification="")

        from agents.reflection.reflection_agent import ReflectionResult
        ok_reflect = ReflectionResult(
            success=True, confidence=1.0, observation="ok",
            error_description="", should_retry=False, recovery_hint="",
            ocr_text="")

        orch = _make_orch()
        orch.planner.plan_next_step = MagicMock(side_effect=[type_step, enter_step, None])
        orch.reflector.verify = MagicMock(return_value=ok_reflect)

        # Simulate the Enter actually creating the file
        def _exec(step, **kw):
            if step.action_type == "key_press":
                f.write_text("x")
            return True
        orch.actor.execute = MagicMock(side_effect=_exec)

        with patch("core.orchestrator.time.sleep"):
            result = orch._execute_subtask(_sub(f"run: echo 'x' > {f}"))

        assert result is True
        # Reflection ran for the type step only — never for the Enter
        reflected = [c.args[0].action_type for c in orch.reflector.verify.call_args_list]
        assert "key_press" not in reflected
