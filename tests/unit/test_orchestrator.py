# tests/unit/test_orchestrator.py
"""Unit tests for core/orchestrator.py — TaskOrchestrator.

Organised by concern (each section was originally its own file; the helper
functions below are suffixed per section since several sections build a
differently-configured mock orchestrator under the same conceptual name).

  1. Burst detection at the instruction level (Fix C)
  2. Deterministic terminal-command verification (_verify_command_effect)
  3. Non-idempotent actions are not blind-retried (Fix C5)
  4. Action-type-aware step deduplication / loop guard
  5. "App already running" new-window launch semantics
  6. _verify_launch trigger conditions and OCR fallback (Fix B)
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from agents.grounding import GroundingResult
from agents.reflection import ReflectionResult
from core.burst_executor import detect_burst_from_instruction
from core.orchestrator import DEDUP_LIMIT_BY_ACTION_TYPE, OrchestratorConfig, TaskOrchestrator
from core.protocols import ActionBurst, ActionStep, SubTask

sys.path.insert(0, ".")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Burst detection at the instruction level (Fix C)
#
# Verifies:
#   1. When detect_burst_from_instruction matches, router.decompose is NOT called.
#   2. The synthetic SubTask passed to _execute_subtask has burst set to the detected burst.
#   3. When no burst matches, the router IS called normally.
#   4. detect_burst_from_instruction correctly recognises the full folder-creation instruction.
#   5. _execute_subtask uses subtask.burst directly when it is set (no second detect_burst call).
# ═══════════════════════════════════════════════════════════════════════════

def _make_orch_burst() -> TaskOrchestrator:
    """Minimal orchestrator with collaborators fully mocked."""
    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=MagicMock(),
        grounder=MagicMock(),
        actor=MagicMock(),
        reflector=MagicMock(),
        capturer=MagicMock(),
        task_memory=MagicMock(),
        config=OrchestratorConfig(max_steps_per_subtask=1),
        on_step_log=lambda _: None,
        ocr=MagicMock(),
    )
    orch.router.summarize_completion = MagicMock(return_value="done")
    orch.memory.find_similar = MagicMock(return_value=None)
    orch.memory.store_successful_task = MagicMock()
    orch._get_screen_context = MagicMock(return_value='"desktop"')
    # Patch out the real subtask execution and launch check so tests stay fast
    orch._execute_subtask = MagicMock(return_value=True)
    orch._verify_launch = MagicMock(return_value=True)
    orch._wait_for_settle = MagicMock()
    return orch


def _fake_burst() -> ActionBurst:
    step = ActionStep(
        id=1, subtask_id=1, action_type="right_click",
        target="desktop", description="test", verification="",
    )
    return ActionBurst(steps=[step], verify_at_end=False, timeout_ms=3000)


class TestBurstSkipsRouter:

    def test_router_decompose_not_called_when_burst_detected(self):
        """router.decompose must not be called when a burst is found at instruction level."""
        orch = _make_orch_burst()
        burst = _fake_burst()
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")
        orch.router.decompose.assert_not_called()

    def test_router_decompose_called_when_no_burst(self):
        """router.decompose IS called when detect_burst_from_instruction returns None."""
        orch = _make_orch_burst()
        mock_subtask = SubTask(id=1, description="open Notepad", depends_on=[])
        orch.router.decompose = MagicMock(return_value=("task_1", [mock_subtask]))
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=None):
            orch.execute("open Notepad")
        orch.router.decompose.assert_called_once()

    def test_only_one_subtask_created_for_burst_instruction(self):
        """Instruction-level burst produces exactly one synthetic subtask."""
        orch = _make_orch_burst()
        burst = _fake_burst()
        captured = []
        def _capture(subtask, **kwargs):
            captured.append(subtask)
            return True
        orch._execute_subtask = _capture

        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")

        assert len(captured) == 1, "Exactly one synthetic subtask must be created"

    def test_synthetic_subtask_has_burst_attached(self):
        """The synthetic subtask passed to _execute_subtask carries burst=<detected>."""
        orch = _make_orch_burst()
        burst = _fake_burst()
        captured_subtask = []

        def _capture(subtask, **kwargs):
            captured_subtask.append(subtask)
            return True
        orch._execute_subtask = _capture

        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute("Right click on the desktop, click New, click Folder, type Foo, press Enter.")

        assert captured_subtask[0].burst is burst, (
            "subtask.burst must be the ActionBurst returned by detect_burst_from_instruction"
        )

    def test_synthetic_subtask_description_is_full_instruction(self):
        """The synthetic subtask description is the full instruction, not a router fragment."""
        orch = _make_orch_burst()
        instruction = "Right click on the desktop, click New, click Folder, type MyFolder, press Enter."
        captured_subtask = []

        def _capture(subtask, **kwargs):
            captured_subtask.append(subtask)
            return True
        orch._execute_subtask = _capture

        burst = _fake_burst()
        with patch("core.orchestrator.detect_burst_from_instruction", return_value=burst):
            orch.execute(instruction)

        assert captured_subtask[0].description == instruction


class TestDetectBurstFromInstruction:

    def test_full_folder_creation_instruction_detected(self):
        """The canonical test instruction matches the 5-step shortcut-key folder burst."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, "
            "type TestFolder, press Enter."
        )
        assert burst is not None, "Should detect burst for full folder creation instruction"
        assert len(burst.steps) == 5

    def test_full_folder_creation_step_types(self):
        """5-step visual burst has the expected action types in order."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, type MyDir, press Enter."
        )
        types = [s.action_type for s in burst.steps]
        assert types == ["right_click", "click", "click", "type", "key_press"]

    def test_full_folder_creation_captures_name(self):
        """The type step carries the exact folder name from the instruction."""
        burst = detect_burst_from_instruction(
            "Right click on the desktop, click New, click Folder, type Projects, press Enter."
        )
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Projects"

    def test_create_folder_called_detected(self):
        """'create a folder called X' matches pattern 1c → 5-step shortcut-key burst."""
        burst = detect_burst_from_instruction("create a folder called Reports")
        assert burst is not None
        assert len(burst.steps) == 5
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Reports"

    def test_create_new_folder_named_detected(self):
        """'new folder named X' matches pattern 1c."""
        burst = detect_burst_from_instruction("new folder named Archive")
        assert burst is not None
        type_step = next(s for s in burst.steps if s.action_type == "type")
        assert type_step.value == "Archive"

    def test_unrelated_instruction_returns_none(self):
        """An instruction that has nothing to do with burst patterns returns None."""
        burst = detect_burst_from_instruction("open Firefox and navigate to gmail.com")
        assert burst is None

    def test_short_new_folder_keywords_still_match_pattern1a(self):
        """'new folder' keyword still fires (5-step shortcut-key burst with default name)."""
        burst = detect_burst_from_instruction("create new folder on the desktop")
        assert burst is not None
        assert len(burst.steps) == 5


class TestExecuteSubtaskUsesBurstField:

    def _make_subtask_orch(self):
        """Build an orchestrator where all collaborators are stubbed for burst execution."""
        orch = TaskOrchestrator(
            router=MagicMock(),
            planner=MagicMock(),
            grounder=MagicMock(),
            actor=MagicMock(),
            reflector=MagicMock(),
            capturer=MagicMock(),
            task_memory=MagicMock(),
            config=OrchestratorConfig(max_steps_per_subtask=5),
            on_step_log=lambda _: None,
            ocr=MagicMock(),
        )
        gr = GroundingResult(
            found=True, confidence=0.9, x=500, y=400,
            latency_ms=5.0, target="desktop", element_type="foreground_interactive",
        )
        orch.grounder.ground = MagicMock(return_value=gr)
        orch.grounder.min_confidence = 0.5
        orch.actor.execute = MagicMock(return_value=True)
        orch.reflector.verify = MagicMock(return_value=MagicMock(
            success=True, confidence=0.95, ocr_text="", error_description="",
            should_retry=False,
        ))
        orch.memory.get_failure_hints = MagicMock(return_value=[])
        orch._get_screen_context = MagicMock(return_value='"desktop"')
        return orch

    def test_subtask_with_burst_runs_burst_not_planning_loop(self):
        """When subtask.burst is set, _execute_subtask must use the burst executor
        and NOT call planner.plan_steps.
        """
        orch = self._make_subtask_orch()
        burst = _fake_burst()
        subtask = SubTask(id=1, description="folder task", depends_on=[], burst=burst)

        with patch.object(orch.burst_executor, 'run',
                          return_value=MagicMock(success=True, failed_at_step=None, reason="ok")) \
             as mock_burst_run:
            result = orch._execute_subtask(subtask)

        assert result is True
        mock_burst_run.assert_called_once_with(burst)
        orch.planner.plan_steps.assert_not_called()

    def test_subtask_without_burst_falls_back_to_planning_loop(self):
        """When subtask.burst is None and detect_burst returns None,
        _execute_subtask must fall through to the LLM planning loop.
        """
        orch = self._make_subtask_orch()
        subtask = SubTask(id=1, description="click the OK button", depends_on=[])
        # Planner returns one step then done
        step = ActionStep(
            id=1, subtask_id=1, action_type="key_press",
            target=None, value=None, key="escape",
            description="Press Escape", verification="dialog dismissed",
        )
        orch.planner.plan_steps = MagicMock(side_effect=[[step], None])

        with patch("core.orchestrator.detect_burst", return_value=None):
            result = orch._execute_subtask(subtask)

        assert result is True
        orch.planner.plan_steps.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Deterministic terminal-command verification (_verify_command_effect)
#
# Live failure this guards against: `echo 'x' > file` SUCCEEDED (file on disk),
# but OCR reflection read the silent new prompt as "no change → failed" and
# aborted a subtask whose goal was already achieved.
# ═══════════════════════════════════════════════════════════════════════════

def _sub_cmd(desc):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_orch_cmd():
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
        yield _make_orch_cmd()


class TestCreateCommands:

    def test_fresh_file_passes(self, orch, tmp_path):
        f = tmp_path / "notes.txt"
        started = time.time()
        f.write_text("hello agent")
        ok, why = orch._verify_command_effect(
            _sub_cmd(f"run: echo 'hello agent' > {f}"), started, typed_ok=True)
        assert ok is True
        assert str(f) in why

    def test_missing_file_fails(self, orch, tmp_path):
        f = tmp_path / "nope.txt"
        ok, why = orch._verify_command_effect(
            _sub_cmd(f"run: echo 'x' > {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "does not exist" in why

    def test_stale_file_fails(self, orch, tmp_path):
        """A file left over from an earlier run must not pass."""
        f = tmp_path / "old.txt"
        f.write_text("old content")
        old_mtime = time.time() - 3600
        os.utime(f, (old_mtime, old_mtime))
        ok, why = orch._verify_command_effect(
            _sub_cmd(f"run: echo 'x' > {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "stale" in why.lower() or "not modified" in why.lower()

    def test_mkdir_fresh_folder_passes(self, orch, tmp_path):
        d = tmp_path / "projects"
        started = time.time()
        d.mkdir()
        ok, _ = orch._verify_command_effect(
            _sub_cmd(f"run: mkdir {d}"), started, typed_ok=True)
        assert ok is True

    def test_quoted_path_with_spaces(self, orch, tmp_path):
        f = tmp_path / "my notes.txt"
        started = time.time()
        f.write_text("x")
        ok, _ = orch._verify_command_effect(
            _sub_cmd(f"run: echo 'x' > \"{f}\""), started, typed_ok=True)
        assert ok is True


class TestDeleteCommands:

    def test_deleted_file_passes(self, orch, tmp_path):
        f = tmp_path / "gone.txt"   # never created
        ok, _ = orch._verify_command_effect(
            _sub_cmd(f"run: del {f}"), time.time(), typed_ok=True)
        assert ok is True

    def test_surviving_file_fails(self, orch, tmp_path):
        f = tmp_path / "alive.txt"
        f.write_text("x")
        ok, why = orch._verify_command_effect(
            _sub_cmd(f"run: del {f}"), time.time(), typed_ok=True)
        assert ok is False
        assert "still exists" in why


class TestGenericCommands:

    def _with_ocr_text(self, orch, text):
        from agents.grounding import OCRWord
        words = [OCRWord(t, 0, 0, 10, 10, 0.9) for t in text.split()]
        orch._ocr = MagicMock()
        orch._ocr.extract = MagicMock(return_value=words)
        img = MagicMock()
        orch.capturer.capture = MagicMock(return_value=img)
        return orch

    def test_silence_means_success(self, orch):
        orch = self._with_ocr_text(orch, "PS C: Users sharo")
        ok, why = orch._verify_command_effect(
            _sub_cmd("run: git status"), time.time(), typed_ok=True)
        assert ok is True
        assert "silence" in why

    def test_error_marker_fails(self, orch):
        orch = self._with_ocr_text(
            orch, "out-file Access to the path is denied CategoryInfo OpenError")
        ok, why = orch._verify_command_effect(
            _sub_cmd("run: git status"), time.time(), typed_ok=True)
        assert ok is False

    def test_enter_without_typed_command_fails(self, orch):
        ok, why = orch._verify_command_effect(
            _sub_cmd("run: git status"), time.time(), typed_ok=False)
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

        ok_reflect = ReflectionResult(
            success=True, confidence=1.0, observation="ok",
            error_description="", should_retry=False, recovery_hint="",
            ocr_text="")

        orch = _make_orch_cmd()
        orch.planner.plan_steps = MagicMock(side_effect=[[type_step], [enter_step], None])
        orch.reflector.verify = MagicMock(return_value=ok_reflect)

        # Simulate the Enter actually creating the file
        def _exec(step, **kw):
            if step.action_type == "key_press":
                f.write_text("x")
            return True
        orch.actor.execute = MagicMock(side_effect=_exec)

        with patch("core.orchestrator.time.sleep"):
            result = orch._execute_subtask(_sub_cmd(f"run: echo 'x' > {f}"))

        assert result is True
        # Reflection ran for the type step only — never for the Enter
        reflected = [c.args[0].action_type for c in orch.reflector.verify.call_args_list]
        assert "key_press" not in reflected


class TestSaveTargetExtraction:
    """_subtask_save_target parses the destination path from a save subtask."""

    def test_extracts_windows_path(self):
        assert TaskOrchestrator._subtask_save_target(
            _sub_cmd("with text in Notepad, save the document as C:/Users/x/Desktop/haiku.txt")
        ) == "C:/Users/x/Desktop/haiku.txt"

    def test_no_path_returns_none(self):
        assert TaskOrchestrator._subtask_save_target(_sub_cmd("save the document")) is None

    def test_non_save_returns_none(self):
        assert TaskOrchestrator._subtask_save_target(
            _sub_cmd("click in the document area and type: hello world")) is None

    def test_quoted_path_with_spaces(self):
        assert TaskOrchestrator._subtask_save_target(
            _sub_cmd("save the report as 'D:/work/report v2.pdf'")) == "D:/work/report v2.pdf"


class TestFileSavedFresh:
    """_file_saved_fresh confirms a save by checking the file on disk."""

    def test_fresh_file_passes(self, orch, tmp_path):
        f = tmp_path / "a.txt"
        started = time.time()
        f.write_text("x")
        assert orch._file_saved_fresh(str(f), started) is True

    def test_missing_file_fails(self, orch, tmp_path):
        assert orch._file_saved_fresh(str(tmp_path / "nope.txt"), time.time()) is False

    def test_stale_file_fails(self, orch, tmp_path):
        f = tmp_path / "old.txt"
        f.write_text("x")
        old = time.time() - 3600
        os.utime(f, (old, old))
        assert orch._file_saved_fresh(str(f), time.time()) is False


class TestSaveSubtaskIntegration:

    def test_save_subtask_completes_when_file_appears(self, tmp_path):
        """A "save as <path>" subtask returns True the moment the file lands on
        disk — it must NOT loop on ctrl+s waiting for an OCR confirmation that
        editors never show.
        """
        f = tmp_path / "haiku.txt"
        ctrls = ActionStep(id=1, subtask_id=1, action_type="hotkey",
                           target=None, value=None, key="ctrl+s",
                           description="Save the document", verification="")
        enter = ActionStep(id=2, subtask_id=1, action_type="key_press",
                           target=None, value=None, key="enter",
                           description="Confirm save", verification="")

        ok_reflect = ReflectionResult(
            success=True, confidence=0.95, observation="ok",
            error_description="", should_retry=False, recovery_hint="",
            ocr_text="")

        orch = _make_orch_cmd()
        # Plenty of ctrl+s steps queued — the save-check must short-circuit
        # before they are all consumed.
        orch.planner.plan_steps = MagicMock(
            side_effect=[[ctrls], [enter], [ctrls], [ctrls], [ctrls]])
        orch.reflector.verify = MagicMock(return_value=ok_reflect)

        def _exec(step, **kw):
            if step.key == "enter":
                f.write_text("a haiku")   # the save writes the file
            return True
        orch.actor.execute = MagicMock(side_effect=_exec)

        with patch("core.orchestrator.time.sleep"):
            result = orch._execute_subtask(_sub_cmd(f"save the document as {f}"))

        assert result is True
        assert f.exists()
        # Confirmed shortly after the save — no long ctrl+s loop.
        assert orch.planner.plan_steps.call_count <= 3


class TestDeterministicSaveAs:

    def test_try_save_as_writes_and_confirms(self, tmp_path):
        """ctrl+s → (dialog) → ctrl+a → type path → enter, confirmed on disk."""
        f = tmp_path / "haiku.txt"
        orch = _make_orch_cmd()
        # No dialog yet → ctrl+s fires; dialog visible on the confirm check.
        orch._save_dialog_visible = MagicMock(side_effect=[False, True])
        orch._wait_for_settle = MagicMock()

        def _exec(step):
            if step.key == "enter":
                f.write_text("a haiku")
            return True
        orch._execute_step = MagicMock(side_effect=_exec)

        with patch("core.orchestrator.time.sleep"):
            ok = orch._try_save_as(str(f), time.time())

        assert ok is True
        assert f.exists()
        keys = [c.args[0].key for c in orch._execute_step.call_args_list]
        assert "ctrl+s" in keys and "ctrl+a" in keys and "enter" in keys

    def test_try_save_as_types_backslashes_on_windows(self, tmp_path):
        """The router emits forward-slash paths, but the Windows Save dialog
        rejects them — the deterministic save must type a backslash path.
        """
        f = tmp_path / "haiku.txt"
        forward = str(f).replace("\\", "/")   # as the router/sub-task emits it
        orch = _make_orch_cmd()
        orch._save_dialog_visible = MagicMock(return_value=True)
        orch._wait_for_settle = MagicMock()

        def _exec(step):
            if step.key == "enter":
                f.write_text("x")
            return True
        orch._execute_step = MagicMock(side_effect=_exec)

        with patch("core.orchestrator.time.sleep"):
            ok = orch._try_save_as(forward, time.time())

        assert ok is True
        typed = [c.args[0].value for c in orch._execute_step.call_args_list
                 if c.args[0].action_type == "type"]
        assert typed and "/" not in typed[0] and "\\" in typed[0]

    def test_try_save_as_defers_when_no_dialog(self, tmp_path):
        """If ctrl+s opens no dialog, never type the path into the document —
        fall back to the planning loop and don't write the file.
        """
        f = tmp_path / "haiku.txt"   # never created
        orch = _make_orch_cmd()
        orch._save_dialog_visible = MagicMock(return_value=False)
        orch._wait_for_settle = MagicMock()
        orch._execute_step = MagicMock(return_value=True)

        with patch("core.orchestrator.time.sleep"):
            ok = orch._try_save_as(str(f), time.time())

        assert ok is False
        typed = [c.args[0] for c in orch._execute_step.call_args_list
                 if c.args[0].action_type == "type"]
        assert typed == [], "must not type a path when no Save dialog is visible"

    def test_save_subtask_uses_deterministic_path(self, tmp_path):
        """A save subtask runs _try_save_as up front and completes without ever
        entering the planning loop.
        """
        f = tmp_path / "haiku.txt"
        orch = _make_orch_cmd()
        orch._try_save_as = MagicMock(side_effect=lambda *_: (f.write_text("x"), True)[1])

        with patch("core.orchestrator.time.sleep"):
            result = orch._execute_subtask(_sub_cmd(f"save the document as {f}"))

        assert result is True
        orch._try_save_as.assert_called_once()
        orch.planner.plan_steps.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Non-idempotent actions are not blind-retried (Fix C5)
#
# A `type` step (and Enter/paste) changes state every time it runs. When such a
# step physically executes but verification comes back uncertain or failed, the
# orchestrator must NOT re-execute it (which would type the text twice). It hands
# control back to the planner instead.
#
# Contrast: an idempotent step (click) may still be retried on an uncertain result.
# ═══════════════════════════════════════════════════════════════════════════

_UNCERTAIN = ReflectionResult(
    success=False, confidence=0.40, observation="unclear",
    error_description="", should_retry=True, recovery_hint="", ocr_text="",
)


def _make_orch_idem(plan_steps, reflection):
    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(return_value=reflection)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(return_value=GroundingResult(
        found=True, confidence=0.9, x=10, y=20, latency_ms=1.0,
        target="x", element_type="foreground_interactive"))

    planner = MagicMock()
    planner.plan_steps = MagicMock(side_effect=[[s] for s in plan_steps] + [None] * 10)

    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(), planner=planner, grounder=grounder, actor=actor,
        reflector=reflector, capturer=MagicMock(), task_memory=memory,
        config=OrchestratorConfig(max_retries_per_step=3, max_steps_per_subtask=1,
                                  consecutive_failures_limit=10),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value="")
    orch._firewall_allows = MagicMock(return_value=True)
    return orch, actor


def _type_step_idem():
    return ActionStep(id=1, subtask_id=1, action_type="type", target=None,
                      value="hello", key=None, description="type hello",
                      verification="hello visible")


def _click_step_idem():
    return ActionStep(id=1, subtask_id=1, action_type="click", target="Btn",
                      value=None, key=None, description="click Btn",
                      verification="changed")


class TestIdempotency:

    def test_type_executes_once_on_uncertain(self):
        """An uncertain `type` verdict must not cause a second type within one step."""
        orch, actor = _make_orch_idem([_type_step_idem()], _UNCERTAIN)
        orch._execute_subtask(SubTask(id=1, description="do it", depends_on=[]))
        type_calls = [c for c in actor.execute.call_args_list
                      if c.args and c.args[0].action_type == "type"]
        assert len(type_calls) == 1, f"expected 1 type execution, got {len(type_calls)}"

    def test_click_may_retry_on_uncertain(self):
        """An idempotent click is allowed to retry on an uncertain verdict."""
        orch, actor = _make_orch_idem([_click_step_idem()], _UNCERTAIN)
        orch._execute_subtask(SubTask(id=1, description="do it", depends_on=[]))
        click_calls = [c for c in actor.execute.call_args_list
                       if c.args and c.args[0].action_type == "click"]
        assert len(click_calls) >= 2, "click should retry at least once on uncertain"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Action-type-aware step deduplication (loop guard)
#
# The loop guard fires when _same_step_streak > DEDUP_LIMIT_BY_ACTION_TYPE[action_type].
# Streak starts at 0 on the first success of a given signature, increments on each
# consecutive repeat. Trigger semantics (limit N = allow N repeats):
#
#   type        limit=1 → fires on 3rd total success (2nd repeat)
#   click       limit=2 → fires on 4th total success (3rd repeat)
#   right_click limit=1 → fires on 3rd total success (2nd repeat)
#   key_press   limit=3 → fires on 5th total success (4th repeat)
# ═══════════════════════════════════════════════════════════════════════════

def _subtask_loop(desc="do something"):
    return SubTask(id=1, description=desc, depends_on=[])


def _step_loop(action_type="click", target="Button", key=None, value=None):
    return ActionStep(
        id=1, subtask_id=1,
        action_type=action_type,
        target=target, value=value, key=key,
        description=f"{action_type} {target or key or value}",
        verification="expected result",
    )


_SUCCESS = ReflectionResult(
    success=True, confidence=0.9, observation="ok",
    error_description="", should_retry=False, recovery_hint="", ocr_text="",
)


def _make_orch_loop(plan_steps):
    """Build an orchestrator where:
    - planner returns plan_steps in order, then None (goal achieved).
    - every reflection returns success.
    - actor.execute always returns True.
    - grounding always finds the target.
    """
    reflector = MagicMock()
    reflector.min_confidence = 0.75
    reflector.verify = MagicMock(return_value=_SUCCESS)

    actor = MagicMock()
    actor.execute = MagicMock(return_value=True)

    grounder = MagicMock()
    grounder.min_confidence = 0.5
    grounder.ground = MagicMock(
        return_value=GroundingResult(
            found=True, confidence=0.9, x=100, y=200,
            latency_ms=5.0, target="Button",
            element_type="foreground_interactive",
        )
    )

    planner = MagicMock()
    planner.plan_steps = MagicMock(side_effect=[[s] for s in plan_steps] + [None])

    memory = MagicMock()
    memory.get_failure_hints = MagicMock(return_value=[])
    memory.store_failure_pattern = MagicMock()

    orch = TaskOrchestrator(
        router=MagicMock(),
        planner=planner,
        grounder=grounder,
        actor=actor,
        reflector=reflector,
        capturer=MagicMock(),
        task_memory=memory,
        config=OrchestratorConfig(
            max_retries_per_step=1,
            max_steps_per_subtask=25,
            consecutive_failures_limit=10,
        ),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"Button"')
    return orch


class TestDedupLimitValues:
    """The dict must contain the exact values specified in the design document."""

    def test_type_limit_is_1(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["type"] == 1

    def test_click_limit_is_2(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["click"] == 2

    def test_right_click_limit_is_1(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["right_click"] == 1

    def test_key_press_limit_is_3(self):
        assert DEDUP_LIMIT_BY_ACTION_TYPE["key_press"] == 3


class TestTypeDedupLimit:
    """type limit=1 → streak 0→1: allowed; streak 1→2: trigger.
    Total appearances: 2 = allowed, 3 = triggers.
    """

    def test_type_appears_twice_is_allowed(self):
        """type_step × 2, then a different step, then None.
        Streak after 2nd type = 1 (1 > 1 is False) → no trigger.
        Planner is called 4 times (2 type + 1 diff + None).
        """
        ts = _step_loop("type", target=None, value="hello")
        ds = _step_loop("key_press", key="enter")
        orch = _make_orch_loop([ts, ts, ds])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 4  # all 3 steps + None

    def test_type_appears_three_times_triggers(self):
        """type_step × 3 → streak reaches 2 (2 > 1) → loop guard fires after step 3.
        Planner is called exactly 3 times (4th call never happens).
        """
        ts = _step_loop("type", target=None, value="hello")
        orch = _make_orch_loop([ts, ts, ts])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 3

    def test_type_trigger_returns_true(self):
        """Loop guard always returns True (declares goal achieved, not failure)."""
        ts = _step_loop("type", target=None, value="x")
        orch = _make_orch_loop([ts, ts, ts])
        assert orch._execute_subtask(_subtask_loop()) is True


class TestClickDedupLimit:
    """click limit=2 → streak 0→1→2: allowed; streak 2→3: trigger.
    Total appearances: 3 = allowed, 4 = triggers.
    """

    def test_click_appears_three_times_is_allowed(self):
        """Click × 3, then different, then None.
        Streak after 3rd click = 2 (2 > 2 is False) → no trigger.
        Planner called 5 times.
        """
        cs = _step_loop("click", target="Btn")
        ds = _step_loop("key_press", key="escape")
        orch = _make_orch_loop([cs, cs, cs, ds])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 5

    def test_click_appears_four_times_triggers(self):
        """Click × 4 → streak = 3 (3 > 2) → loop guard fires after step 4.
        Planner called exactly 4 times.
        """
        cs = _step_loop("click", target="Btn")
        orch = _make_orch_loop([cs, cs, cs, cs])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 4

    def test_click_trigger_injects_escape(self):
        """When the click loop guard fires, actor.execute must be called with an
        Escape key_press step (in addition to the 4 regular click executions).
        """
        cs = _step_loop("click", target="Btn")
        orch = _make_orch_loop([cs, cs, cs, cs])
        orch._execute_subtask(_subtask_loop())
        # 4 click executions + 1 escape injection = 5 actor.execute calls
        assert orch.actor.execute.call_count == 5
        # Verify the last call was the escape step
        last_call_step = orch.actor.execute.call_args_list[-1][0][0]
        assert last_call_step.action_type == "key_press"
        assert last_call_step.key == "escape"


class TestRightClickDedupLimit:

    def test_right_click_appears_twice_is_allowed(self):
        rc = _step_loop("right_click", target="Desktop")
        ds = _step_loop("key_press", key="escape")
        orch = _make_orch_loop([rc, rc, ds])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 4

    def test_right_click_appears_three_times_triggers(self):
        rc = _step_loop("right_click", target="Desktop")
        orch = _make_orch_loop([rc, rc, rc])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 3

    def test_right_click_trigger_injects_escape(self):
        rc = _step_loop("right_click", target="Desktop")
        orch = _make_orch_loop([rc, rc, rc])
        orch._execute_subtask(_subtask_loop())
        # 3 right_click executions + 1 escape = 4 actor calls
        assert orch.actor.execute.call_count == 4
        last_step = orch.actor.execute.call_args_list[-1][0][0]
        assert last_step.action_type == "key_press"
        assert last_step.key == "escape"


class TestStreakReset:
    """A different action signature resets the streak; subsequent runs start fresh."""

    def test_type_streak_resets_after_click(self):
        """Type × 2 → streak = 1 (at limit, but NOT >1 so not triggered).
        click (different) → streak resets to 0.
        type × 2 → streak = 1 again (fresh start, NOT triggered).
        None → done normally.
        Planner called 6 times (5 steps + None).
        """
        ts = _step_loop("type", target=None, value="hello")
        cs = _step_loop("click", target="Foo")
        orch = _make_orch_loop([ts, ts, cs, ts, ts])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 6

    def test_click_streak_resets_after_type(self):
        """Click × 3 → streak = 2 (at limit, NOT triggered).
        type → streak resets.
        click × 3 → streak = 2 again (NOT triggered).
        """
        cs = _step_loop("click", target="Btn")
        ts = _step_loop("type", target=None, value="x")
        orch = _make_orch_loop([cs, cs, cs, ts, cs, cs, cs])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 8  # 7 steps + None

    def test_different_target_same_type_resets_streak(self):
        """Two clicks on the same target, then a click on a different target,
        then two more clicks on the original target — streak for the original
        resets after the different target, so no trigger.
        """
        ca = _step_loop("click", target="Alpha")
        cb = _step_loop("click", target="Beta")
        orch = _make_orch_loop([ca, ca, cb, ca, ca])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 6

    def test_only_type_no_escape_on_trigger(self):
        """Type loop guard does NOT inject Escape (only click/right_click do)."""
        ts = _step_loop("type", target=None, value="hello")
        orch = _make_orch_loop([ts, ts, ts])
        orch._execute_subtask(_subtask_loop())
        # Exactly 3 type executions, no escape step
        assert orch.actor.execute.call_count == 3
        for call in orch.actor.execute.call_args_list:
            assert call[0][0].action_type == "type"


# ═══════════════════════════════════════════════════════════════════════════
# 5. "App already running" new-window launch semantics
#
# Live failure this guards against: "open Windows Terminal" was satisfied by
# clicking the taskbar button of the user's EXISTING terminal (which was running
# another program), and the agent then typed shell commands into that session.
#
# Rules under test:
#   1. When the goal app's process pre-exists the subtask, the planner receives a
#      [NOTE] instructing it to open a NEW window (never focus the existing one).
#   2. _verify_launch with a recorded window-count baseline passes ONLY when the
#      window count increased; bare process existence is no longer sufficient.
#   3. Without a baseline (app was not running), behaviour is unchanged.
# ═══════════════════════════════════════════════════════════════════════════

def _sub_nwl(desc):
    return SubTask(id=1, description=desc, depends_on=[])


def _make_orch_nwl():
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
        config=OrchestratorConfig(max_retries_per_step=1, max_steps_per_subtask=3),
        on_step_log=lambda _: None,
    )
    orch._get_screen_context = MagicMock(return_value='"Terminal"')
    return orch


class TestPreExistingAppNote:

    def test_note_injected_when_app_already_running(self):
        """Process pre-exists → planner's task_context gets the NEW-window NOTE."""
        orch = _make_orch_nwl()
        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            orch._execute_subtask(_sub_nwl("open windows terminal"))

        ctx = orch.planner.plan_steps.call_args.kwargs.get("task_context")
        assert ctx, "task_context must be passed to the planner"
        joined = " ".join(ctx)
        assert "ALREADY running" in joined
        assert "NEW" in joined
        # Baseline must be recorded for _verify_launch
        assert orch._launch_window_baseline.get("WindowsTerminal.exe") == 1

    def test_no_note_when_app_not_running(self):
        orch = _make_orch_nwl()
        with patch.object(orch, "_is_process_running", return_value=False), \
             patch.object(orch, "_count_process_windows", return_value=0):
            orch._execute_subtask(_sub_nwl("open windows terminal"))

        ctx = orch.planner.plan_steps.call_args.kwargs.get("task_context")
        assert not ctx or "ALREADY running" not in " ".join(ctx)
        assert "WindowsTerminal.exe" not in orch._launch_window_baseline

    def test_non_launch_subtask_records_no_baseline(self):
        orch = _make_orch_nwl()
        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            orch._execute_subtask(_sub_nwl("with the terminal already open, run: dir"))
        assert orch._launch_window_baseline == {}


class TestVerifyLaunchWithBaseline:

    def test_focusing_existing_window_does_not_pass(self):
        """Window count flat at baseline → launch NOT confirmed."""
        orch = _make_orch_nwl()
        orch._launch_window_baseline["WindowsTerminal.exe"] = 1
        with patch.object(orch, "_count_process_windows", return_value=1), \
             patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_launch_confirmed", return_value=True), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub_nwl("open windows terminal")) is False, (
                "bare process existence must NOT confirm a launch when the app "
                "pre-existed the subtask"
            )

    def test_new_window_passes(self):
        orch = _make_orch_nwl()
        orch._launch_window_baseline["WindowsTerminal.exe"] = 1
        with patch.object(orch, "_count_process_windows", return_value=2), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub_nwl("open windows terminal")) is True

    def test_no_baseline_falls_back_to_process_check(self):
        orch = _make_orch_nwl()
        with patch.object(orch, "_launch_confirmed", return_value=True), \
             patch("core.orchestrator.time.sleep"):
            assert orch._verify_launch(_sub_nwl("open windows terminal")) is True


class TestCtrlCTerminalGuard:

    def test_ctrl_c_blocked_when_foreground_is_terminal(self):
        step = ActionStep(id=1, subtask_id=1, action_type="hotkey", target=None,
                          value=None, key="ctrl+c", description="copy error",
                          verification="")
        orch = _make_orch_nwl()
        with patch.object(orch, "_foreground_is_terminal", return_value=True):
            assert orch._execute_step(step) is False
        orch.actor.execute.assert_not_called()

    def test_ctrl_c_allowed_outside_terminal(self):
        step = ActionStep(id=1, subtask_id=1, action_type="hotkey", target=None,
                          value=None, key="ctrl+c", description="copy text",
                          verification="")
        orch = _make_orch_nwl()
        with patch.object(orch, "_foreground_is_terminal", return_value=False):
            orch._execute_step(step)
        orch.actor.execute.assert_called_once()


class TestLoopGuardCommandSubtask:

    def _looping_orch(self, desc):
        """Orchestrator where a key_press enter loops 5× after one failure."""
        enter = ActionStep(id=1, subtask_id=1, action_type="key_press",
                           target=None, value=None, key="enter",
                           description="Execute command", verification="")
        fail = ReflectionResult(success=False, confidence=0.9, observation="err",
                                error_description="Access denied",
                                should_retry=True, recovery_hint="", ocr_text="")
        ok = ReflectionResult(success=True, confidence=1.0, observation="ok",
                              error_description="", should_retry=False,
                              recovery_hint="", ocr_text="")
        orch = _make_orch_nwl()
        orch.config.max_steps_per_subtask = 10
        orch.config.max_retries_per_step = 1
        orch.config.consecutive_failures_limit = 10
        orch.config.visual_replan_after = 0
        orch.planner.plan_steps = MagicMock(return_value=[enter])
        # First execution fails, then the identical step "succeeds" repeatedly
        # until the loop guard fires (key_press dedup limit = 3).
        orch.reflector.verify = MagicMock(side_effect=[fail] + [ok] * 10)
        return orch

    def test_command_subtask_loop_after_failure_fails(self):
        orch = self._looping_orch("with the terminal already open, run: echo x > f.txt")
        result = orch._execute_subtask(
            _sub_nwl("with the terminal already open, run: echo x > f.txt"))
        assert result is False, (
            "a 'run:' subtask that looped after a failed execution must FAIL — "
            "dependents would otherwise build on state that doesn't exist"
        )

    def test_non_command_subtask_loop_still_returns_true(self):
        orch = self._looping_orch("press arrow keys")
        result = orch._execute_subtask(_sub_nwl("press arrow keys repeatedly"))
        assert result is True
        assert orch._degraded is True


class TestGoalCheckWithBaseline:

    def test_goal_check_requires_new_window_when_preexisting(self):
        """The in-loop GOAL-CHECK early exit must not fire while count is flat."""
        click = ActionStep(id=1, subtask_id=1, action_type="click",
                           target="Terminal", value=None, key=None,
                           description="click Terminal", verification="")
        ok = ReflectionResult(success=True, confidence=0.9, observation="ok",
                              error_description="", should_retry=False,
                              recovery_hint="", ocr_text="")

        orch = _make_orch_nwl()
        orch.planner.plan_steps = MagicMock(side_effect=[[click], None])
        orch.reflector.verify = MagicMock(return_value=ok)
        orch.grounder.ground = MagicMock(return_value=GroundingResult(
            found=True, confidence=0.9, x=1, y=2, latency_ms=1.0,
            target="Terminal", element_type="foreground_interactive"))

        with patch.object(orch, "_is_process_running", return_value=True), \
             patch.object(orch, "_count_process_windows", return_value=1):
            result = orch._execute_subtask(_sub_nwl("open windows terminal"))

        # Subtask still completes (planner returned None), but it must have been
        # the planner's decision — the GOAL-CHECK shortcut must not have fired
        # after the first step (count never rose above baseline), so the planner
        # must have been called a second time.
        assert result is True
        assert orch.planner.plan_steps.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# 6. _verify_launch trigger conditions and OCR fallback (Fix B)
#
# _verify_launch must NOT trigger on subtasks that mention "open" as part of
# "open the context menu", "with the context menu open", etc. It should still
# trigger correctly for genuine app-launch subtasks ("open Notepad", "open
# Windows Terminal") and only fire if a known app keyword is present alongside
# the word "open".
#
# Also verifies that the OCR-based check uses foreground-only snapshot regions
# (not raw OCR) so the agent's own log window cannot cause false positives.
# ═══════════════════════════════════════════════════════════════════════════

def _sub_vl(description: str) -> SubTask:
    return SubTask(id=1, description=description, depends_on=[])


def _make_orch_vl() -> TaskOrchestrator:
    """Build a minimal orchestrator with all collaborators mocked."""
    return TaskOrchestrator(
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


class TestVerifyLaunchTriggerCondition:

    def test_right_click_to_open_context_menu_skips_verification(self):
        """'right click on the desktop to open the context menu' contains 'open'
        but no known app keyword. _verify_launch must return True immediately
        (skip verification) and never try a process/OCR check.
        """
        orch = _make_orch_vl()
        subtask = _sub_vl("right click on the desktop to open the context menu")
        result = orch._verify_launch(subtask)
        # True = "no verification needed" (correctly skipped)
        assert result is True

    def test_context_menu_open_click_new_skips_verification(self):
        """'with the context menu open, click New' — 'open' in desc, no app keyword.
        Must skip verification.
        """
        orch = _make_orch_vl()
        result = orch._verify_launch(_sub_vl("with the context menu open, click New"))
        assert result is True

    def test_new_menu_open_click_folder_skips_verification(self):
        """'with the New menu open, click Folder' — same pattern, must skip."""
        orch = _make_orch_vl()
        result = orch._verify_launch(_sub_vl("with the New menu open, click Folder"))
        assert result is True

    def test_already_open_always_skips(self):
        """'already open' in desc → always True regardless of other words."""
        orch = _make_orch_vl()
        result = orch._verify_launch(_sub_vl("open notepad (already open from previous step)"))
        assert result is True

    def test_already_running_always_skips(self):
        """'already running' in desc → always True."""
        orch = _make_orch_vl()
        result = orch._verify_launch(_sub_vl("use the terminal already running"))
        assert result is True

    def test_no_launch_word_at_all_skips(self):
        """Subtask with no 'open'/'launch'/'search launcher' — skips immediately."""
        orch = _make_orch_vl()
        result = orch._verify_launch(_sub_vl("type TestFolder and press enter"))
        assert result is True

    def test_search_launcher_always_triggers(self):
        """'search launcher' explicitly → always runs verification path.
        Mock process check to return True so the call completes.
        """
        orch = _make_orch_vl()
        with patch.object(orch, '_is_process_running', return_value=False), \
             patch.object(orch, '_process_has_visible_window', return_value=False), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:
            snap = MagicMock()
            snap.ocr_regions = []
            mock_snap.return_value = snap
            # 'search launcher' but no known app — signals list empty → returns True
            result = orch._verify_launch(_sub_vl("use search launcher to open an app"))
            # With no signals derived, _verify_launch returns True (nothing to check)
            assert result is True

    def test_launch_word_always_triggers_verification(self):
        """'launch' (not 'open') always triggers regardless of app keyword presence.
        Mock process check to fail so we can confirm it ran.
        """
        orch = _make_orch_vl()
        with patch.object(orch, '_is_process_running', return_value=False), \
             patch.object(orch, '_process_has_visible_window', return_value=False), \
             patch('core.orchestrator.capture_snapshot') as mock_snap:
            snap = MagicMock()
            # Simulate no matching OCR text on screen
            snap.ocr_regions = []
            mock_snap.return_value = snap
            # 'launch Firefox' triggers. Process not found → False.
            result = orch._verify_launch(_sub_vl("launch Firefox browser"))
            # Firefox is in _PROCESS_MAP_WINDOWS and _APP_SIGNALS; process not found → False
            assert result is False


# These exercise the process-check branch for apps in _PROCESS_MAP_WINDOWS.
class TestVerifyLaunchAppKeywordCheck:

    def test_open_notepad_triggers_verification(self):
        """'open notepad' has 'open' + 'notepad' (known app) → triggers verification.
        Process check is mocked to fail → _verify_launch returns False.
        This confirms the verification path was entered (not skipped).
        """
        orch = _make_orch_vl()
        with patch('core.orchestrator.time.sleep'), \
             patch.object(orch, '_is_process_running', return_value=False), \
             patch.object(orch, '_process_has_visible_window', return_value=False):
            result = orch._verify_launch(_sub_vl("open notepad"))
        assert result is False, (
            "'open notepad' must trigger launch verification; "
            "with process not found it should return False"
        )

    def test_open_calculator_triggers_verification(self):
        """'open calculator' → triggers verification (process not found → False)."""
        orch = _make_orch_vl()
        with patch('core.orchestrator.time.sleep'), \
             patch.object(orch, '_is_process_running', return_value=False), \
             patch.object(orch, '_process_has_visible_window', return_value=False):
            result = orch._verify_launch(_sub_vl("open calculator"))
        assert result is False

    def test_open_terminal_triggers_verification(self):
        """'open windows terminal' → triggers verification."""
        orch = _make_orch_vl()
        with patch('core.orchestrator.time.sleep'), \
             patch.object(orch, '_is_process_running', return_value=False), \
             patch.object(orch, '_process_has_visible_window', return_value=False):
            result = orch._verify_launch(_sub_vl("open windows terminal"))
        assert result is False

    def test_open_notepad_process_found_returns_true(self):
        """'open notepad' with Notepad process running → verification passes → True."""
        orch = _make_orch_vl()
        with patch('core.orchestrator.time.sleep'), \
             patch.object(orch, '_is_process_running', return_value=True), \
             patch.object(orch, '_process_has_visible_window', return_value=False):
            result = orch._verify_launch(_sub_vl("open notepad"))
        assert result is True


class TestVerifyLaunchForegroundOCR:

    def test_ocr_fallback_only_reads_foreground_regions(self):
        """For an app not in the curated process map, the OCR fallback must read
        snapshot.ocr_regions filtered to is_in_foreground=True, not raw OCR.
        """
        orch = _make_orch_vl()

        # "libreoffice" is in _APP_SIGNALS but not in _PROCESS_MAP_WINDOWS, so
        # _verify_launch falls through to the OCR-signal check unconditionally.
        with patch('core.orchestrator.capture_snapshot') as mock_snap:

            # Two regions: one foreground with the signal, one background without
            fg_region = MagicMock()
            fg_region.text = "LibreOffice"
            fg_region.is_in_foreground = True

            bg_region = MagicMock()
            bg_region.text = "SomethingElse"
            bg_region.is_in_foreground = False

            snap = MagicMock()
            snap.ocr_regions = [fg_region, bg_region]
            mock_snap.return_value = snap

            result = orch._verify_launch(_sub_vl("launch libreoffice"))
            # "LibreOffice" is in _APP_SIGNALS signals; foreground region has it → True
            assert result is True
            # Confirm capture_snapshot was called (not raw self._ocr.extract)
            assert mock_snap.called

    def test_ocr_fallback_ignores_background_region_signal(self):
        """If the matching signal word appears only in a BACKGROUND region, it must
        NOT count as a confirmed launch.
        """
        orch = _make_orch_vl()

        with patch('core.orchestrator.capture_snapshot') as mock_snap:

            bg_region = MagicMock()
            bg_region.text = "LibreOffice"
            bg_region.is_in_foreground = False  # background — must be ignored

            fg_region = MagicMock()
            fg_region.text = "SomeOtherWord"
            fg_region.is_in_foreground = True

            snap = MagicMock()
            snap.ocr_regions = [bg_region, fg_region]
            mock_snap.return_value = snap

            result = orch._verify_launch(_sub_vl("launch libreoffice"))
            # "LibreOffice" is only in background → not confirmed → False
            assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# Plan queue — plan_steps() returns the whole remaining sequence in ONE LLM
# call; the orchestrator executes it from a queue (no per-step planning call)
# and flushes the queue whenever a step fails or its outcome is uncertain, so
# the next step is always planned against the live screen.
# ═══════════════════════════════════════════════════════════════════════════

_FAILED_CLEAR = ReflectionResult(
    success=False, confidence=0.9, observation="wrong outcome",
    error_description="did not produce expected result", should_retry=False,
    recovery_hint="", ocr_text="",
)


class TestPlanQueue:

    def test_batch_executes_without_replanning(self):
        """A 3-step batch runs on ONE planning call plus one goal-check call."""
        a = _step_loop("click", target="A")
        b = _step_loop("key_press", target=None, key="enter")
        c = _step_loop("type", target=None, value="hello")
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(side_effect=[[a, b, c], None])
        assert orch._execute_subtask(_subtask_loop()) is True
        assert orch.planner.plan_steps.call_count == 2
        assert orch.actor.execute.call_count == 3

    def test_queue_flushed_on_step_failure(self):
        """A confident failure drops the queued remainder and re-plans live."""
        a = _step_loop("click", target="A")
        b = _step_loop("key_press", target=None, key="enter")
        recovery = _step_loop("click", target="B")
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(
            side_effect=[[a, b], [recovery], None])
        orch.reflector.verify = MagicMock(
            side_effect=[_FAILED_CLEAR, _SUCCESS])
        assert orch._execute_subtask(_subtask_loop()) is True
        # b was queued behind the failed a and must never execute
        assert orch.actor.execute.call_count == 2
        assert orch.planner.plan_steps.call_count == 3

    def test_queue_flushed_on_uncertain_nonidempotent(self):
        """An uncertain-but-accepted type drops the queue: the 'next step
        verifies live' guarantee requires a fresh plan."""
        t = _step_loop("type", target=None, value="hello world")
        b = _step_loop("key_press", target=None, key="enter")
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(side_effect=[[t, b], None])
        orch.reflector.verify = MagicMock(return_value=_UNCERTAIN)
        assert orch._execute_subtask(_subtask_loop()) is True
        # only the type step ran; enter was dropped with the stale queue
        assert orch.actor.execute.call_count == 1
        assert orch.planner.plan_steps.call_count == 2
