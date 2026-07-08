# tests/unit/test_orchestrator.py
"""Unit tests for core/orchestrator.py — TaskOrchestrator.

Organised by concern (each section was originally its own file; the helper
functions below are suffixed per section since several sections build a
differently-configured mock orchestrator under the same conceptual name).

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
from core.orchestrator import DEDUP_LIMIT_BY_ACTION_TYPE, OrchestratorConfig, TaskOrchestrator
from core.protocols import ActionStep, SubTask

sys.path.insert(0, ".")


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


class TestInterleavedRepeats:
    """Repeats are counted as per-subtask TOTALS, not consecutively.

    An interleaved different step must NOT launder a loop: A-B-A-B ping-pong
    (seen live: hallucinated verifier success alternating 'click NEW' /
    'click Schedule' for 10 minutes) trips the guard exactly like A-A-A.
    Different signatures still count independently.
    """

    def test_type_repeat_survives_interleaved_click(self):
        """Type ×2 allowed (limit 1 = one repeat); an interleaved click does
        not reset the count, so the 3rd type total fires the guard.
        Planner called 4 times (guard fires while recording step 4).
        """
        ts = _step_loop("type", target=None, value="hello")
        cs = _step_loop("click", target="Foo")
        orch = _make_orch_loop([ts, ts, cs, ts])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 4

    def test_click_repeat_survives_interleaved_type(self):
        """Click ×3 allowed (limit 2); an interleaved type does not reset the
        count, so the 4th click total fires the guard at step 5.
        """
        cs = _step_loop("click", target="Btn")
        ts = _step_loop("type", target=None, value="x")
        orch = _make_orch_loop([cs, cs, cs, ts, cs])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 5

    def test_different_targets_count_independently(self):
        """Clicks on Alpha and Beta have different signatures — three Alpha
        clicks plus one Beta click stay within Alpha's allowance (3 total),
        so no trigger: all 4 steps run, then None ends the subtask.
        """
        ca = _step_loop("click", target="Alpha")
        cb = _step_loop("click", target="Beta")
        orch = _make_orch_loop([ca, ca, cb, ca])
        result = orch._execute_subtask(_subtask_loop())
        assert result is True
        assert orch.planner.plan_steps.call_count == 5  # 4 steps + None

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
        verifies live' guarantee requires a fresh plan.
        """
        t = _step_loop("type", target=None, value="hello world")
        b = _step_loop("key_press", target=None, key="enter")
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(side_effect=[[t, b], None])
        orch.reflector.verify = MagicMock(return_value=_UNCERTAIN)
        assert orch._execute_subtask(_subtask_loop()) is True
        # only the type step ran; enter was dropped with the stale queue
        assert orch.actor.execute.call_count == 1
        assert orch.planner.plan_steps.call_count == 2


class TestSaveTargetDiskGate:
    """A "save as <path>" subtask has one ground truth: the file on disk.
    The planner's "goal achieved" must never overrule its absence.
    """

    def _save_subtask(self):
        return SubTask(
            id=3, depends_on=[],
            description="save the document as C:/Users/x/Desktop/a.txt",
        )

    def test_planner_done_rejected_when_file_missing(self):
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(return_value=None)
        orch._try_save_as = MagicMock(return_value=False)
        orch._file_saved_fresh = MagicMock(return_value=False)
        orch.config.visual_replan_after = 0
        assert orch._execute_subtask(self._save_subtask()) is False
        # every "done" claim was rejected until the failure limit tripped
        assert (orch.planner.plan_steps.call_count
                == orch.config.consecutive_failures_limit)

    def test_file_on_disk_short_circuits_before_planning(self):
        orch = _make_orch_loop([])
        orch.planner.plan_steps = MagicMock(return_value=None)
        orch._try_save_as = MagicMock(return_value=False)
        orch._file_saved_fresh = MagicMock(return_value=True)
        assert orch._execute_subtask(self._save_subtask()) is True
        orch.planner.plan_steps.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# _goal_already_satisfied — pre-plan goal check
#
# Live failure (AI-PC log 2026-07-05 14:25–14:34): one invoke opened the
# Outlook event form, but the subtask "click New event to open the form" kept
# re-planning that click for 8 more cycles — state-describing verifiers kept
# blessing the no-ops ("the form is visible → success") and nothing ever asked
# "is the goal already met?". The check must say yes there, and stay
# conservative everywhere else.
# ═══════════════════════════════════════════════════════════════════════════

from core.orchestrator import _SubtaskRun


def _goal_check_orch(llm_reply: str):
    orch = _make_orch_loop([])
    resp = MagicMock()
    resp.content = llm_reply
    orch.reflector.client.query_llm = MagicMock(return_value=resp)
    return orch


def _run_state(**kw):
    return _SubtaskRun(started_at=0.0, screen_context=(
        "CLICKABLE CONTROLS: 'New event' [Document], 'Save' [Button], "
        "'Add title' [Edit], 'Event body' [Edit]"
    ), **kw)


def _subtask(desc="click the New event button to open the schedule-meeting form"):
    return SubTask(id=1, description=desc, depends_on=[])


class TestGoalAlreadySatisfied:

    def test_confident_yes_returns_true(self):
        orch = _goal_check_orch(
            '{"satisfied": true, "confidence": 0.95, '
            '"evidence": "Save button and Add title field on screen"}'
        )
        assert orch._goal_already_satisfied(_run_state(), _subtask()) is True

    def test_low_confidence_yes_returns_false(self):
        orch = _goal_check_orch('{"satisfied": true, "confidence": 0.5}')
        assert orch._goal_already_satisfied(_run_state(), _subtask()) is False

    def test_no_returns_false(self):
        orch = _goal_check_orch('{"satisfied": false, "confidence": 0.9}')
        assert orch._goal_already_satisfied(_run_state(), _subtask()) is False

    def test_think_block_and_prose_around_json_tolerated(self):
        orch = _goal_check_orch(
            '<think>the form is open already</think>\n'
            'Here is my verdict: {"satisfied": true, "confidence": 0.9, '
            '"evidence": "form controls visible"}'
        )
        assert orch._goal_already_satisfied(_run_state(), _subtask()) is True

    def test_garbage_reply_returns_false(self):
        orch = _goal_check_orch("the goal seems achieved to me")
        assert orch._goal_already_satisfied(_run_state(), _subtask()) is False

    def test_launch_subtasks_excluded(self):
        orch = _goal_check_orch('{"satisfied": true, "confidence": 1.0}')
        run = _run_state(is_launch_goal=True)
        assert orch._goal_already_satisfied(run, _subtask("open Outlook")) is False
        orch.reflector.client.query_llm.assert_not_called()

    def test_deterministic_subtask_kinds_excluded(self):
        orch = _goal_check_orch('{"satisfied": true, "confidence": 1.0}')
        for kw in ({"is_cmd_subtask": True}, {"save_target": "C:/x.txt"},
                   {"type_payload": "hello"}):
            assert orch._goal_already_satisfied(_run_state(**kw), _subtask()) is False
        orch.reflector.client.query_llm.assert_not_called()

    def test_unsatisfied_evidence_fed_to_planner_context(self):
        """Live failure 2026-07-05 16:31+: after a replan the planner redid the
        whole fill-details subtask (re-set the title, re-clicked the date)
        because the goal check's "what's missing" evidence was discarded.
        """
        orch = _goal_check_orch(
            '{"satisfied": false, "confidence": 0.9, '
            '"evidence": "subject is set, but start time is not 3:00 PM"}'
        )
        run = _run_state()
        assert orch._goal_already_satisfied(run, _subtask()) is False
        assert "start time is not 3:00 PM" in run.screen_context
        assert "GOAL CHECK (what is still missing)" in run.screen_context

    def test_satisfied_verdict_leaves_context_untouched(self):
        orch = _goal_check_orch(
            '{"satisfied": true, "confidence": 0.95, "evidence": "form open"}'
        )
        run = _run_state()
        before = run.screen_context
        assert orch._goal_already_satisfied(run, _subtask()) is True
        assert run.screen_context == before


class TestStopEventHaltsAttempts:
    """Live failure 2026-07-05 16:06: EMERGENCY STOP fired mid-step, but the
    agent kept grounding/clicking for 38 more seconds — only the outer step
    loop checked the stop event. The attempt loop must bail immediately.
    """

    def test_attempt_loop_exits_on_stop_event(self):
        orch = _make_orch_loop([])
        orch._stop_event.set()
        step = ActionStep(
            id=1, subtask_id=1, action_type="click", target="New event",
            value=None, key=None, description="click", verification="",
        )
        run = _SubtaskRun(started_at=0.0)
        assert orch._run_step_attempts(run, MagicMock(), step) == "step_failed"
        orch.actor.execute.assert_not_called()

    def test_unsatisfied_evidence_stored_for_final_report(self):
        orch = _goal_check_orch(
            '{"satisfied": false, "confidence": 0.9, '
            '"evidence": "only a sign-in screen is visible"}'
        )
        orch._goal_already_satisfied(_run_state(), _subtask())
        assert orch._last_goal_evidence == "only a sign-in screen is visible"


# ═══════════════════════════════════════════════════════════════════════════
# Invoke-dead blacklist
#
# Live failure (AI-PC 18:41-18:44): 'Invite to Teams' was pattern-invoked six
# times with zero screen change — WebView2 providers accept Invoke/Toggle
# without doing anything, and the coordinate blacklist never fires because no
# pixel is involved. One failed verify must push retries onto the pixel path.
# ═══════════════════════════════════════════════════════════════════════════

class TestInvokeDeadBlacklist:

    def _orch(self):
        orch = _make_orch_loop([])
        orch.grounder.ground = MagicMock(return_value=GroundingResult(
            found=True, confidence=1.0, x=533, y=752,
            latency_ms=5.0, target="Invite to Teams", method="uia",
            element_type="foreground_interactive",
        ))
        return orch

    def _click_step(self):
        return ActionStep(
            id=1, subtask_id=1, action_type="click", target="Invite to Teams",
            value=None, key=None, description="click invite", verification="",
        )

    def test_failed_invoke_blacklists_target_and_retry_uses_pixels(self):
        orch = self._orch()
        step = self._click_step()
        with patch("core.windows_uia.invoke_element", return_value=True):
            assert orch._execute_step(step) is True
            assert orch._last_was_invoke is True

        # Reflection says the screen never changed → invoke path goes dead.
        run = _SubtaskRun(started_at=0.0)
        run.last_error = "Screen unchanged after click"
        reflection = MagicMock(success=False, confidence=0.98,
                               should_retry=True, observation="unchanged")
        orch.reflector.verify = MagicMock(return_value=reflection)
        orch._judge_reflection(run, step, non_idempotent=False,
                               pre_click_hash=None)
        assert "invite to teams" in orch._invoke_dead

        # Retry: invoke must be skipped; the actor performs a real click.
        with patch("core.windows_uia.invoke_element", return_value=True) as inv2:
            assert orch._execute_step(step) is True
            inv2.assert_not_called()
        orch.actor.execute.assert_called()

    def test_other_targets_still_invoke(self):
        orch = self._orch()
        orch._invoke_dead.add("invite to teams")
        step = ActionStep(
            id=2, subtask_id=1, action_type="click", target="Calendar",
            value=None, key=None, description="click calendar", verification="",
        )
        orch.grounder.ground = MagicMock(return_value=GroundingResult(
            found=True, confidence=1.0, x=100, y=200,
            latency_ms=5.0, target="Calendar", method="uia",
            element_type="foreground_interactive",
        ))
        with patch("core.windows_uia.invoke_element", return_value=True) as inv:
            assert orch._execute_step(step) is True
            inv.assert_called_once()

    def test_same_screen_verdict_cached_no_second_llm_call(self):
        """Latency: identical screens must not re-pay the goal-check LLM call
        (live: the same 'Save present but…' answer was recomputed ~10×).
        """
        orch = _goal_check_orch(
            '{"satisfied": false, "confidence": 0.9, '
            '"evidence": "attendee field is empty"}'
        )
        run = _run_state()
        base_ctx = run.screen_context
        assert orch._goal_already_satisfied(run, _subtask()) is False
        assert orch.reflector.client.query_llm.call_count == 1

        run.screen_context = base_ctx   # same screen next cycle
        assert orch._goal_already_satisfied(run, _subtask()) is False
        assert orch.reflector.client.query_llm.call_count == 1   # cache hit
        assert "attendee field is empty" in run.screen_context   # evidence re-fed

    def test_changed_screen_misses_cache(self):
        orch = _goal_check_orch('{"satisfied": false, "confidence": 0.9}')
        run = _run_state()
        orch._goal_already_satisfied(run, _subtask())
        run.screen_context = "completely different screen"
        orch._goal_already_satisfied(run, _subtask())
        assert orch.reflector.client.query_llm.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Invoke failures must not poison the pixel path, and occluded targets must
# fail fast with the blocker's name (regression, live AI-PC run 2026-07-06
# 07:26: Teams' 'Meeting created' popup covered the form; 'Send' invoke →
# delta=0 → the REAL button's point was dead-marked although no pixel was
# ever clicked → the retry's pixel click landed on a VLM title-bar guess).
# ═══════════════════════════════════════════════════════════════════════════

def _make_orch_occl(method="uia"):
    orch = _make_orch_loop([])
    orch.grounder.ground = MagicMock(return_value=GroundingResult(
        found=True, confidence=1.0, x=1404, y=168,
        latency_ms=5.0, target="Send", method=method,
        element_type="foreground_interactive",
    ))
    orch.grounder.mark_dead = MagicMock()
    return orch


def _send_click():
    return ActionStep(
        id=1, subtask_id=1, action_type="click", target="Send",
        value=None, key=None, description="click Send", verification="",
    )


def _unchanged_reflection():
    return MagicMock(
        success=False, confidence=0.98, should_retry=True,
        error_description="Screen unchanged after click",
    )


class TestInvokeFailureKeepsPixelAlive:

    def test_failed_invoke_does_not_mark_point_dead(self):
        orch = _make_orch_occl()
        step = _send_click()
        with patch("core.windows_uia.covering_element", return_value=None), \
             patch("core.windows_uia.invoke_element", return_value=True):
            assert orch._execute_step(step) is True
        assert orch._last_was_invoke is True

        run = _SubtaskRun(started_at=0.0)
        orch.reflector.verify = MagicMock(return_value=_unchanged_reflection())
        with patch("core.windows_uia.covering_element", return_value=None):
            orch._judge_reflection(run, step, non_idempotent=False,
                                   pre_click_hash=None)
        orch.grounder.mark_dead.assert_not_called()   # point never clicked
        assert "send" in orch._invoke_dead            # invoke path blacklisted

    def test_failed_pixel_click_still_marks_dead(self):
        orch = _make_orch_occl()
        step = _send_click()
        with patch("core.windows_uia.covering_element", return_value=None), \
             patch("core.windows_uia.invoke_element", return_value=False):
            assert orch._execute_step(step)           # falls to pixel click
        assert orch._last_was_invoke is False

        run = _SubtaskRun(started_at=0.0)
        orch.reflector.verify = MagicMock(return_value=_unchanged_reflection())
        with patch("core.windows_uia.covering_element", return_value=None):
            orch._judge_reflection(run, step, non_idempotent=False,
                                   pre_click_hash=None)
        orch.grounder.mark_dead.assert_called_once_with("Send", 1404, 168)

    def test_covered_point_failure_names_blocker_in_error(self):
        orch = _make_orch_occl()
        step = _send_click()
        with patch("core.windows_uia.covering_element", return_value=None), \
             patch("core.windows_uia.invoke_element", return_value=False):
            orch._execute_step(step)

        run = _SubtaskRun(started_at=0.0)
        orch.reflector.verify = MagicMock(return_value=_unchanged_reflection())
        with patch("core.windows_uia.covering_element",
                   return_value="Meeting created"):
            orch._judge_reflection(run, step, non_idempotent=False,
                                   pre_click_hash=None)
        assert "Meeting created" in run.last_error
        assert "overlay" in run.last_error


class TestOcclusionGate:

    def test_covered_uia_point_fails_fast_with_blocker_name(self):
        orch = _make_orch_occl(method="uia")
        step = _send_click()
        with patch("core.windows_uia.covering_element",
                   return_value="Meeting created") as cov, \
             patch("core.windows_uia.invoke_element") as inv:
            assert orch._execute_step(step) is False
            cov.assert_called_once()
            inv.assert_not_called()
        orch.actor.execute.assert_not_called()
        assert "Meeting created" in orch._exec_fail_reason

    def test_uncovered_point_proceeds_to_invoke(self):
        orch = _make_orch_occl(method="uia")
        step = _send_click()
        with patch("core.windows_uia.covering_element", return_value=None), \
             patch("core.windows_uia.invoke_element", return_value=True) as inv:
            assert orch._execute_step(step) is True
            inv.assert_called_once()

    def test_ocr_grounded_click_skips_hit_test(self):
        # OCR text rarely equals the accessible name — hit-testing there
        # would false-positive, so the gate only guards UIA groundings.
        orch = _make_orch_occl(method="ocr_fuzzy")
        step = _send_click()
        with patch("core.windows_uia.covering_element") as cov:
            assert orch._execute_step(step)
            cov.assert_not_called()
        orch.actor.execute.assert_called_once()

    def test_occlusion_reason_reaches_step_failure_record(self):
        orch = _make_orch_occl(method="uia")
        step = _send_click()
        run = _SubtaskRun(started_at=0.0)
        subtask = SubTask(id=1, description="click Send", depends_on=[])
        orch.config.max_retries_per_step = 1
        with patch("core.windows_uia.covering_element",
                   return_value="Meeting created"):
            outcome = orch._run_step_attempts(run, subtask, step)
        assert outcome == "step_failed"
        assert "Meeting created" in run.last_error


# ═══════════════════════════════════════════════════════════════════════════
# Planner-done needs evidence + typed text must land in the NAMED field
# (regression, live AI-PC run 2026-07-06 13:05: planner returned [] "goal
# achieved" for 'set the date to 07/08/2026' while Start date read
# '7/6/2026'; later an attendee email was typed into the Title field and the
# content-only type verify blessed it).
# ═══════════════════════════════════════════════════════════════════════════

def _plan_orch(planned):
    orch = _make_orch_loop([])
    orch.planner.plan_steps = MagicMock(return_value=planned)
    orch._ensure_anchor_foreground = MagicMock()
    return orch


class TestPlannerDoneNeedsEvidence:

    def test_empty_plan_with_no_actions_is_rejected(self):
        orch = _plan_orch([])
        run = _SubtaskRun(started_at=0.0)
        subtask = SubTask(id=1, description="set the date to 07/08/2026",
                          depends_on=[])
        outcome, step = orch._plan_next_step(run, subtask, [])
        assert outcome != "done"
        assert step is None
        assert any("no action was executed" in c for c in run.completed)

    def test_empty_plan_after_real_action_is_trusted(self):
        orch = _plan_orch([])
        run = _SubtaskRun(started_at=0.0)
        run.completed = ["[set_value] date set to 07/08/2026"]
        subtask = SubTask(id=1, description="set the date to 07/08/2026",
                          depends_on=[])
        outcome, _ = orch._plan_next_step(run, subtask, [])
        assert outcome == "done"

    def test_excluded_kind_gets_forced_goal_check(self):
        orch = _plan_orch([])
        run = _SubtaskRun(started_at=0.0)
        run.type_payload = "07/08/2026"
        subtask = SubTask(id=1, description="set the date to 07/08/2026",
                          depends_on=[])
        # Pre-plan check (excluded kind) → False; forced confirm → True.
        orch._goal_already_satisfied = MagicMock(side_effect=[False, True])
        outcome, _ = orch._plan_next_step(run, subtask, [])
        assert outcome == "done"
        _, kwargs = orch._goal_already_satisfied.call_args
        assert kwargs.get("skip_exclusions") is True

    def test_skip_exclusions_reaches_the_llm(self):
        orch = _goal_check_orch(
            '{"satisfied": true, "confidence": 0.9, "evidence": "date shows"}'
        )
        run = _run_state()
        run.type_payload = "07/08/2026"
        # Excluded normally: no LLM call, plain False.
        assert orch._goal_already_satisfied(run, _subtask()) is False
        assert orch.reflector.client.query_llm.call_count == 0
        # Forced: the LLM is consulted.
        assert orch._goal_already_satisfied(
            run, _subtask(), skip_exclusions=True
        ) is True
        assert orch.reflector.client.query_llm.call_count == 1


class TestTypedTextTargetCheck:

    def _info(self, name, value):
        return {"name": name, "value": value,
                "control_type": "EditControl", "rect": (0, 0, 10, 10)}

    def test_wrong_field_rejected(self, monkeypatch):
        orch = _make_orch_loop([])
        monkeypatch.setattr(
            "core.windows_uia.focused_element_info",
            lambda **kw: self._info(
                "Add title", "project discussionshehrozkashif57@gmail.com"),
        )
        assert orch._typed_text_in_focused_control(
            "shehrozkashif57@gmail.com", "Add required attendees"
        ) is False

    def test_right_field_accepted(self, monkeypatch):
        orch = _make_orch_loop([])
        monkeypatch.setattr(
            "core.windows_uia.focused_element_info",
            lambda **kw: self._info(
                "Add required attendees", "shehrozkashif57@gmail.com"),
        )
        assert orch._typed_text_in_focused_control(
            "shehrozkashif57@gmail.com", "Add required attendees"
        ) is True

    def test_no_target_checks_content_only(self, monkeypatch):
        orch = _make_orch_loop([])
        monkeypatch.setattr(
            "core.windows_uia.focused_element_info",
            lambda **kw: self._info("Whatever", "hello world"),
        )
        assert orch._typed_text_in_focused_control("hello world") is True


class TestLatencyGuards:
    """Two pure-waste patterns from the live 13:36 run: a 10-16 s goal check
    between every queued step (the planner is not even consulted there), and
    3 identical retries of deterministic tree actions.
    """

    def test_goal_check_skipped_while_queue_pending(self):
        orch = _plan_orch([])
        orch._goal_already_satisfied = MagicMock(return_value=True)
        run = _SubtaskRun(started_at=0.0)
        run.step_queue = [ActionStep(
            id=9, subtask_id=1, action_type="click", target="X",
            value=None, key=None, description="queued", verification="",
        )]
        outcome, step = orch._plan_next_step(
            run, SubTask(id=1, description="fill the form", depends_on=[]), [],
        )
        assert outcome == "step"
        assert step.description == "queued"
        orch._goal_already_satisfied.assert_not_called()

    def test_failed_tree_action_not_blind_retried(self):
        orch = _make_orch_loop([])
        orch.config.max_retries_per_step = 3
        orch._execute_step = MagicMock(return_value=False)
        step = ActionStep(
            id=1, subtask_id=1, action_type="select", target="Start time",
            value="3:00 PM", key=None, description="set time", verification="",
        )
        out = orch._run_step_attempts(
            _SubtaskRun(started_at=0.0),
            SubTask(id=1, description="set the start time", depends_on=[]),
            step,
        )
        assert out == "step_failed"
        orch._execute_step.assert_called_once()

    def test_failed_click_still_gets_retries(self):
        orch = _make_orch_loop([])
        orch.config.max_retries_per_step = 3
        orch._execute_step = MagicMock(return_value=False)
        step = ActionStep(
            id=1, subtask_id=1, action_type="click", target="Save",
            value=None, key=None, description="click save", verification="",
        )
        out = orch._run_step_attempts(
            _SubtaskRun(started_at=0.0),
            SubTask(id=1, description="click Save", depends_on=[]),
            step,
        )
        assert out == "step_failed"
        assert orch._execute_step.call_count == 3


class TestSelectMissReachesPlanner:
    """Regression (live 14:12 run): 'select GST' failed with the real labels
    visible only in the log — the planner's failure record said nothing, so
    it retried 'GST' until the replan budget died.
    """

    def test_miss_details_land_in_last_error(self, monkeypatch):
        orch = _make_orch_loop([])
        orch._execute_step = MagicMock(return_value=False)
        monkeypatch.setattr(
            "core.windows_uia.pop_select_miss",
            lambda: {"target": "Time zone", "option": "GST",
                     "items": ["(UTC-12:00) International Date Line West",
                               "(UTC-08:00) Pacific Time (US & Canada)"]},
        )
        run = _SubtaskRun(started_at=0.0)
        step = ActionStep(
            id=1, subtask_id=1, action_type="select", target="Time zone",
            value="GST", key=None, description="set tz", verification="",
        )
        out = orch._run_step_attempts(
            run, SubTask(id=1, description="set tz to GST", depends_on=[]),
            step,
        )
        assert out == "step_failed"
        assert "GST" in run.last_error
        assert "(UTC-08:00) Pacific Time (US & Canada)" in run.last_error

    def test_no_miss_keeps_generic_error(self, monkeypatch):
        orch = _make_orch_loop([])
        orch._execute_step = MagicMock(return_value=False)
        monkeypatch.setattr("core.windows_uia.pop_select_miss", lambda: None)
        run = _SubtaskRun(started_at=0.0)
        step = ActionStep(
            id=1, subtask_id=1, action_type="select", target="Time zone",
            value="GST", key=None, description="set tz", verification="",
        )
        orch._run_step_attempts(
            run, SubTask(id=1, description="set tz", depends_on=[]), step,
        )
        assert "execution failed" in run.last_error


class TestGoalCheckSeesLastAction:
    """Regression (live 14:12 run): the goal check re-failed an OPEN form
    with 'the button was not clicked' — the planner re-clicked 4x and a
    stray Escape closed the form. The check now sees the last verified
    action and must judge its end state.
    """

    def test_last_verified_action_named_in_prompt(self):
        orch = _goal_check_orch(
            '{"satisfied": true, "confidence": 0.9, "evidence": "form open"}'
        )
        run = _run_state()
        run.completed = ["[FAILED: something] earlier", "[click] Click 'Schedule a meeting'"]
        assert orch._goal_already_satisfied(run, _subtask()) is True
        prompt = orch.reflector.client.query_llm.call_args[0][0][1]["content"]
        assert "Last verified action" in prompt
        assert "Click 'Schedule a meeting'" in prompt

    def test_no_verified_action_no_extra_line(self):
        orch = _goal_check_orch(
            '{"satisfied": false, "confidence": 0.9, "evidence": "x"}'
        )
        run = _run_state()
        run.completed = ["[FAILED: y] only failures"]
        orch._goal_already_satisfied(run, _subtask())
        prompt = orch.reflector.client.query_llm.call_args[0][0][1]["content"]
        assert "Last verified action" not in prompt


class TestBlockingOverlayInContext:
    """Regression (live 15:14-15:24 run): the occlusion gate correctly
    refused to click Save under the 'Meeting created' popup, but its hint in
    the step-failure record was ignored for six ~60 s planning cycles while
    'Close' sat in the controls list. The proven overlay must enter the
    SCREEN CONTEXT, the channel the planner reliably follows.
    """

    def test_gate_records_overlay_for_context(self):
        orch = _make_orch_occl(method="uia")
        step = _send_click()
        with patch("core.windows_uia.covering_element",
                   return_value="Meeting created"):
            assert orch._execute_step(step) is False
        assert orch._blocking_overlay == ("Send", "Meeting created")

    def test_overlay_line_reaches_planner_context_once(self):
        orch = _plan_orch([ActionStep(
            id=1, subtask_id=1, action_type="click", target="Close",
            value=None, key=None, description="close popup", verification="",
        )])
        orch._blocking_overlay = ("Save", "Meeting created")
        run = _SubtaskRun(started_at=0.0)
        outcome, _ = orch._plan_next_step(
            run, SubTask(id=1, description="click Save", depends_on=[]), [],
        )
        assert outcome == "step"
        assert "BLOCKING OVERLAY" in run.screen_context
        assert "Meeting created" in run.screen_context
        assert orch._blocking_overlay is None   # one-shot until re-blocked

    def test_no_overlay_no_line(self):
        orch = _plan_orch([ActionStep(
            id=1, subtask_id=1, action_type="click", target="Save",
            value=None, key=None, description="save", verification="",
        )])
        run = _SubtaskRun(started_at=0.0)
        orch._plan_next_step(
            run, SubTask(id=1, description="click Save", depends_on=[]), [],
        )
        assert "BLOCKING OVERLAY" not in run.screen_context
