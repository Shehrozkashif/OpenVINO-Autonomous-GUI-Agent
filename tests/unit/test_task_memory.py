# tests/unit/test_task_memory.py
"""Tests for TaskMemory checkpointing (resume support for long tasks).

Live failure this guards against: a 20-minute multi-subtask task fails at
subtask 7/8 and a re-run starts from scratch, repeating work whose effects are
already on the machine (duplicate files, double-typed text). Checkpoints record
completed subtasks so the router plans only the remaining work.
"""
import sys
import time

sys.path.insert(0, ".")

from memory.task_memory import TaskMemory


def _memory(tmp_path) -> TaskMemory:
    return TaskMemory(db_path=str(tmp_path / "test_memory.db"))


class TestCheckpointRoundTrip:

    def test_save_then_load_returns_completed_descs(self, tmp_path):
        mem = _memory(tmp_path)
        mem.save_checkpoint("schedule a zoom meeting", ["open Zoom", "click Schedule"])
        assert mem.load_checkpoint("schedule a zoom meeting") == [
            "open Zoom", "click Schedule",
        ]

    def test_load_unknown_instruction_returns_none(self, tmp_path):
        mem = _memory(tmp_path)
        assert mem.load_checkpoint("never ran") is None

    def test_save_overwrites_previous_checkpoint(self, tmp_path):
        mem = _memory(tmp_path)
        mem.save_checkpoint("task", ["a"])
        mem.save_checkpoint("task", ["a", "b"])
        assert mem.load_checkpoint("task") == ["a", "b"]

    def test_empty_completed_list_loads_as_none(self, tmp_path):
        # A checkpoint with nothing completed carries no resume value.
        mem = _memory(tmp_path)
        mem.save_checkpoint("task", [])
        assert mem.load_checkpoint("task") is None


class TestCheckpointLifecycle:

    def test_clear_removes_checkpoint(self, tmp_path):
        mem = _memory(tmp_path)
        mem.save_checkpoint("task", ["a"])
        mem.clear_checkpoint("task")
        assert mem.load_checkpoint("task") is None

    def test_clear_unknown_instruction_is_a_noop(self, tmp_path):
        mem = _memory(tmp_path)
        mem.clear_checkpoint("never ran")  # must not raise

    def test_stale_checkpoint_is_ignored(self, tmp_path):
        # The desktop state a 2-hour-old checkpoint describes is gone —
        # resuming from it would skip work that actually needs redoing.
        mem = _memory(tmp_path)
        mem.save_checkpoint("task", ["a"])
        mem.conn.execute(
            "UPDATE task_checkpoints SET updated_at = ?", (time.time() - 7200,)
        )
        mem.conn.commit()
        assert mem.load_checkpoint("task") is None
        assert mem.load_checkpoint("task", max_age_s=10_000) == ["a"]

    def test_checkpoints_are_per_instruction(self, tmp_path):
        mem = _memory(tmp_path)
        mem.save_checkpoint("task one", ["a"])
        mem.save_checkpoint("task two", ["x", "y"])
        mem.clear_checkpoint("task one")
        assert mem.load_checkpoint("task one") is None
        assert mem.load_checkpoint("task two") == ["x", "y"]
