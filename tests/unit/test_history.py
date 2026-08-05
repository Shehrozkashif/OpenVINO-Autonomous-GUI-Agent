# tests/unit/test_history.py
"""Tests for TaskHistory — the record of completed tasks the UI reads.

Nothing in the agent loop reads this back, so the contract is small: a clean
completion is recorded once, re-running the same instruction updates that one
row, and the UI's listing returns newest first.
"""
from unittest.mock import MagicMock

import pytest

from core.history import TaskHistory


def _subtask(description: str) -> MagicMock:
    sub = MagicMock()
    sub.model_dump.return_value = {"description": description}
    return sub


@pytest.fixture
def history(tmp_path) -> TaskHistory:
    return TaskHistory(db_path=str(tmp_path / "history.db"))


class TestRecording:
    def test_completed_task_is_stored_with_its_steps(self, history):
        history.store_successful_task("open notepad", [_subtask("launch Notepad")], 12.0)
        (row,) = history.get_recent_tasks()
        assert row["instruction"] == "open notepad"
        assert row["steps"] == [{"description": "launch Notepad"}]
        assert row["success_count"] == 1
        assert row["avg_duration_s"] == 12.0

    def test_repeat_run_updates_the_same_row(self, history):
        sub = _subtask("launch Notepad")
        history.store_successful_task("open notepad", [sub], 60.0)
        history.store_successful_task("open notepad", [sub], 80.0)
        rows = history.get_recent_tasks()
        assert len(rows) == 1
        assert rows[0]["success_count"] == 2
        assert rows[0]["avg_duration_s"] == 70.0    # rolling average of 60 and 80

    def test_different_instructions_are_separate_rows(self, history):
        history.store_successful_task("open notepad", [_subtask("a")], 1.0)
        history.store_successful_task("open paint", [_subtask("b")], 2.0)
        assert len(history.get_recent_tasks()) == 2


class TestListing:
    def test_newest_first(self, history):
        history.store_successful_task("first", [_subtask("a")], 1.0)
        history.store_successful_task("second", [_subtask("b")], 1.0)
        assert [t["instruction"] for t in history.get_recent_tasks()] == ["second", "first"]

    def test_limit_is_honoured(self, history):
        for i in range(5):
            history.store_successful_task(f"task {i}", [_subtask("a")], 1.0)
        assert len(history.get_recent_tasks(limit=2)) == 2

    def test_empty_history_lists_nothing(self, history):
        assert history.get_recent_tasks() == []
