# core/history.py
"""A record of finished tasks — for the operator, not for the agent.

One SQLite table (`tasks`) holding the instructions that completed cleanly,
how long they took, and the subtask list that worked. The UI's Home, Sessions,
Workflows and Memory pages read it; nothing in the agent loop reads it back.

That one-way rule is deliberate. Earlier versions fed past runs back into
planning — a "similar past task" hint to the router and "this target failed
before" hints to the planner. Both changed what the agent did based on a run
the user could not see, and the router hint was observed replacing a proper
decomposition with a stale plan. The current agent decides from the live
screen only; this file just remembers what happened.
"""
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = "data/task_history.db"


class TaskHistory:
    """Append-only log of successful tasks, keyed by instruction text."""

    def __init__(self, db_path: str = DEFAULT_DB):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT NOT NULL UNIQUE,
                steps_json TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                last_used REAL,
                avg_duration_s REAL
            )
        """)
        self.conn.commit()

    def store_successful_task(self, instruction: str, subtasks: list, duration_s: float):
        """Record a clean completion. Re-running the same instruction bumps its
        count and rolls the average duration rather than adding a second row.
        """
        self.conn.execute(
            "INSERT INTO tasks (instruction, steps_json, last_used, avg_duration_s) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(instruction) DO UPDATE SET "
            "  steps_json     = excluded.steps_json, "
            "  success_count  = success_count + 1, "
            "  last_used      = excluded.last_used, "
            "  avg_duration_s = (avg_duration_s + excluded.avg_duration_s) / 2",
            (
                instruction,
                json.dumps([s.model_dump() for s in subtasks]),
                time.time(),
                duration_s,
            ),
        )
        self.conn.commit()

    def get_recent_tasks(self, limit: int = 20) -> list[dict]:
        """The most recently completed tasks, newest first."""
        rows = self.conn.execute(
            "SELECT id, instruction, steps_json, success_count, last_used, avg_duration_s "
            "FROM tasks ORDER BY last_used DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": row_id,
                "instruction": instruction,
                "steps": json.loads(steps_json),
                "success_count": success_count,
                "last_used": last_used,
                "avg_duration_s": avg_duration_s,
            }
            for row_id, instruction, steps_json, success_count, last_used, avg_duration_s in rows
        ]
