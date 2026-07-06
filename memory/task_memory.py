# memory/task_memory.py
"""SQLite-backed task memory — two tables:

  tasks            — successful task sequences (used for memory hints to router)
  failure_patterns — steps/targets that failed and how they were recovered
                     (used as planning hints to avoid repeating known failures)

Similarity is plain difflib string matching: the feature exists to recognise
the user re-running a near-identical instruction, which is exactly string
similarity — no embedding model (and its torch stack, startup cost, and
network dependency) needed for that.
"""
import difflib
import json
import sqlite3
import time
from pathlib import Path


class TaskMemory:
    def __init__(self, db_path: str = "memory/interaction_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instruction TEXT NOT NULL,
                embedding BLOB,
                steps_json TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                last_used REAL,
                avg_duration_s REAL
            )
        """)
        # Episodic failure memory — what targets/actions fail and why
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS failure_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                action_type TEXT,
                error TEXT,
                app_context TEXT,
                recovery_hint TEXT,
                fail_count INTEGER DEFAULT 1,
                last_seen REAL
            )
        """)
        # In-flight task progress — lets an interrupted/failed long task resume
        # from its last completed subtask instead of starting over.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS task_checkpoints (
                instruction TEXT PRIMARY KEY,
                completed_json TEXT NOT NULL,
                updated_at REAL
            )
        """)
        self.conn.commit()

    def store_successful_task(self, instruction: str, subtasks: list, duration_s: float):
        steps_json = json.dumps([s.model_dump() for s in subtasks])

        existing = self.find_similar(instruction, threshold=0.95)
        if existing:
            self.conn.execute("""
                UPDATE tasks SET success_count = success_count + 1,
                last_used = ?, avg_duration_s = (avg_duration_s + ?) / 2
                WHERE id = ?
            """, (time.time(), duration_s, existing["id"]))
        else:
            self.conn.execute("""
                INSERT INTO tasks (instruction, steps_json, last_used, avg_duration_s)
                VALUES (?, ?, ?, ?)
            """, (instruction, steps_json, time.time(), duration_s))
        self.conn.commit()

    def find_similar(self, instruction: str, threshold: float = 0.85) -> dict | None:
        """Find a similar past task by string ratio. None if nothing above threshold."""
        rows = self.conn.execute(
            "SELECT id, instruction, steps_json FROM tasks"
        ).fetchall()

        query = instruction.lower().strip()
        best_score, best_row = 0.0, None
        for row_id, inst, steps_json in rows:
            score = difflib.SequenceMatcher(None, query, inst.lower().strip()).ratio()
            if score > best_score:
                best_score = score
                best_row = {"id": row_id, "instruction": inst,
                            "steps": json.loads(steps_json), "similarity": score}

        return best_row if (best_row and best_score >= threshold) else None

    def get_recent_tasks(self, limit: int = 20) -> list:
        """Return the most recently used tasks, newest first."""
        rows = self.conn.execute(
            "SELECT id, instruction, steps_json, success_count, last_used, avg_duration_s "
            "FROM tasks ORDER BY last_used DESC LIMIT ?",
            (limit,)
        ).fetchall()
        result = []
        for row_id, instruction, steps_json, success_count, last_used, avg_duration_s in rows:
            result.append({
                "id": row_id,
                "instruction": instruction,
                "steps": json.loads(steps_json),
                "success_count": success_count,
                "last_used": last_used,
                "avg_duration_s": avg_duration_s,
            })
        return result

    # ── Task checkpoints (resume support for long tasks) ─────────────────────

    def save_checkpoint(self, instruction: str, completed_descs: list[str]):
        """Persist the descriptions of subtasks completed so far for `instruction`."""
        self.conn.execute(
            "INSERT INTO task_checkpoints (instruction, completed_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(instruction) DO UPDATE SET completed_json = excluded.completed_json, "
            "updated_at = excluded.updated_at",
            (instruction, json.dumps(completed_descs), time.time()),
        )
        self.conn.commit()

    def load_checkpoint(self, instruction: str, max_age_s: float = 1800.0) -> list[str] | None:
        """Return the completed-subtask descriptions of a recent interrupted run
        of this exact instruction, or None. Stale checkpoints (older than
        max_age_s) are ignored — the desktop state they describe is gone.
        """
        row = self.conn.execute(
            "SELECT completed_json, updated_at FROM task_checkpoints WHERE instruction = ?",
            (instruction,),
        ).fetchone()
        if not row:
            return None
        completed_json, updated_at = row
        if time.time() - updated_at > max_age_s:
            return None
        completed = json.loads(completed_json)
        return completed or None

    def clear_checkpoint(self, instruction: str):
        self.conn.execute(
            "DELETE FROM task_checkpoints WHERE instruction = ?", (instruction,)
        )
        self.conn.commit()

    # ── Failure memory ────────────────────────────────────────────────────────

    def store_failure_pattern(
        self,
        target: str,
        action_type: str,
        error: str,
        app_context: str = "",
        recovery_hint: str = "",
    ):
        """Record that a specific target/action combination failed.
        Repeated failures on the same target increment the counter.
        """
        existing = self.conn.execute(
            "SELECT id, fail_count FROM failure_patterns "
            "WHERE target = ? AND action_type = ?",
            (target, action_type)
        ).fetchone()

        if existing:
            self.conn.execute(
                "UPDATE failure_patterns SET fail_count = fail_count + 1, "
                "last_seen = ?, error = ?, recovery_hint = ? WHERE id = ?",
                (time.time(), error, recovery_hint, existing[0])
            )
        else:
            self.conn.execute(
                "INSERT INTO failure_patterns "
                "(target, action_type, error, app_context, recovery_hint, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (target, action_type, error, app_context, recovery_hint, time.time())
            )
        self.conn.commit()

    def get_failure_hints(self, description: str, limit: int = 5) -> list[str]:
        """Return a list of failure hints relevant to the given step description.
        Matches by checking if any known-failed target appears in the description.
        Used by the planner to avoid repeating past failures.
        """
        rows = self.conn.execute(
            "SELECT target, action_type, error, recovery_hint, fail_count "
            "FROM failure_patterns ORDER BY fail_count DESC, last_seen DESC LIMIT 50"
        ).fetchall()

        desc_lower = description.lower()
        hints = []
        for target, action_type, error, recovery_hint, fail_count in rows:
            if target.lower() in desc_lower:
                hint = f"WARNING: '{target}' ({action_type}) failed {fail_count}× before"
                if error:
                    hint += f" — error: {error[:80]}"
                if recovery_hint:
                    hint += f" — try: {recovery_hint}"
                hints.append(hint)
                if len(hints) >= limit:
                    break

        return hints
