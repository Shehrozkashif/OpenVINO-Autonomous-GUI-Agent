# memory/task/task_memory.py
"""
SQLite-backed task memory.
Stores successful task sequences. Uses embeddings to find similar past tasks.
First use downloads ~80MB sentence-transformer model (all-MiniLM-L6-v2).
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np


class TaskMemory:
    def __init__(self, db_path: str = "memory/interaction_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._embedder = None

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
        self.conn.commit()

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def store_successful_task(self, instruction: str, subtasks: list, duration_s: float):
        embedding = self.embedder.encode(instruction).astype(np.float32).tobytes()
        # FIX: use model_dump() for Pydantic models, not __dict__
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
                INSERT INTO tasks (instruction, embedding, steps_json, last_used, avg_duration_s)
                VALUES (?, ?, ?, ?, ?)
            """, (instruction, embedding, steps_json, time.time(), duration_s))
        self.conn.commit()

    def find_similar(self, instruction: str, threshold: float = 0.85) -> Optional[dict]:
        """Find a semantically similar past task. Returns None if nothing above threshold."""
        query_emb = self.embedder.encode(instruction).astype(np.float32)
        rows = self.conn.execute(
            "SELECT id, instruction, embedding, steps_json FROM tasks"
        ).fetchall()

        best_score, best_row = 0.0, None
        for row_id, inst, emb_bytes, steps_json in rows:
            stored = np.frombuffer(emb_bytes, dtype=np.float32)
            score = float(np.dot(query_emb, stored) /
                         (np.linalg.norm(query_emb) * np.linalg.norm(stored)))
            if score > best_score:
                best_score = score
                best_row = {"id": row_id, "instruction": inst,
                            "steps": json.loads(steps_json), "similarity": score}

        return best_row if (best_row and best_score >= threshold) else None
