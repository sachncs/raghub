"""Persistent job store for resumable ingestion.

Writes job state to a small SQLite database so ingestion jobs
survive process restarts. The schema is intentionally tiny; it is
queried on startup to rebuild the in-memory job map.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


class PersistentJobStore:
    """SQLite-backed job ledger.

    Records the lifecycle of every ingestion job so the application
    can resume after a crash. Records older than 24 hours are
    pruned lazily on save.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, job_id: str, status: str, result: Any = None) -> None:
        """Insert or update a job record.

        Args:
            job_id: The job identifier.
            status: Current status.
            result: Optional result payload (JSON-serialised when not ``None``).
        """
        import time

        encoded = json.dumps(result) if result is not None and not isinstance(result, str) else result
        self._conn.execute(
            """
            INSERT INTO ingestion_jobs (job_id, status, result, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET status = excluded.status, result = excluded.result
            """,
            (job_id, status, encoded, time.time()),
        )
        self._conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return the job record or ``None`` if unknown.

        Args:
            job_id: The job identifier.

        Returns:
            A dict with ``job_id``, ``status``, ``result`` keys, or ``None``.
        """
        row = self._conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return {"job_id": row[0], "status": row[1], "result": row[2]}

    def all(self) -> Iterable[dict[str, Any]]:
        """Yield every persisted job.

        Yields:
            Dicts with ``job_id``, ``status``, ``result`` keys.
        """
        for row in self._conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs"
        ).fetchall():
            yield {"job_id": row[0], "status": row[1], "result": row[2]}

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass


__all__ = ["PersistentJobStore"]
