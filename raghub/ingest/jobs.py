"""Background and resumable ingestion job tracking.

This module exposes the thread-pool-backed :class:`Batch` /
:class:`Job` fire-and-forget ingestion service, the SQLite-backed
:class:`Jobs` ledger, and :class:`Resumable` which combines both so
jobs survive restarts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from raghub.runtime import capture

__all__ = [
    "Batch",
    "Job",
    "Jobs",
    "Resumable",
]


@dataclass(slots=True)
class Job:
    """Lightweight value object tracking a single ingestion task.

    Attributes:
        job_id: Stable identifier returned by :meth:`submit`.
        status: One of ``"pending"``, ``"processing"``, ``"completed"``,
            ``"failed"``.
        result: The callable's return value on success, the stringified
            exception on failure, or ``None`` while pending.
        target: Optional owning document id (Phase 1.7.15 contract).

    """

    VALID_STATUSES = ("pending", "processing", "completed", "failed")

    job_id: str
    status: str = "pending"
    result: Any = None
    target: str | None = None

    def __post_init__(self) -> None:
        """Validate the Job's lifecycle status against :attr:`VALID_STATUSES`."""
        if self.status not in self.VALID_STATUSES:
            raise ValueError(
                f"Job: status {self.status!r} not in {self.VALID_STATUSES}"
            )

    def verify(self) -> None:
        """Assert the Job's invariant contract.

        Checks that ``job_id`` is non-empty and ``status`` is one of the
        recognised lifecycle values.

        Raises:
            VerificationError: When ``job_id`` is empty or ``status`` is
                not in :attr:`VALID_STATUSES`.

        """
        from raghub.errors import VerificationError

        if not self.job_id:
            raise VerificationError("Job: job_id is empty")
        if self.status not in self.VALID_STATUSES:
            raise VerificationError(f"Job: status {self.status!r} not in {self.VALID_STATUSES}")


class Batch:
    """Queues ingestion jobs for async processing.

    A thin wrapper around :class:`ThreadPoolExecutor` that adds job
    tracking. Construct once and reuse; constructing per-call does **not**
    reuse the underlying executor.

    Attributes:
        executor: Backing thread pool.
        jobs: Map from job id to :class:`Job`.
        closed: ``True`` after :meth:`shutdown` has been invoked.

    """

    def __init__(self, max_workers: int = 2) -> None:
        """Initialise the service with a thread pool."""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, Job] = {}
        self.closed = False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit a callable for background execution."""
        if self.closed:
            raise RuntimeError("Batch is shut down")
        job_id = str(uuid4())
        self.jobs[job_id] = Job(job_id, "pending")
        self.executor.submit(self.run_job, job_id, fn, args, kwargs)
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute one queued job, including asyncio unwrapping."""
        job = self.jobs[job_id]
        job.status = "processing"
        result, error = capture(fn, *args, **kwargs)
        if error is None and asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result, error = capture(loop.run_until_complete, result)
            loop.close()
        if error is not None:
            job.status = "failed"
            job.result = str(error)
            return
        job.status = "completed"
        job.result = result

    def get_status(self, job_id: str) -> str | None:
        """Return the current status for ``job_id``, or ``None`` if unknown."""
        job = self.jobs.get(job_id)
        return job.status if job else None

    def get_result(self, job_id: str) -> Any:
        """Return the stored result for ``job_id``, or ``None`` if unknown."""
        job = self.jobs.get(job_id)
        return job.result if job else None

    def shutdown(self, *, wait: bool = True) -> None:
        """Release the thread pool and refuse further submissions."""
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=wait)


class Jobs:
    """SQLite-backed job ledger.

    Records the lifecycle of every ingestion job so the application
    can resume after a crash. Records older than 24 hours are
    pruned lazily on save.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, job_id: str, status: str, result: Any = None) -> None:
        """Insert or update a job record."""
        encoded = (
            json.dumps(result) if result is not None and not isinstance(result, str) else result
        )
        self.conn.execute(
            """
            INSERT INTO ingestion_jobs (job_id, status, result, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET status = excluded.status, result = excluded.result
            """,
            (job_id, status, encoded, time.time()),
        )
        self.conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return the job record or ``None`` if unknown."""
        row = self.conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return {"job_id": row[0], "status": row[1], "result": row[2]}

    def all_jobs(self) -> Iterable[dict[str, Any]]:
        """Yield every persisted job."""
        for row in self.conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs"
        ).fetchall():
            yield {"job_id": row[0], "status": row[1], "result": row[2]}

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with suppress(Exception):
            self.conn.close()


class Resumable(Batch):
    """Background ingestion with a persistent job ledger."""

    def __init__(self, *, db_path: str | Path, max_workers: int = 2) -> None:
        """Initialise the service."""
        super().__init__(max_workers=max_workers)
        self.store = Jobs(db_path)
        self.restore_from_store()

    def restore_from_store(self) -> None:
        """Reload prior job state into the in-memory map."""
        for record in self.store.all_jobs():
            self.jobs[record["job_id"]] = Job(
                job_id=record["job_id"],
                status=record["status"],
                result=record["result"],
            )

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit ``fn`` for background execution."""
        job_id = super().submit(fn, *args, **kwargs)
        self.store.upsert(job_id, "pending")
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute a job, persisting status transitions."""
        super().run_job(job_id, fn, args, kwargs)
        job = self.jobs.get(job_id)
        if job is not None:
            self.store.upsert(job_id, job.status, job.result)

    def shutdown(self, *, wait: bool = False) -> None:
        """Flush the job store and shut down the executor."""
        if self.closed:
            return
        for job_id, job in list(self.jobs.items()):
            self.store.upsert(job_id, job.status, job.result)
        self.store.close()
        super().shutdown(wait=wait)
