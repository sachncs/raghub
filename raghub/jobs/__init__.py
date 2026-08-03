"""Persistent ingestion queue.

Replaces the in-process :class:`raghub.ingest.Batch` threadpool with
a SQLite-backed :class:`PersistentQueue` plus a :class:`Worker`
runtime. Jobs survive process restarts, support exponential-backoff
retries, and emit a dead-letter on exhaustion.

This release ships only the SQLite backend; Celery / Arq / Dramatiq
are pluggable via the entry-point contract
``group="raghub.queues"`` (no first-class adapters ship).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

__all__ = [
    "Job",
    "JobStateError",
    "JobStatus",
    "PersistentQueue",
    "QueueSaturatedError",
    "SqliteQueue",
    "Worker",
]


class JobStatus(StrEnum):
    """Lifecycle states for one queued job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"


# Allowed transitions per the state machine in todo/v0.7.4.
_VALID_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.DEAD}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.DEAD}
    ),
    JobStatus.FAILED: frozenset({JobStatus.PENDING}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.DEAD: frozenset(),
}


class JobStateError(RuntimeError):
    """Raised when an illegal state transition is requested."""


@dataclass(frozen=True, slots=True)
class Job:
    """One queued unit of work."""

    id: str
    kind: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    last_error: str | None = None
    next_run_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    worker_id: str | None = None
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def can_transition_to(self, target: JobStatus) -> bool:
        """Return ``True`` when the transition is allowed."""
        return target in _VALID_TRANSITIONS[self.status]

    def transition_to(self, target: JobStatus) -> Job:
        """Return a copy of ``self`` with ``status`` set to ``target``.

        Raises:
            JobStateError: When the transition is not permitted.

        """
        if not self.can_transition_to(target):
            raise JobStateError(
                f"Job {self.id}: illegal transition {self.status} -> {target}"
            )
        return Job(
            id=self.id,
            kind=self.kind,
            payload=self.payload,
            status=target,
            attempts=self.attempts,
            max_attempts=self.max_attempts,
            last_error=self.last_error,
            next_run_at=self.next_run_at,
            worker_id=self.worker_id,
            tenant_id=self.tenant_id,
            created_at=self.created_at,
            updated_at=datetime.now(UTC),
        )


class QueueSaturatedError(RuntimeError):
    """Raised when a queue refuses new submissions due to back-pressure."""


class PersistentQueue(Protocol):
    """Storage contract for a persistent ingestion queue.

    All methods are async because SQLite access is async under
    :mod:`aiosqlite`. Implementations may add additional methods
    (e.g. ``ack_batch``) but the protocol below is the minimum.
    """

    async def submit(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        tenant_id: str | None = None,
        max_attempts: int = 3,
    ) -> str:
        """Submit a job; return its id."""
        ...

    async def claim(self, worker_id: str, lease_seconds: int = 60) -> Job | None:
        """Claim the next pending job; returns ``None`` when the queue is empty."""
        ...

    async def ack(self, job_id: str) -> None:
        """Mark ``job_id`` as succeeded."""
        ...

    async def nack(self, job_id: str, error: str) -> None:
        """Record ``error``; the job moves back to pending or dead-lettered."""
        ...

    async def dead_letter(self, job_id: str) -> None:
        """Move ``job_id`` to the ``dead`` state."""
        ...

    async def retry(self, job_id: str, delay_seconds: int) -> None:
        """Move ``job_id`` back to ``pending`` with ``next_run_at`` offset."""
        ...

    async def purge(self, status: JobStatus | None = None) -> int:
        """Delete jobs by status; return the count deleted."""
        ...

    async def list(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        ...

    async def stats(self) -> dict[str, int]:
        """Return counts per status."""
        ...


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raghub_queue (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,
    next_run_at TEXT NOT NULL,
    worker_id TEXT,
    tenant_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS raghub_queue_status_next_run
    ON raghub_queue (status, next_run_at);
CREATE INDEX IF NOT EXISTS raghub_queue_tenant_id
    ON raghub_queue (tenant_id);
"""


class SqliteQueue:
    """SQLite-backed :class:`PersistentQueue` implementation."""

    def __init__(self, db_path: str, *, max_inflight: int = 256) -> None:
        """Initialise the queue.

        Args:
            db_path: Path to the SQLite database file.
            max_inflight: Maximum number of pending+running jobs allowed
                before :meth:`submit` raises :class:`QueueSaturatedError`.

        """
        self.db_path = db_path
        self.max_inflight = max_inflight

    async def initialize(self) -> None:
        """Create the ``raghub_queue`` table on first use."""
        async with self.connect() as conn:
            await conn.executescript(SCHEMA_SQL)

    def connect(self) -> AioSqliteConnection:
        import aiosqlite

        return AioSqliteConnection(aiosqlite.connect(self.db_path))

    async def count(self, conn: Any) -> int:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM raghub_queue "
            "WHERE status IN (?, ?)",
            (JobStatus.PENDING.value, JobStatus.RUNNING.value),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def submit(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        tenant_id: str | None = None,
        max_attempts: int = 3,
    ) -> str:
        """Submit a job; return its id."""
        import json

        job_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        async with self.connect() as conn:
            current = await self.count(conn)
            if current >= self.max_inflight:
                raise QueueSaturatedError(
                    f"queue saturated: {current} pending/running jobs"
                )
            await conn.execute(
                "INSERT INTO raghub_queue "
                "(id, kind, payload, status, attempts, max_attempts, "
                " next_run_at, tenant_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    json.dumps(payload),
                    JobStatus.PENDING.value,
                    max_attempts,
                    now,
                    tenant_id,
                    now,
                    now,
                ),
            )
            await conn.commit()
        return job_id

    async def claim(self, worker_id: str, lease_seconds: int = 60) -> Job | None:
        """Claim the next pending job past ``next_run_at``."""
        import aiosqlite

        now = datetime.now(UTC)
        async with self.connect() as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM raghub_queue "
                "WHERE status = ? AND next_run_at <= ? "
                "ORDER BY next_run_at ASC LIMIT 1",
                (JobStatus.PENDING.value, now.isoformat()),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            lease_until = (now + timedelta(seconds=lease_seconds)).isoformat()
            await conn.execute(
                "UPDATE raghub_queue SET status = ?, worker_id = ?, "
                "next_run_at = ?, updated_at = ? WHERE id = ?",
                (
                    JobStatus.RUNNING.value,
                    worker_id,
                    lease_until,
                    now.isoformat(),
                    row["id"],
                ),
            )
            await conn.commit()
            return row_to_job(row)

    async def ack(self, job_id: str) -> None:
        """Mark ``job_id`` as succeeded."""
        now = datetime.now(UTC).isoformat()
        async with self.connect() as conn:
            await conn.execute(
                "UPDATE raghub_queue SET status = ?, updated_at = ? WHERE id = ?",
                (JobStatus.SUCCEEDED.value, now, job_id),
            )
            await conn.commit()

    async def nack(self, job_id: str, error: str) -> None:
        """Record ``error``; transition to ``failed`` or ``dead``."""
        import aiosqlite

        now = datetime.now(UTC)
        async with self.connect() as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM raghub_queue WHERE id = ?", (job_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return
            attempts = int(row["attempts"]) + 1
            max_attempts = int(row["max_attempts"])
            next_status = (
                JobStatus.DEAD.value
                if attempts >= max_attempts
                else JobStatus.PENDING.value
            )
            backoff = 0 if next_status == JobStatus.DEAD.value else int(
                2 ** (attempts - 1)
            )
            await conn.execute(
                "UPDATE raghub_queue SET status = ?, attempts = ?, "
                "last_error = ?, next_run_at = ?, "
                "updated_at = ?, worker_id = NULL WHERE id = ?",
                (
                    next_status,
                    attempts,
                    error,
                    (
                        now
                        if next_status == JobStatus.DEAD.value
                        else (
                            datetime.now(UTC)
                            + timedelta(seconds=backoff)
                        ).isoformat()
                    ),
                    now,
                    job_id,
                ),
            )
            await conn.commit()

    async def dead_letter(self, job_id: str) -> None:
        """Move ``job_id`` to the ``dead`` state."""
        now = datetime.now(UTC).isoformat()
        async with self.connect() as conn:
            await conn.execute(
                "UPDATE raghub_queue SET status = ?, updated_at = ? WHERE id = ?",
                (JobStatus.DEAD.value, now, job_id),
            )
            await conn.commit()

    async def retry(self, job_id: str, delay_seconds: int = 0) -> None:
        """Move ``job_id`` back to ``pending`` with the given delay."""
        next_run_at = (
            datetime.now(UTC) + timedelta(seconds=delay_seconds)
        ).isoformat()
        now = datetime.now(UTC).isoformat()
        async with self.connect() as conn:
            await conn.execute(
                "UPDATE raghub_queue SET status = ?, next_run_at = ?, "
                "worker_id = NULL, updated_at = ? WHERE id = ?",
                (JobStatus.PENDING.value, next_run_at, now, job_id),
            )
            await conn.commit()

    async def purge(self, status: JobStatus | None = None) -> int:
        """Delete jobs by status; return the count deleted."""
        async with self.connect() as conn:
            if status is None:
                cursor = await conn.execute("DELETE FROM raghub_queue")
            else:
                cursor = await conn.execute(
                    "DELETE FROM raghub_queue WHERE status = ?",
                    (status.value,),
                )
            await conn.commit()
            return cursor.rowcount or 0

    async def list(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        import aiosqlite

        async with self.connect() as conn:
            conn.row_factory = aiosqlite.Row
            if status is None:
                cursor = await conn.execute(
                    "SELECT * FROM raghub_queue ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor = await conn.execute(
                    "SELECT * FROM raghub_queue WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (status.value, limit),
                )
            rows = await cursor.fetchall()
            return [row_to_job(row) for row in rows]

    async def stats(self) -> dict[str, int]:
        """Return counts per status."""
        counts: dict[str, int] = {s.value: 0 for s in JobStatus}
        async with self.connect() as conn:
            cursor = await conn.execute(
                "SELECT status, COUNT(*) FROM raghub_queue GROUP BY status"
            )
            rows = await cursor.fetchall()
        for status, count in rows:
            counts[status] = int(count)
        return counts


class AioSqliteConnection:
    """Async context manager wrapping ``aiosqlite.connect``."""

    def __init__(self, coro: Any) -> None:
        self.coro = coro

    async def __aenter__(self) -> Any:
        self.conn = await self.coro
        return self.conn

    async def __aexit__(self, *exc: object) -> None:
        await self.conn.close()


def row_to_job(row: Any) -> Job:
    """Convert a SQLite row to a :class:`Job`."""
    import json

    payload = json.loads(row["payload"]) if row["payload"] else {}
    return Job(
        id=row["id"],
        kind=row["kind"],
        payload=payload,
        status=JobStatus(row["status"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        last_error=row["last_error"],
        next_run_at=datetime.fromisoformat(row["next_run_at"]),
        worker_id=row["worker_id"],
        tenant_id=row["tenant_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


@dataclass(slots=True)
class Worker:
    """Pool of consumer tasks that drain a :class:`PersistentQueue`."""

    queue: PersistentQueue
    handler: Any  # Callable[[Job], Awaitable[Any]]
    concurrency: int = 4
    max_attempts: int = 3
    max_wall_seconds: float = 30.0
    lease_seconds: int = 60
    backoff_base: float = 1.0
    backoff_cap: float = 300.0

    async def run(self) -> None:
        """Run the worker pool until cancelled."""
        await asyncio.gather(*(self.loop(f"worker-{i}") for i in range(self.concurrency)))

    async def loop(self, worker_id: str) -> None:
        while True:
            job = await self.queue.claim(worker_id, lease_seconds=self.lease_seconds)
            if job is None:
                await asyncio.sleep(0.5)
                continue
            try:
                result = await asyncio.wait_for(
                    self.handler(job), timeout=self.max_wall_seconds
                )
            except Exception as exc:
                await self.queue.nack(job.id, repr(exc))
                continue
            await self.queue.ack(job.id)
            _ = result  # handlers decide what to do with the result.

    async def iter_due(self) -> AsyncIterator[Job]:
        """Yield jobs that are due, in lease order, for testing.

        This is a generator intended for test harnesses; production
        code should use :meth:`run` instead.
        """
        while True:
            job = await self.queue.claim("test", lease_seconds=self.lease_seconds)
            if job is None:
                return
            yield job
            await self.queue.ack(job.id)
