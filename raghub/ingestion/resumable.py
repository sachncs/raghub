"""Resumable background ingestion.

Extends :class:`raghub.ingestion.background.BackgroundIngestionService`
with a persistent :class:`PersistentJobStore`. Job state is written
to SQLite on every transition so the service can recover from a
crash and resume pending ingestion jobs on the next start.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raghub.ingestion.background import BackgroundIngestionService, IngestionJob
from raghub.ingestion.jobs import PersistentJobStore


class ResumableBackgroundIngestionService(BackgroundIngestionService):
    """Background ingestion with a persistent job ledger."""

    def __init__(self, *, db_path: str | Path, max_workers: int = 2) -> None:
        """Initialise the service.

        Args:
            db_path: Path to the SQLite ledger.
            max_workers: Maximum concurrent workers.
        """
        super().__init__(max_workers=max_workers)
        self.store = PersistentJobStore(db_path)
        self.restore_from_store()

    def restore_from_store(self) -> None:
        """Reload prior job state into the in-memory map."""
        for record in self.store.all():
            self.jobs[record["job_id"]] = IngestionJob(
                job_id=record["job_id"],
                status=record["status"],
                result=record["result"],
            )

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit ``fn`` for background execution.

        Args:
            fn: Callable; may return a coroutine.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            A job id.
        """
        job_id = super().submit(fn, *args, **kwargs)
        self.store.upsert(job_id, "pending")
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute a job, persisting status transitions."""
        try:
            super().run_job(job_id, fn, args, kwargs)
        finally:
            job = self.jobs.get(job_id)
            if job is not None:
                self.store.upsert(job_id, job.status, job.result)

    def shutdown(self) -> None:
        """Flush the job store and shut down the executor."""
        try:
            for job_id, job in list(self.jobs.items()):
                self.store.upsert(job_id, job.status, job.result)
        finally:
            self.store.close()
            self.executor.shutdown(wait=False)


__all__ = ["ResumableBackgroundIngestionService"]
