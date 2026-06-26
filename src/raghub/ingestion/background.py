"""Background document ingestion using a thread pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from typing import Any


class IngestionJob:
    def __init__(self, job_id: str, status: str, result: Any = None) -> None:
        self.job_id = job_id
        self.status = status  # pending, processing, completed, failed
        self.result = result


class BackgroundIngestionService:
    """Queues ingestion jobs for async processing."""

    def __init__(self, max_workers: int = 2) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, IngestionJob] = {}

    def submit(self, fn, *args, **kwargs) -> str:
        job_id = str(uuid4())
        self.jobs[job_id] = IngestionJob(job_id, "pending")
        self.executor.submit(self._run_job, job_id, fn, args, kwargs)
        return job_id

    def _run_job(self, job_id: str, fn, args, kwargs) -> None:
        job = self.jobs[job_id]
        job.status = "processing"
        try:
            result = fn(*args, **kwargs)
            job.status = "completed"
            job.result = result
        except Exception as e:
            job.status = "failed"
            job.result = str(e)

    def get_status(self, job_id: str) -> str | None:
        job = self.jobs.get(job_id)
        return job.status if job else None

    def get_result(self, job_id: str) -> Any:
        job = self.jobs.get(job_id)
        return job.result if job else None
