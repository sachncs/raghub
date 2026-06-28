"""Background document ingestion using a thread pool.

The :class:`BackgroundIngestionService` wraps a
:class:`concurrent.futures.ThreadPoolExecutor` to provide fire-and-forget
ingestion with status tracking. Each call to :meth:`submit` returns a job
id; the caller can poll :meth:`get_status` and :meth:`get_result` later.

Design notes:

* **Per-call event loop.** When the wrapped callable returns a coroutine,
  ``run_job`` allocates a fresh :func:`asyncio.new_event_loop` and runs
  the coroutine on it. This is necessary because a worker thread does
  not have an inherited asyncio loop binding; reusing the calling
  thread's loop would deadlock. The loop is closed after the coroutine
  finishes so resources are released promptly.
* **In-memory job map.** ``jobs`` is a process-local dict. Restarting
  the process loses all pending jobs; for persistence use a queue like
  Celery or RQ (see :class:`raghub.services.workers.InMemoryTaskQueue`).
* **No exception propagation.** Failures inside ``run_job`` are stored
  on ``job.result`` as a string and ``job.status`` becomes ``"failed"``.
  The calling thread never sees the exception; check the job before
  trusting its output.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4


class IngestionJob:
    """Lightweight value object tracking a single ingestion task.

    Attributes:
        job_id: Stable identifier returned by :meth:`submit`.
        status: One of ``"pending"``, ``"processing"``, ``"completed"``,
            ``"failed"``. Free-form strings rather than an enum because
            the set is small and stable.
        result: The callable's return value on success, the stringified
            exception on failure, or ``None`` while pending.
    """

    def __init__(self, job_id: str, status: str, result: Any = None) -> None:
        """Initialise the job record.

        Args:
            job_id: Unique identifier.
            status: Initial status (``"pending"`` is conventional).
            result: Optional initial result (defaults to ``None``).
        """
        self.job_id = job_id
        self.status = status
        self.result = result


class BackgroundIngestionService:
    """Queues ingestion jobs for async processing.

    A thin wrapper around :class:`ThreadPoolExecutor` that adds job
    tracking. Construct once and reuse; constructing per-call does **not**
    reuse the underlying executor.

    Attributes:
        executor: Backing thread pool.
        jobs: Map from job id to :class:`IngestionJob`.
    """

    def __init__(self, max_workers: int = 2) -> None:
        """Initialise the service with a thread pool.

        Args:
            max_workers: Maximum concurrent ingestion jobs. Default 2
                is conservative; raise for higher throughput, but be
                aware that each worker may pin a large chunk embedding
                call.
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, IngestionJob] = {}

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit a callable for background execution.

        Args:
            fn: The callable to run. May return either a value or a
                coroutine; coroutines are awaited on a fresh event loop.
            *args: Positional arguments forwarded to ``fn``.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            A job id that can be passed to :meth:`get_status` and
            :meth:`get_result`.
        """
        job_id = str(uuid4())
        # Register the job *before* submitting so ``run_job`` can find
        # it when the worker thread starts. ``"pending"`` is the
        # initial state until the worker picks the job up.
        self.jobs[job_id] = IngestionJob(job_id, "pending")
        self.executor.submit(self.run_job, job_id, fn, args, kwargs)
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute one queued job, including asyncio unwrapping.

        Args:
            job_id: The job id; looked up in :pyattr:`jobs`.
            fn: The callable to invoke.
            args: Positional args tuple.
            kwargs: Keyword args dict.
        """
        job = self.jobs[job_id]
        job.status = "processing"
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                # Worker threads do not inherit an asyncio loop, so we
                # spin up a fresh one, install it as the thread-local
                # loop, run the coroutine, and close it on the way out.
                # Reusing the calling thread's loop here would either
                # raise or hang because the worker is not that thread.
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
            job.status = "completed"
            job.result = result
        except Exception as e:
            # Swallow the exception; the failure is communicated through
            # the job's status and stringified result. Callers must
            # ``get_status`` before trusting ``get_result``.
            job.status = "failed"
            job.result = str(e)

    def get_status(self, job_id: str) -> str | None:
        """Return the current status for ``job_id``, or ``None`` if unknown.

        Args:
            job_id: The id returned by :meth:`submit`.

        Returns:
            The job's status string, or ``None`` if no such job exists.
        """
        job = self.jobs.get(job_id)
        return job.status if job else None

    def get_result(self, job_id: str) -> Any:
        """Return the stored result for ``job_id``, or ``None`` if unknown.

        Args:
            job_id: The id returned by :meth:`submit`.

        Returns:
            The callable's return value on success, the stringified
            exception on failure, or ``None`` if the job is unknown.
        """
        job = self.jobs.get(job_id)
        return job.result if job else None