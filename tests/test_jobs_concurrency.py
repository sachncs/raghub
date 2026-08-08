"""Concurrency stress tests for :class:`raghub.jobs.SqliteQueue`.

Exercises the SQLite-backed queue against 8 concurrent workers draining
100 jobs, asserting that every job reaches a terminal state within the
deadlock budget and that no job is processed twice.

The workers run as real ``asyncio`` tasks backed by ``aiosqlite``, so a
hang or a lock-up in either layer will surface as a test timeout rather
than being papered over by mocks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from raghub.jobs import Job, JobStatus, SqliteQueue, Worker

_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {JobStatus.Succeeded.value, JobStatus.Failed.value, JobStatus.Dead.value}
)


async def _wait_for_terminal(
    queue: SqliteQueue,
    expected_count: int,
    poll_seconds: float = 0.05,
) -> None:
    """Block until ``expected_count`` jobs reach a terminal state.

    Polls :meth:`SqliteQueue.stats` until both ``pending`` and
    ``running`` drop to zero and the terminal-state total equals
    ``expected_count``. Returns when the queue is fully drained.
    """
    while True:
        counts = await queue.stats()
        in_flight = counts[JobStatus.Pending.value] + counts[JobStatus.Running.value]
        terminal_total = sum(counts[status] for status in _TERMINAL_STATUSES)
        if in_flight == 0 and terminal_total == expected_count:
            return
        await asyncio.sleep(poll_seconds)


class TestSqliteQueueConcurrency:
    """End-to-end concurrency stress for ``SqliteQueue`` + ``Worker``."""

    async def test_8_workers_100_jobs_no_deadlock(self, tmp_path: Path) -> None:
        """8 workers drain 100 jobs within 60 s with no double-processing.

        Submits 100 jobs, runs 8 concurrent workers, and asserts:
          * every job reaches a terminal state (SUCCEEDED, FAILED, or DEAD);
          * the handler is invoked exactly once per job;
          * the test completes within 60 s (deadlock guard).
        """
        expected_count = 100
        worker_count = 8

        db_path = tmp_path / "queue.db"
        queue = SqliteQueue(db_path=str(db_path), max_inflight=expected_count)
        await queue.initialize()

        for index in range(expected_count):
            await queue.submit(
                kind="stress",
                payload={"index": index},
                max_attempts=1,
            )

        processed: list[int] = []

        async def run_ingest_job(job: Job) -> None:
            processed.append(int(job.payload["index"]))

        workers = [
            Worker(queue=queue, dispatcher=run_ingest_job, concurrency=1)
            for _ in range(worker_count)
        ]
        tasks = [
            asyncio.create_task(worker.loop(f"stress-w{i}"))
            for i, worker in enumerate(workers)
        ]

        try:
            await asyncio.wait_for(
                _wait_for_terminal(queue, expected_count=expected_count),
                timeout=60,
            )
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        final_counts = await queue.stats()
        assert final_counts[JobStatus.Pending.value] == 0
        assert final_counts[JobStatus.Running.value] == 0
        terminal_total = sum(final_counts[status] for status in _TERMINAL_STATUSES)
        assert terminal_total == expected_count

        assert len(processed) == expected_count
        assert sorted(processed) == list(range(expected_count))