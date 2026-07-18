"""Tests for background ingestion service."""

from __future__ import annotations

import asyncio
import time

from raghub.ingestion.background import BackgroundIngestionService


def test_submit_sync_function() -> None:
    svc = BackgroundIngestionService(max_workers=1)

    def sync_fn(x: int) -> int:
        return x * 2

    job_id = svc.submit(sync_fn, 21)
    time.sleep(0.2)
    assert svc.get_status(job_id) == "completed"
    assert svc.get_result(job_id) == 42


def test_submit_async_function() -> None:
    svc = BackgroundIngestionService(max_workers=1)

    async def async_fn(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 3

    job_id = svc.submit(async_fn, 14)
    time.sleep(0.5)
    assert svc.get_status(job_id) == "completed"
    assert svc.get_result(job_id) == 42


def test_submit_async_function_failure() -> None:
    svc = BackgroundIngestionService(max_workers=1)

    async def failing_fn() -> None:
        await asyncio.sleep(0.01)
        msg = "simulated failure"
        raise ValueError(msg)

    job_id = svc.submit(failing_fn)
    time.sleep(0.5)
    assert svc.get_status(job_id) == "failed"
    assert "simulated failure" in str(svc.get_result(job_id))


def test_submit_multiple_jobs() -> None:
    svc = BackgroundIngestionService(max_workers=2)
    results: dict[str, int] = {}

    def slow_sync(n: int) -> int:
        time.sleep(0.1)
        return n

    ids = [svc.submit(slow_sync, i) for i in range(5)]
    time.sleep(1.0)
    for i, job_id in enumerate(ids):
        assert svc.get_status(job_id) == "completed", f"Job {i} failed"
        results[job_id] = svc.get_result(job_id)
    assert len(results) == 5


def test_unknown_job_returns_none() -> None:
    svc = BackgroundIngestionService()
    assert svc.get_status("nonexistent") is None
    assert svc.get_result("nonexistent") is None
