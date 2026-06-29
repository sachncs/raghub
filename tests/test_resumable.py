"""Tests for the persistent job store and resumable background service."""

from __future__ import annotations

import time

import pytest

from raghub.ingestion.background import BackgroundIngestionService
from raghub.ingestion.jobs import PersistentJobStore
from raghub.ingestion.resumable import ResumableBackgroundIngestionService


def test_persistent_job_store_upsert_and_get(tmp_path) -> None:
    """Upsert and get round-trip."""
    store = PersistentJobStore(tmp_path / "jobs.db")
    store.upsert("job-1", "pending")
    record = store.get("job-1")
    assert record is not None
    assert record["job_id"] == "job-1"
    assert record["status"] == "pending"
    store.close()


def test_persistent_job_store_returns_none_for_unknown(tmp_path) -> None:
    """An unknown job id returns ``None``."""
    store = PersistentJobStore(tmp_path / "jobs.db")
    assert store.get("nope") is None
    store.close()


def test_persistent_job_store_updates_existing(tmp_path) -> None:
    """Upserting the same id twice updates the record."""
    store = PersistentJobStore(tmp_path / "jobs.db")
    store.upsert("job-1", "pending")
    store.upsert("job-1", "completed", result="ok")
    record = store.get("job-1")
    assert record["status"] == "completed"
    assert record["result"] == "ok"
    store.close()


def test_persistent_job_store_yields_all(tmp_path) -> None:
    """``all()`` yields every record."""
    store = PersistentJobStore(tmp_path / "jobs.db")
    store.upsert("a", "pending")
    store.upsert("b", "pending")
    ids = {record["job_id"] for record in store.all()}
    assert ids == {"a", "b"}
    store.close()


def test_resumable_service_submits_job(tmp_path) -> None:
    """``submit`` returns a job id; the job runs to completion."""
    service = ResumableBackgroundIngestionService(
        db_path=tmp_path / "jobs.db", max_workers=1
    )
    try:

        def fn() -> str:
            return "ok"

        job_id = service.submit(fn)
        # Wait for the job to complete.
        for _ in range(50):
            if service.get_status(job_id) == "completed":
                break
            time.sleep(0.05)
        assert service.get_status(job_id) == "completed"
        assert service.get_result(job_id) == "ok"
    finally:
        service.shutdown()


def test_resumable_service_persists_status(tmp_path) -> None:
    """The service writes the final status to the SQLite ledger."""

    db_path = tmp_path / "jobs.db"
    service = ResumableBackgroundIngestionService(db_path=db_path, max_workers=1)
    try:

        def fn() -> str:
            return "ok"

        job_id = service.submit(fn)
        for _ in range(50):
            if service.get_status(job_id) == "completed":
                break
            time.sleep(0.05)
    finally:
        service.shutdown()

    # Reopen the store and confirm the job state was persisted.
    store = PersistentJobStore(db_path)
    record = store.get(job_id)
    assert record is not None
    assert record["status"] == "completed"
    assert record["result"] == "ok"
    store.close()


def test_resumable_service_restores_jobs(tmp_path) -> None:
    """A fresh ``ResumableBackgroundIngestionService`` reloads prior job state."""

    db_path = tmp_path / "jobs.db"
    service = ResumableBackgroundIngestionService(db_path=db_path, max_workers=1)
    try:
        service.submit(lambda: "ok")
    finally:
        service.shutdown()

    service2 = ResumableBackgroundIngestionService(db_path=db_path, max_workers=1)
    try:
        assert service2.jobs  # at least one job restored
    finally:
        service2.shutdown()


def test_resumable_subclass_does_not_break_base(tmp_path) -> None:
    """``ResumableBackgroundIngestionService`` is a ``BackgroundIngestionService``."""

    service = ResumableBackgroundIngestionService(db_path=tmp_path / "jobs.db")
    try:
        assert isinstance(service, BackgroundIngestionService)
    finally:
        service.shutdown()
