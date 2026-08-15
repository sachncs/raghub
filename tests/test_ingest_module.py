"""Ingest module coverage tests.

Exercises the small helpers in :mod:`raghub.ingest`: build_refinery,
apply_refinery, record_from_pipeline, Job, Batch, Jobs, Resumable,
and the IngestionResult / Ingestor class skeletons.
"""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import Any
from unittest.mock import MagicMock

import pytest

from raghub.ingest import (
    Batch,
    IngestionResult,
    Job,
    Jobs,
    Resumable,
    apply_refinery,
    build_refinery,
    record_from_pipeline,
)
from raghub.models import Classification, DocumentLifecycleStatus

# ---------------------------------------------------------------------------
# build_refinery / apply_refinery
# ---------------------------------------------------------------------------


def test_build_refinery_returns_none_when_chonkie_unavailable() -> None:
    """build_refinery returns None when chonkie is not installed."""

    # Patch the module-level constant to simulate the missing dep.
    from raghub.ingest import chunker

    saved = chunker.CHONKIE_MODULE
    chunker.CHONKIE_MODULE = None
    try:
        assert build_refinery() is None
    finally:
        chunker.CHONKIE_MODULE = saved


def test_apply_refinery_noop_when_none() -> None:
    """apply_refinery returns the original list when refinery is None."""

    pieces = ["a", "b"]
    assert apply_refinery(pieces, None) == pieces


def test_apply_refinery_noop_when_empty() -> None:
    """apply_refinery returns [] when pieces is empty."""

    assert apply_refinery([], MagicMock()) == []


# ---------------------------------------------------------------------------
# Job / Batch
# ---------------------------------------------------------------------------


def test_job_init_attributes() -> None:
    """Job stores job_id / status / result."""

    job = Job("j1", "pending")
    assert job.job_id == "j1"
    assert job.status == "pending"
    assert job.result is None


def test_job_init_with_result() -> None:
    """Job accepts an explicit result."""

    job = Job("j1", "completed", {"ok": True})
    assert job.result == {"ok": True}


def test_batch_submit_returns_a_job_id() -> None:
    """Batch.submit returns a job id and stores the Job in batch.jobs."""

    batch = Batch(max_workers=1)

    def _op() -> None:
        return None

    job_id = batch.submit(_op)
    assert job_id is not None
    assert job_id in batch.jobs
    assert batch.jobs[job_id].job_id == job_id
    batch.shutdown(wait=False)


def test_batch_submit_raises_when_shut_down() -> None:
    """submit() after shutdown raises RuntimeError."""

    batch = Batch(max_workers=1)
    batch.shutdown(wait=False)
    with pytest.raises(RuntimeError, match="shut down"):
        batch.submit(lambda: None)


def test_batch_run_job_marks_completed() -> None:
    """run_job transitions a synchronous job to 'completed'."""

    batch = Batch(max_workers=1)

    def _op() -> int:
        return 42

    job_id = batch.submit(_op)
    # Wait for the worker to finish so the assertion sees 'completed'.
    asyncio.run(_wait_for_status(batch, job_id, "completed"))
    assert batch.jobs[job_id].result == 42
    batch.shutdown(wait=False)


def test_batch_run_job_marks_failed_on_exception() -> None:
    """run_job records 'failed' and the exception when the fn raises."""

    batch = Batch(max_workers=1)

    def _op() -> None:
        raise ValueError("boom")

    job_id = batch.submit(_op)
    asyncio.run(_wait_for_status(batch, job_id, "failed"))
    assert "boom" in batch.jobs[job_id].result
    batch.shutdown(wait=False)


def test_batch_run_job_unwraps_async_coroutine() -> None:
    """run_job awaits an asyncio coroutine and stores the result."""

    batch = Batch(max_workers=1)

    async def _op() -> int:
        await asyncio.sleep(0.01)
        return 99

    job_id = batch.submit(_op)
    asyncio.run(_wait_for_status(batch, job_id, "completed"))
    assert batch.jobs[job_id].result == 99
    batch.shutdown(wait=False)


async def _wait_for_status(batch: Batch, job_id: str, status: str, timeout: float = 2.0) -> None:
    """Poll batch.jobs[job_id].status until it equals ``status`` or timeout."""

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if batch.jobs[job_id].status == status:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached status {status!r}")


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


def test_job_store_upsert_and_get(tmp_path: Any) -> None:
    """Jobs.upsert + get round-trip."""

    store = Jobs(tmp_path / "jobs.db")
    store.upsert("j1", "completed", {"ok": True})
    loaded = store.get("j1")
    assert loaded is not None
    assert loaded["status"] == "completed"
    store.close()


def test_job_store_get_unknown_returns_none(tmp_path: Any) -> None:
    """Jobs.get returns None for unknown ids."""

    store = Jobs(tmp_path / "jobs.db")
    assert store.get("missing") is None
    store.close()


def test_job_store_all_jobs_yields_each(tmp_path: Any) -> None:
    """Jobs.all_jobs yields every persisted job."""

    store = Jobs(tmp_path / "jobs.db")
    store.upsert("j1", "completed", None)
    store.upsert("j2", "pending", None)
    jobs = list(store.all_jobs())
    assert {j["job_id"] for j in jobs} == {"j1", "j2"}
    store.close()


# ---------------------------------------------------------------------------
# Resumable
# ---------------------------------------------------------------------------


def test_resumable_init_with_db_path(tmp_path: Any) -> None:
    """Resumable uses its db_path to open a Jobs."""

    r = Resumable(db_path=tmp_path / "jobs.db", max_workers=1)
    assert r.executor is not None
    assert r.store.db_path == tmp_path / "jobs.db"


# ---------------------------------------------------------------------------
# record_from_pipeline
# ---------------------------------------------------------------------------


def test_record_from_pipeline_projects_chunks() -> None:
    """record_from_pipeline produces a Document with chunk ids."""

    from raghub.models import Chunk, Pipeline, PipelineOutputs

    user = MagicMock()
    user.email = "alice@x.com"
    chunks = [
        Chunk(
            id="c1",
            text="first",
            checksum=sha256(b"first").hexdigest(),
            document_id="d1",
            version=1,
            company="acme",
            owner="alice@example.com",
        ),
        Chunk(
            id="c2",
            text="second",
            checksum=sha256(b"second").hexdigest(),
            document_id="d1",
            version=1,
            company="acme",
            owner="alice@example.com",
        ),
    ]
    pipeline = Pipeline(
        outputs=PipelineOutputs(extra={"chunks": chunks, "version": 1, "document_id": "d1"})
    )
    document = record_from_pipeline(
        pipeline,
        file_name="doc.txt",
        mime_type="text/plain",
        owner=user,
        organization="acme",
        classification=Classification.Internal,
        checksum="h",
        tags=["t1"],
    )
    assert document.id == "d1"
    assert document.chunks == ["c1", "c2"]
    assert document.chunk_count == 2
    assert document.owner == "alice@x.com"
    assert document.organization == "acme"
    assert document.classification == Classification.Internal
    assert document.status == DocumentLifecycleStatus.Ready
    assert document.mime_type == "text/plain"


def test_record_from_pipeline_no_chunks() -> None:
    """record_from_pipeline handles an empty chunk list."""

    user = MagicMock()
    user.email = "alice@x.com"
    pipeline = MagicMock()
    pipeline.outputs = {"id": "d2", "chunks": []}
    document = record_from_pipeline(
        pipeline,
        file_name="x.txt",
        mime_type="text/plain",
        owner=user,
        organization="co",
        classification=Classification.Internal,
        checksum="h",
        tags=None,
    )
    assert document.chunks == []
    assert document.chunk_count == 0
    assert document.tags == []


# ---------------------------------------------------------------------------
# IngestionResult
# ---------------------------------------------------------------------------


def test_ingestion_result_attributes() -> None:
    """IngestionResult carries a Document + chunk-id list."""

    from datetime import UTC, datetime

    from raghub.models import Document

    document = Document(
        id="d1",
        version=1,
        owner="alice@example.com",
        organization="acme",
        checksum="h",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    result = IngestionResult(document=document, chunks=["c1"])
    assert result.document.id == "d1"
    assert result.chunks == ["c1"]


def test_ingestion_result_default_chunks() -> None:
    """IngestionResult has an empty chunks list by default."""

    result = IngestionResult(document=None)  # type: ignore[arg-type]
    assert result.chunks == []  # type: ignore[attr-defined]
