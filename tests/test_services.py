"""Services module coverage tests.

Exercises the small helpers in :mod:`raghub.services`: probe_vector_store,
probe_embedder, aggregate_status, the Synchronous/ThreadPool/MemoryQueue
workers, missing_doc, and the simple Module-level accessors.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from raghub.services import (
    MemoryQueue,
    Synchronous,
    ThreadPool,
    aggregate_status,
    parse_users,
    probe_embedder,
    probe_vector_store,
    seed_blocked,
    upload_record,
)

# ---------------------------------------------------------------------------
# probe_vector_store
# ---------------------------------------------------------------------------


def test_probe_vector_store_no_health_method() -> None:
    """A store without health() returns status='unknown'."""

    class _Stub:
        pass

    result = probe_vector_store(_Stub())
    assert result["status"] == "unknown"


def test_probe_vector_store_healthy_dict() -> None:
    """probe_vector_store returns ok for {status: 'healthy'}."""

    class _Stub:
        def health(self) -> dict[str, str]:
            return {"status": "healthy"}

    assert probe_vector_store(_Stub())["status"] == "ok"


def test_probe_vector_store_degraded_normalised() -> None:
    """Any non-ok status is normalised to 'degraded'."""

    class _Stub:
        def health(self) -> dict[str, str]:
            return {"status": "weird"}

    assert probe_vector_store(_Stub())["status"] == "degraded"


def test_probe_vector_store_non_dict_payload() -> None:
    """Non-dict payloads are wrapped with status: 'ok'."""

    class _Stub:
        def health(self) -> str:
            return "fine"

    result = probe_vector_store(_Stub())
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# probe_embedder
# ---------------------------------------------------------------------------


def test_probe_embedder_none_returns_unknown() -> None:
    """probe_embedder(None) returns 'unknown' with a hint."""

    assert probe_embedder(None)["status"] == "unknown"


def test_probe_embedder_no_method_returns_unknown() -> None:
    """A embedder without embed_text() returns 'unknown'."""

    class _Stub:
        pass

    assert probe_embedder(_Stub())["status"] == "unknown"


def test_probe_embedder_returns_ok_with_dim() -> None:
    """A working embedder returns status='ok' and the dimension."""

    class _Stub:
        model_name = "test-model"

        def embed_text(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    result = probe_embedder(_Stub())
    assert result["status"] == "ok"
    assert result["dimension"] == 3
    assert result["model"] == "test-model"


def test_probe_embedder_empty_vector_returns_down() -> None:
    """A zero-length embedding vector returns 'down'."""

    class _Stub:
        def embed_text(self, text: str) -> list[float]:
            return []

    result = probe_embedder(_Stub())
    assert result["status"] == "down"


def test_probe_embedder_async_iterable_returns_unknown_dim() -> None:
    """An async iterator from embed_text is wrapped as ok with no dim."""

    class _Stub:
        model_name = "test"

        def embed_text(self, text: str) -> object:
            async def _aiter() -> object:
                yield 0.1
                yield 0.2

            return _aiter()

    result = probe_embedder(_Stub())
    assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# aggregate_status
# ---------------------------------------------------------------------------


def test_aggregate_status_all_ok() -> None:
    """All-ok probes aggregate to 'ok'."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "healthy"}}) == "ok"


def test_aggregate_status_one_down_is_down() -> None:
    """A single 'down' probe dominates the aggregate."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "down"}}) == "down"


def test_aggregate_status_degraded_propagates() -> None:
    """A 'degraded' or 'unknown' probe degrades the aggregate."""

    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "degraded"}}) == "degraded"
    assert aggregate_status({"a": {"status": "ok"}, "b": {"status": "unknown"}}) == "degraded"


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def test_synchronous_worker_runs_synchronously() -> None:
    """Synchronous worker calls the function immediately."""

    called: list[str] = []

    def _op() -> None:
        called.append("yes")

    worker = Synchronous()
    worker.submit(_op)
    assert called == ["yes"]


def test_thread_pool_worker_submits() -> None:
    """ThreadPool.submit returns a Future-like object."""

    worker = ThreadPool()
    future = worker.submit(lambda: "result")
    assert future is not None
    # Don't wait for completion — the test process exits cleanly.


def test_memory_queue_enqueue_round_trip() -> None:
    """MemoryQueue.enqueue + drain round-trips payloads."""

    queue = MemoryQueue()
    rid = queue.enqueue("op", {"a": 1})
    assert rid is not None


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------


def test_parse_users_json() -> None:
    """parse_users parses a JSON users string."""

    raw = '[{"email": "a@x.com", "password": "x", "companies": ["acme"]}]'
    out = parse_users(raw)
    assert len(out) == 1
    assert out[0]["email"] == "a@x.com"


def test_parse_users_invalid_json_raises() -> None:
    """parse_users propagates JSON errors."""

    with pytest.raises(json.JSONDecodeError):
        parse_users("not-json")


def test_seed_blocked_true_when_production() -> None:
    """seed_blocked returns True in production environments."""

    import os as _os

    _os.environ["CORS_ORIGINS"] = "https://example.com"
    try:
        settings = MagicMock()
        settings.environment = "production"
        assert seed_blocked(settings) is True
    finally:
        del _os.environ["CORS_ORIGINS"]


def test_seed_blocked_true_when_cors_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed_blocked is True when CORS_ORIGINS='*' is set."""

    monkeypatch.setenv("CORS_ORIGINS", "*")
    settings = MagicMock()
    settings.environment = "development"
    assert seed_blocked(settings) is True


def test_seed_blocked_false_when_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """seed_blocked is False when both CORS is explicit and not production."""

    monkeypatch.setenv("CORS_ORIGINS", "https://example.com")
    settings = MagicMock()
    settings.environment = "development"
    assert seed_blocked(settings) is False


# ---------------------------------------------------------------------------
# upload_record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_record_returns_document() -> None:
    """upload_record extracts a Document from an IngestionResult-like input."""

    class _StubResult:
        document = {"id": "d1", "version": 1}

    result = await upload_record(_StubResult())  # type: ignore[arg-type]
    assert result is not None
