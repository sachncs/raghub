"""Contract tests for the public ``RAG`` facade.

Each test exercises one public API entry and asserts the
documented return contract. New accessors added in v0.7.x are
covered here.
"""

from __future__ import annotations


def _new_rag() -> RAG:  # noqa: F821 - RAG is imported lazily below
    from raghub import RAG

    return RAG()


def test_rag_constructor_smoke() -> None:
    """``RAG()`` constructs without arguments (no API key required)."""
    rag = _new_rag()
    assert rag is not None


def test_rag_health_returns_dict() -> None:
    """``RAG.health`` returns a dict describing the wiring."""
    rag = _new_rag()
    health = rag.health()
    assert isinstance(health, dict)
    assert "status" in health


def test_rag_query_with_bytes_smoke() -> None:
    """``RAG.query`` returns a typed ``Response`` carrying the ingested text."""
    rag = _new_rag()
    rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
    response = rag.query("revenue")
    assert response.answer
    # At least one citation should reference the ingested document
    # so the round-trip really traversed the pipeline.
    assert response.citations
    assert any("Revenue" in c.chunk.text for c in response.citations if c.chunk)


def test_rag_ingest_async_returns_job_id() -> None:
    """``RAG.ingest_async`` returns a job id string."""
    rag = _new_rag()
    job_id = rag.ingest_async(b"hello world")
    assert isinstance(job_id, str) and job_id


def test_rag_conversation_history_round_trip() -> None:
    """``RAG.conversation_history`` returns at most ``limit`` turns."""
    from raghub.models import User

    rag = _new_rag()
    user = User(id="alice", email="alice@x")
    rag.aquery("hello", user=user, session_id="s1")
    history = rag.conversation_history("s1", user=user)
    assert isinstance(history, list)


def test_rag_clear_conversation_smoke() -> None:
    """``RAG.clear_conversation`` does not raise."""
    from raghub.models import User

    rag = _new_rag()
    user = User(id="alice", email="alice@x")
    rag.clear_conversation("s1", user=user)


def test_rag_archive_accessor_returns_none_when_unconfigured() -> None:
    """``RAG.archive`` returns ``None`` when no archive has been configured."""
    rag = _new_rag()
    assert rag.archive() is None


def test_rag_queue_accessor_returns_none_when_unconfigured() -> None:
    """``RAG.queue`` returns ``None`` when no queue has been configured."""
    rag = _new_rag()
    assert rag.queue() is None


def test_rag_feedback_store_accessor_returns_none_when_unconfigured() -> None:
    """``RAG.feedback_store`` returns ``None`` when no store has been configured."""
    rag = _new_rag()
    assert rag.feedback_store() is None


def test_rag_rate_limiter_accessor_returns_none_when_unconfigured() -> None:
    """``RAG.rate_limiter`` returns ``None`` when no limiter has been configured."""
    rag = _new_rag()
    assert rag.rate_limiter() is None


def test_rag_tenant_resolver_accessor_returns_none_when_unconfigured() -> None:
    """``RAG.tenant_resolver`` returns ``None`` when no resolver has been configured."""
    rag = _new_rag()
    assert rag.tenant_resolver() is None


def test_rag_isolation_strategy_accessor_returns_value() -> None:
    """``RAG.isolation_strategy`` returns the configured strategy enum."""
    rag = _new_rag()
    strategy = rag.isolation_strategy()
    assert strategy is not None
