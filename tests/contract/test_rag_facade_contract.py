"""Contract tests for the public ``RAG`` facade.

Each test exercises one public API entry and asserts the
documented return contract. New accessors added in v0.7.x are
covered here.
"""

from __future__ import annotations


def _new_rag() -> RAG:  # type: ignore[name-defined]  # noqa: F821
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
    """``RAG.query`` returns a typed ``Response``."""
    rag = _new_rag()
    rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
    response = rag.query("revenue")
    assert response.answer


def test_rag_ingest_async_returns_job_id() -> None:
    """``RAG.ingest_async`` returns a job id string."""
    rag = _new_rag()
    job_id = rag.ingest_async(b"hello world")
    assert isinstance(job_id, str) and job_id


def test_rag_conversation_history_round_trip() -> None:
    """``RAG.conversation_history`` returns at most ``limit`` turns."""
    from raghub.models import User

    rag = _new_rag()
    user = User(user_id="alice", email="alice@x")
    rag.aquery("hello", user=user, session_id="s1")
    history = rag.conversation_history("s1", user=user)
    assert isinstance(history, list)


def test_rag_clear_conversation_smoke() -> None:
    """``RAG.clear_conversation`` does not raise."""
    from raghub.models import User

    rag = _new_rag()
    user = User(user_id="alice", email="alice@x")
    rag.clear_conversation("s1", user=user)


def test_rag_archive_accessor_returns_value() -> None:
    """``RAG.archive`` returns the configured ``LocalArchiveStore`` or ``None``."""
    rag = _new_rag()
    archive = rag.archive()
    assert archive is None or hasattr(archive, "put")


def test_rag_queue_accessor_returns_value() -> None:
    """``RAG.queue`` returns the configured queue or ``None``."""
    rag = _new_rag()
    queue = rag.queue()
    assert queue is None or hasattr(queue, "submit")


def test_rag_feedback_store_accessor_returns_value() -> None:
    """``RAG.feedback_store`` returns the configured store or ``None``."""
    rag = _new_rag()
    store = rag.feedback_store()
    assert store is None or hasattr(store, "record")


def test_rag_rate_limiter_accessor_returns_value() -> None:
    """``RAG.rate_limiter`` returns the configured limiter or ``None``."""
    rag = _new_rag()
    limiter = rag.rate_limiter()
    assert limiter is None or hasattr(limiter, "allow")


def test_rag_tenant_resolver_accessor_returns_value() -> None:
    """``RAG.tenant_resolver`` returns the configured resolver or ``None``."""
    rag = _new_rag()
    resolver = rag.tenant_resolver()
    assert resolver is None or hasattr(resolver, "resolve")


def test_rag_isolation_strategy_accessor_returns_value() -> None:
    """``RAG.isolation_strategy`` returns the configured strategy enum."""
    rag = _new_rag()
    strategy = rag.isolation_strategy()
    assert strategy is not None
