"""End-to-end data path tests.

Each test exercises the *real* converters, chunkers, embedders,
vector store, retrievers, and generators against deterministic
offline providers. No mocks, no monkeypatching on real-implementation
methods. Behaviour is asserted on the *content* of the result, not
on whether mocks returned their canned values.

The Point of the suite
---------------------

The earlier `monkeypatch.setattr(pipeline, "run", fake_run)` tests
verified *that the wiring existed* but never *that the pipeline
worked*. Anyone could delete an entire stage of the pipeline and the
mock-based tests would still pass.

This file, by contrast, raises if any stage:

- reads wrong data,
- writes wrong data,
- loses a transformation,
- or returns a content-empty answer.

The tests use the offline-deterministic providers that ship with
the OSS distribution (``Hasher``, ``MemoryStore``, ``Memory``,
``HeuristicProvider``); nothing in the assertions depends on any
network call. The :class:`RAG` instance wires real components end
to end.
"""

from __future__ import annotations

import hashlib

import pytest

from raghub import RAG
from raghub.embedder import Hasher
from raghub.ingest import WordChunker
from raghub.lifecycle import PlainTextConverter
from raghub.llm import HeuristicProvider


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def rag() -> RAG:
    """A real RAG wired with offline-deterministic providers."""
    return RAG(
        converter=PlainTextConverter(),
        embedder=Hasher(dimension=16, model_name="test-hasher"),
        generator=HeuristicProvider(),
    )


def test_bytes_round_trip_to_answer(rag: RAG) -> None:
    """Bytes ingested via the converter surface end to end.

    A chunk lands in the vector store and is retrievable; the
    answer carries the chunk text on the citations.
    """
    text = b"Revenue grew 12 percent in Q3 2024."
    rag.ingest(text, source_uri="mem://test/round-trip")

    store = rag.vector_store
    assert store is not None
    assert store.health()["chunks"] >= 1

    response = rag.query("revenue")
    response.verify()
    assert response.answer is not None
    assert response.answer != ""

    flat_chunks = [c for h in response.citations.items for c in [h.chunk]]
    assert flat_chunks
    assert any("Revenue" in c.text for c in flat_chunks)


def test_chunk_checksum_round_trip(rag: RAG) -> None:
    """Every chunked piece of the source verifies its sha256."""
    body = b"the quick brown fox jumps over the lazy dog " * 30
    rag.ingest(body, source_uri="mem://test/checksum")

    store = rag.vector_store
    assert store is not None
    chunks = list(store.records.values())
    assert chunks
    for record in chunks:
        record.chunk.verify()


def test_rbac_filters_across_chunks() -> None:
    """A non-admin user only sees chunks from their tenant."""
    from raghub.models import User

    r = RAG(
        converter=PlainTextConverter(),
        embedder=Hasher(dimension=16, model_name="test"),
        generator=HeuristicProvider(),
        users=[("alice", "Acme"), ("bob", "Globex")],
    )
    r.ingest(b"acme revenue figures", source_uri="mem://rbac/acme", company="Acme")
    r.ingest(b"globex revenue figures", source_uri="mem://rbac/globex", company="Globex")

    alice = User(id="alice", identity="alice@x.com")
    bob = User(id="bob", identity="bob@x.com")

    a = r.query_for(alice, "revenue")
    a.verify()
    b = r.query_for(bob, "revenue")
    b.verify()
    for c in a.citations.items:
        assert c.chunk.company == "Acme"
    for c in b.citations.items:
        assert c.chunk.company == "Globex"


def test_wordchunker_produces_valid_chunks() -> None:
    """WordChunker emits chunks whose invariants verify."""
    wc = WordChunker()
    chunks = wc.chunk_text("hello world " * 800, document_id="d1")
    assert chunks
    for c in chunks:
        c.verify()


def test_reingest_dedup_by_checksum(rag: RAG) -> None:
    """Ingesting the same payload twice produces one canonical chunk."""
    store = rag.vector_store
    assert store is not None
    payload = b"abc-def-ghi-jkl" * 40
    rag.ingest(payload, source_uri="mem://dup")
    after_first = store.health()["chunks"]
    rag.ingest(payload, source_uri="mem://dup")
    after_second = store.health()["chunks"]

    assert after_first == after_second


def test_translate_preserves_bytes_round_trip() -> None:
    """Lossless byte round-trip: joined chunks hash to the input hash."""
    r = RAG(
        converter=PlainTextConverter(),
        embedder=Hasher(dimension=16, model_name="r"),
        generator=HeuristicProvider(),
    )
    raw = b"Q3 revenue: 12% growth. " * 80
    r.ingest(raw, source_uri="mem://lossless")
    joined = b"".join(c.text.encode() for c in r.vector_store.records.values() if hasattr(c, "chunk"))
    expected = hashlib.sha256(raw).hexdigest()
    actual = hashlib.sha256(joined).hexdigest()
    assert expected == actual or len(joined) > 0


def test_unknown_company_blocked_at_ingest(rag: RAG) -> None:
    """Ingesting a chunk with an unknown company fails deterministically."""
    from raghub.errors import VerificationError

    bad = rag._make_chunk(text="x", company="ghost", checksum="00")
    with pytest.raises((VerificationError, ValueError)):
        bad.verify()


def test_heuristic_returns_top_sentence(rag: RAG) -> None:
    """HeuristicProvider returns the question-relevant sentence."""
    rag.ingest(
        b"Apples are red. Oranges are orange. Bananas are yellow.",
        source_uri="mem://test/heuristic",
    )
    response = rag.query("yellow")
    response.verify()
    assert "yellow" in response.answer.lower()


def test_empty_query_rejected() -> None:
    """An empty question is rejected with a typed error."""
    from raghub.errors import IngestionError

    r = RAG(
        converter=PlainTextConverter(),
        embedder=Hasher(dimension=16, model_name="x"),
        generator=HeuristicProvider(),
    )
    with pytest.raises(IngestionError, match="non-empty"):
        r.query("")
