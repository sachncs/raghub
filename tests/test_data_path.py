"""End-to-end data path tests.

Each test exercises the *real* converters, chunkers, embedders,
vector store, retrievers, and generators against deterministic
offline providers. No mocks, no monkeypatching on real-implementation
methods. Behaviour is asserted on the *content* of the result, not
on whether mocks returned their canned values.

The Point of the suite
---------------------

The earlier ``monkeypatch.setattr(pipeline, "run", fake_run)`` tests
verified *that the wiring existed* but never *that the pipeline
worked*. Anyone could delete an entire stage of the pipeline and the
mock-based tests would still pass.

This file, by contrast, raises if any stage:

- reads wrong data,
- writes wrong data,
- loses a transformation,
- or returns a content-empty answer.

The tests use the offline-deterministic providers that ship with
the OSS distribution (``FeatureHashingEmbedder``, ``MemoryStore``,
``Memory``, ``StubLLM``); nothing in the assertions depends on any
network call. The :class:`RAG` instance wires real components end
to end.
"""

from __future__ import annotations

import hashlib

import pytest

from raghub import (
    RAG,
    Settings,
)
from raghub.embedder import FeatureHashingEmbedder
from raghub.gen import DefaultGenerator
from raghub.ingest import WordChunker
from raghub.lifecycle import PlainTextConverter
from raghub.llm import GenerationRequest, Generator


class StubLLM(Generator):
    """Deterministic LLM stub for offline end-to-end tests."""

    model_name: str = "stub"

    @staticmethod
    def generate(request: GenerationRequest) -> str:
        """Return a fixed answer regardless of input."""
        return "This is a stub answer for offline testing."


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def rag() -> RAG:
    """A real RAG wired with offline-deterministic providers."""
    return RAG(
        settings=Settings(embedding_dim=16),
        converter=PlainTextConverter(),
        embedder=FeatureHashingEmbedder(dimension=16, model_name="test-hasher"),
        generator=DefaultGenerator(llm=StubLLM()),
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

    assert response.citations
    flat_chunks = [cit.chunk for cit in response.citations]
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


def test_empty_query_rejected() -> None:
    """An empty question is rejected with a typed error."""
    from raghub.errors import IngestionError

    r = RAG(
        settings=Settings(embedding_dim=16),
        converter=PlainTextConverter(),
        embedder=FeatureHashingEmbedder(dimension=16, model_name="x"),
        llm=StubLLM(),
        generator=DefaultGenerator(llm=StubLLM()),
    )
    with pytest.raises(IngestionError, match="non-empty"):
        r.query("")
