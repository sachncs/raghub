"""Tests for the InMemoryVectorStore."""

from __future__ import annotations

from raghub.errors import VectorStoreError
from raghub.stores import MemoryStore


def test_insert_returns_row_count(sample_chunks, sample_vectors):
    store = MemoryStore(embedding_dim=384)
    written = store.insert(sample_chunks, sample_vectors)
    assert written == 1


def test_upsert_delegates_to_insert(sample_chunks, sample_vectors):
    store = MemoryStore(embedding_dim=384)
    written = store.upsert(sample_chunks, sample_vectors)
    assert written == 1


def test_dimension_mismatch_raises(sample_chunks):
    bad_vectors = [[0.0] * 100]
    store = MemoryStore(embedding_dim=384)
    try:
        store.insert(sample_chunks, bad_vectors)
    except VectorStoreError as exc:
        assert "dimension mismatch" in str(exc)
        assert "384" in str(exc)
        assert "100" in str(exc)
    else:
        raise AssertionError("expected VectorStoreError")


def test_search_returns_top_k(sample_chunks, sample_vectors):
    store = MemoryStore(embedding_dim=384)
    store.insert(sample_chunks, sample_vectors)
    hits = store.search(vector=sample_vectors[0], top_k=5)
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == sample_chunks[0].id


def test_delete_removes_chunk(sample_chunks, sample_vectors):
    store = MemoryStore(embedding_dim=384)
    store.insert(sample_chunks, sample_vectors)
    store.delete([sample_chunks[0].id])
    hits = store.search(vector=sample_vectors[0], top_k=5)
    assert hits == []


def test_health_returns_status():
    store = MemoryStore(embedding_dim=384)
    h = store.health()
    assert h["status"] == "ok"
