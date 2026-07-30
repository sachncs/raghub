"""Tests for the SqliteVectorStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.errors import VectorStoreError
from raghub.store import SqliteStore


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SqliteStore:
    path = tmp_path / "vecstore.db"
    store = SqliteStore(path=str(path), embedding_dim=384)
    yield store
    store.conn.close()


def test_insert_returns_row_count(sqlite_store, sample_chunks, sample_vectors):
    written = sqlite_store.insert(sample_chunks, sample_vectors)
    assert written == 1


def test_dimension_mismatch_raises(sqlite_store, sample_chunks):
    bad_vectors = [[0.0] * 100]
    with pytest.raises(VectorStoreError, match="dimension mismatch"):
        sqlite_store.insert(sample_chunks, bad_vectors)


def test_search_returns_top_k(sqlite_store, sample_chunks, sample_vectors):
    sqlite_store.insert(sample_chunks, sample_vectors)
    hits = sqlite_store.search(vector=sample_vectors[0], top_k=5)
    assert len(hits) >= 1


def test_pragma_foreign_keys_enabled(sqlite_store):
    row = sqlite_store.conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_health_returns_ok(sqlite_store):
    h = sqlite_store.health()
    assert h["status"] == "ok"
    assert "chunks" in h