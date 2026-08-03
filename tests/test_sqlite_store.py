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


def test_insert_or_ignore_skips_duplicates(sqlite_store, sample_chunks, sample_vectors):
    """Re-inserting the same chunk_id is a no-op (INSERT OR IGNORE).

    Returns ``0`` rows written for the duplicate, leaving the original
    row untouched.
    """
    written1 = sqlite_store.insert(sample_chunks, sample_vectors)
    assert written1 == 1
    dupe = sample_chunks[0].model_copy(update={"text": "different text"})
    written2 = sqlite_store.insert([dupe], sample_vectors)
    assert written2 == 0
    row = sqlite_store.conn.execute(
        "SELECT text FROM raghub WHERE chunk_id = ?", (sample_chunks[0].id,)
    ).fetchone()
    assert row[0] == sample_chunks[0].text


def test_concurrent_inserts_are_safe(sqlite_store, sample_chunks):
    """Two threads inserting distinct chunks must not corrupt the index.

    Uses a :class:`threading.Barrier` to maximise the chance of interleaving.
    """
    import threading

    barrier = threading.Barrier(2)
    results: list[int] = []

    def insert_chunks(offset: int) -> None:
        chunks = [
            sample_chunks[0].model_copy(update={"id": f"thread-{offset}-{i}"}) for i in range(5)
        ]
        vectors = [[0.01 * (offset + i)] * 384 for i in range(5)]
        barrier.wait()
        results.append(sqlite_store.insert(chunks, vectors))

    t1 = threading.Thread(target=insert_chunks, args=(0,))
    t2 = threading.Thread(target=insert_chunks, args=(10,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == [5, 5]
    count = sqlite_store.conn.execute("SELECT COUNT(*) FROM raghub").fetchone()[0]
    assert count == 10


# ---------------------------------------------------------------------------
# v0.9.0 Tier 2 — Item 11: SqliteStore tenant_id column filter
# ---------------------------------------------------------------------------


def test_sqlite_store_search_filters_by_tenant_id(tmp_path):
    """Item 11: SqliteStore.search adds ``company = ?`` when tenant_id is bound."""
    from datetime import UTC, datetime
    from raghub.config import Settings, TenantsConfig
    from raghub.config import Settings as _S
    from raghub.models import Chunk, Classification
    from raghub.store import SqliteStore
    from raghub.tenants import TenantContext, set_current_tenant, reset_current_tenant

    db = tmp_path / "vecstore.db"
    store = SqliteStore(path=str(db), embedding_dim=2)

    def _chunk(cid, *, company):
        return Chunk(
            id=cid,
            document_id=f"doc-{cid}",
            version=1,
            company=company,
            owner="alice@x",
            classification=Classification.INTERNAL,
            checksum=cid,
            text=f"text-{cid}",
            created_at=datetime.now(UTC),
        )

    store.insert(
        [_chunk("a", company="acme"), _chunk("b", company="beta")],
        [[0.1, 0.2], [0.1, 0.2]],
    )

    token = set_current_tenant(TenantContext(tenant_id="acme"))
    try:
        hits = store.search(vector=[0.1, 0.2], top_k=10)
    finally:
        reset_current_tenant(token)
    assert {h["chunk_id"] for h in hits} == {"a"}

    token = set_current_tenant(TenantContext(tenant_id="beta"))
    try:
        hits = store.search(vector=[0.1, 0.2], top_k=10)
    finally:
        reset_current_tenant(token)
    assert {h["chunk_id"] for h in hits} == {"b"}


def test_sqlite_store_search_explicit_tenant_overrides_context(tmp_path):
    """Explicit tenant_id wins over the bound tenant context."""
    from datetime import UTC, datetime
    from raghub.models import Chunk, Classification
    from raghub.store import SqliteStore
    from raghub.tenants import TenantContext, set_current_tenant, reset_current_tenant

    db = tmp_path / "vecstore.db"
    store = SqliteStore(path=str(db), embedding_dim=2)

    def _chunk(cid, *, company):
        return Chunk(
            id=cid,
            document_id=f"doc-{cid}",
            version=1,
            company=company,
            owner="alice@x",
            classification=Classification.INTERNAL,
            checksum=cid,
            text=f"text-{cid}",
            created_at=datetime.now(UTC),
        )

    store.insert(
        [_chunk("a", company="acme"), _chunk("b", company="beta")],
        [[0.1, 0.2], [0.1, 0.2]],
    )
    token = set_current_tenant(TenantContext(tenant_id="beta"))
    try:
        hits = store.search(vector=[0.1, 0.2], top_k=10, tenant_id="acme")
    finally:
        reset_current_tenant(token)
    assert {h["chunk_id"] for h in hits} == {"a"}
