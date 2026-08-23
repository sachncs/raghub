"""Integration tests for PgVectorStore.

All tests are gated on ``RAG_TEST_PGVECTOR_DSN``. Without the env var
the suite skips cleanly — no failures.
"""

from __future__ import annotations

import os

import pytest

PG_DSN_ENV = "RAG_TEST_PGVECTOR_DSN"


@pytest.mark.skipif(
    not os.environ.get(PG_DSN_ENV),
    reason=f"{PG_DSN_ENV} not set; skipping live Postgres test",
)
class TestPgVectorStoreLive:
    @pytest.mark.asyncio
    async def test_pgvector_search_filters_by_tenant_id(self) -> None:
        """Item 12: PgVectorStore.search builds WHERE tenant_id = $2 when tenant_id is bound."""
        import uuid
        from datetime import UTC, datetime

        from raghub.models import Chunk, Classification
        from raghub.stores.pgvector import PgVectorStore

        dsn = os.environ[PG_DSN_ENV]
        # Unique tenant ids per test run to avoid collisions
        tenant_a = f"tenant_a_{uuid.uuid4().hex[:8]}"
        tenant_b = f"tenant_b_{uuid.uuid4().hex[:8]}"

        store = PgVectorStore(dsn=dsn, embedding_dim=2)

        async def _insert(tenant_id: str) -> None:
            chunk = Chunk(
                id=f"chunk-{tenant_id}",
                document_id=f"doc-{tenant_id}",
                version=1,
                company="acme",
                owner="alice@x",
                classification=Classification.Internal,
                checksum=f"checksum-{tenant_id}",
                text=f"text for {tenant_id}",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                tenant_id=tenant_id,
            )
            await store.insert([chunk], [[0.1, 0.2]])

        await store.create_collection()
        await _insert(tenant_a)
        await _insert(tenant_b)

        # Search scoped to tenant_a returns only its chunk
        hits = await store.search(query_vector=[0.1, 0.2], top_k=10, tenant_id=tenant_a)
        ids = {hit.chunk_id for hit in hits}
        assert ids == {f"chunk-{tenant_a}"}

        # Search scoped to tenant_b returns only its chunk
        hits = await store.search(query_vector=[0.1, 0.2], top_k=10, tenant_id=tenant_b)
        ids = {hit.chunk_id for hit in hits}
        assert ids == {f"chunk-{tenant_b}"}

        # Search without tenant_id returns the rows with NULL tenant_id
        hits = await store.search(query_vector=[0.1, 0.2], top_k=10)
        ids = {hit.chunk_id for hit in hits}
        # The chunks we just inserted have tenant_id set, so this returns nothing
        assert ids == set()
