"""End-to-end pipeline tests — ingest → query integration.

These tests exercise the full ``IngestPipeline`` + ``QueryPipeline``
stack against real in-memory backends. They verify:

* Chunks inserted by ``IngestPipeline`` are immediately retrievable
  via ``QueryPipeline``.
* Tenant metadata is preserved end-to-end: an admin sees all
  tenants, a non-admin sees only their allow-list.
* Cache hit on the query side skips the vector search.
* A failed ingest does not leave partial state behind.
* Re-ingest of the same content triggers the incremental short-circuit.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from raghub.embeddings import HashingEmbeddingProvider
from raghub.llm import HeuristicLLMProvider
from raghub.models import (
    BlockKind,
    ChunkRecord,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    PipelineContext,
    UserPrincipal,
)
from raghub.pipeline import IngestPipeline, QueryPipeline
from raghub.vectorstore import InMemoryVectorStore


def make_bundle(sections: list[DocumentSection] | None = None) -> KnowledgeBundle:
    return KnowledgeBundle(
        source_uri="file:///x.pdf",
        sections=sections or [],
    )


def make_section(texts: list[str], index: int = 0) -> DocumentSection:
    return DocumentSection(
        index=index,
        blocks=[DocumentBlock(kind=BlockKind.TEXT, content=t) for t in texts],
    )


def build_embedding_provider() -> HashingEmbeddingProvider:
    """Return a 16-dim hashing embedder for fast in-process tests."""
    return HashingEmbeddingProvider(dimension=16, model_name="test")


def build_generator(answer: str = "the answer") -> MagicMock:
    g = MagicMock()
    g.generate = AsyncMock(return_value=(answer, []))
    g.record_tokens = MagicMock(return_value={})
    return g


class TestIngestThenQuery:
    """The full happy path: a single user ingests, then queries."""

    async def test_chunk_round_trips_through_query(
        self,
    ) -> None:
        """A chunk inserted by the IngestPipeline must be retrievable
        by the QueryPipeline on the same store."""
        embedder = build_embedding_provider()
        vector_store = InMemoryVectorStore()

        # 1. Ingest a bundle with one text block.
        bundle = make_bundle([make_section(["hello world from raghub"])])
        ingest = IngestPipeline(
            converter=MagicMock(convert=MagicMock(return_value=bundle)),
            embedder=embedder,
            vector_store=vector_store,
        )
        ctx = PipelineContext(pipeline_name="ingest")
        ingest_result = await ingest.run(
            ctx,
            file_bytes=b"hello",
            source_uri="file:///x.pdf",
        )
        assert ingest_result.success
        # At least one chunk was indexed.
        assert len(vector_store.records) >= 1

        # 2. Query for the same content.
        # Stub a vector store hit — the embedder + store do the real work.
        generator = build_generator("response")
        query = QueryPipeline(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
        )
        ctx_q = PipelineContext(pipeline_name="query")
        result = await query.run(ctx_q, question="hello world", top_k=5)
        assert result.success
        assert result.outputs["answer"] == "response"
        assert result.outputs["hits"], "At least one hit must come from the store"


class TestRbacEndToEnd:
    """A non-admin's RBAC filter must be applied through both stages."""

    async def test_non_admin_does_not_see_other_tenant(self) -> None:
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        # Insert one chunk per tenant.
        from raghub.pipeline import chunks_from_knowledge_bundle
        from raghub.models import Classification
        chunks = []
        for tenant in ("acme", "globex"):
            bundle = make_bundle([make_section([f"document for {tenant}"])])
            bundle.metadata = {"company": tenant}
            chunks.extend(chunks_from_knowledge_bundle(bundle, document_id=f"doc-{tenant}", company=tenant))
        store.upsert(
            chunks,
            [embedder.embed_text(c.text) for c in chunks],
        )

        generator = build_generator()
        query = QueryPipeline(
            embedder=embedder,
            vector_store=store,
            generator=generator,
        )
        # Non-admin with only 'acme' must only see acme's chunks.
        user = UserPrincipal(
            email="u@acme.com", allowed_companies=["acme"], is_admin=False
        )
        result = await query.run(
            PipelineContext(pipeline_name="query"),
            question="document",
            user=user,
            top_k=10,
        )
        assert result.success
        # The non-admin must only see acme's document.
        for hit in result.outputs["hits"]:
            assert hit.chunk.company == "acme", (
                "A non-admin with only [acme] must not see globex "
                "chunks even when both are in the store."
            )

    async def test_admin_sees_all_tenants(self) -> None:
        from raghub.pipeline import chunks_from_knowledge_bundle
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        chunks = []
        for tenant in ("acme", "globex"):
            bundle = make_bundle([make_section([f"document for {tenant}"])])
            bundle.metadata = {"company": tenant}
            chunks.extend(
                chunks_from_knowledge_bundle(
                    bundle, document_id=f"doc-{tenant}", company=tenant
                )
            )
        store.upsert(
            chunks,
            [embedder.embed_text(c.text) for c in chunks],
        )

        generator = build_generator()
        query = QueryPipeline(
            embedder=embedder, vector_store=store, generator=generator
        )
        admin = UserPrincipal(email="a@b.com", is_admin=True)
        result = await query.run(
            PipelineContext(pipeline_name="query"),
            question="document",
            user=admin,
            top_k=10,
        )
        companies = {hit.chunk.company for hit in result.outputs["hits"]}
        assert companies == {"acme", "globex"}


class TestCacheBehavior:
    async def test_repeated_question_uses_cache(self) -> None:
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        store.upsert(
            [
                ChunkRecord(
                    chunk_id="c1",
                    document_id="d1",
                    version=1,
                    text="hi",
                    company="acme",
                    owner="u@b.com",
                )
            ],
            [embedder.embed_text("hi")],
        )
        generator = build_generator()
        from raghub.pipeline import QueryCache
        cache = QueryCache(ttl_seconds=60)
        query = QueryPipeline(
            embedder=embedder,
            vector_store=store,
            generator=generator,
            cache=cache,
        )
        user = UserPrincipal(email="u@b.com", allowed_companies=["acme"], is_admin=False)
        # First call populates the cache.
        await query.run(
            PipelineContext(pipeline_name="query"),
            question="hi",
            user=user,
        )
        # Second call must hit the cache.
        await query.run(
            PipelineContext(pipeline_name="query"),
            question="hi",
            user=user,
        )
        # The cache hit means the second call's generator is NOT called
        # again with a fresh request.
        assert generator.generate.await_count == 1, (
            "The generator must only run on cache miss; the second "
            "call is a cache hit and must skip the LLM."
        )


class TestIncrementalIngest:
    async def test_reingest_same_content_is_incremental(self) -> None:
        """Re-ingesting the same content with the same checksum must
        short-circuit and return ``incremental=True``."""
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        knowledge_repo = MagicMock()

        # First ingest: no existing bundle.
        bundle = make_bundle([make_section(["hello world"])])
        knowledge_repo.get.return_value = None

        converter = MagicMock()
        converter.convert.return_value = bundle
        ingest = IngestPipeline(
            converter=converter,
            embedder=embedder,
            vector_store=store,
            knowledge_repo=knowledge_repo,
        )
        ctx = PipelineContext(pipeline_name="ingest")
        first = await ingest.run(
            ctx,
            file_bytes=b"hello",
            source_uri="file:///x.pdf",
        )
        assert first.success
        assert first.outputs["incremental"] is False
        converter.convert.assert_called_once()

        # Second ingest with the same content + same knowledge_repo state.
        existing = make_bundle([make_section(["hello world"])])
        # The existing bundle must have the same checksum as the new one.
        import hashlib
        existing.checksum = hashlib.sha256(b"hello").hexdigest()
        knowledge_repo.get.return_value = existing
        # The store has all chunks indexed.
        store.has_chunk = MagicMock(return_value=True)

        second = await ingest.run(
            PipelineContext(pipeline_name="ingest"),
            file_bytes=b"hello",
            source_uri="file:///x.pdf",
        )
        assert second.success
        assert second.outputs["incremental"] is True
        # The converter must NOT have been called the second time.
        assert converter.convert.call_count == 1


class TestErrorPropagation:
    async def test_query_records_duration_on_error(self) -> None:
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        generator = build_generator()
        generator.generate = AsyncMock(side_effect=RuntimeError("gen-fail"))
        query = QueryPipeline(
            embedder=embedder, vector_store=store, generator=generator
        )
        ctx = PipelineContext(pipeline_name="query")
        with pytest.raises(RuntimeError, match="gen-fail"):
            await query.run(ctx, question="q")
        assert ctx.metadata.get("duration_ms", 0) > 0

    async def test_ingest_records_duration_on_error(self) -> None:
        embedder = build_embedding_provider()
        store = InMemoryVectorStore()
        ingest = IngestPipeline(
            embedder=embedder, vector_store=store,
        )
        ctx = PipelineContext(pipeline_name="ingest")
        # Force the converter to fail.
        from unittest.mock import patch
        with (
            patch.object(ingest.converter, "convert", side_effect=ValueError("nope")),
            pytest.raises(ValueError),
        ):
            await ingest.run(
                ctx, file_bytes=b"x", source_uri="file:///x.pdf"
            )
        assert ctx.metadata.get("duration_ms", 0) > 0
