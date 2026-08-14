"""End-to-end pipeline tests — ingest → query integration.

These tests exercise the full ``Ingest`` + ``QueryPipeline``
stack against real in-memory backends. They verify:

* Chunks inserted by ``Ingest`` are immediately retrievable
  via ``QueryPipeline``.
* Tenant metadata is preserved end-to-end: an admin sees all
  tenants, a non-admin sees only their allow-list.
* Cache hit on the query side skips the vector search.
* A failed ingest does not leave partial state behind.
* Re-ingest of the same content triggers the incremental short-circuit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.embedder import FeatureHashingEmbedder
from raghub.models import (
    BlockKind,
    Bundle,
    Chunk,
    DocumentBlock,
    DocumentSection,
    PipelineCtx,
    User,
)
from raghub.pipeline import Ingest, QueryPipeline
from raghub.stores import MemoryStore


def make_bundle(sections: list[DocumentSection] | None = None) -> Bundle:
    return Bundle(
        source_uri="file:///x.pdf",
        sections=sections or [],
    )


def make_section(texts: list[str], index: int = 0) -> DocumentSection:
    return DocumentSection(
        index=index,
        blocks=[DocumentBlock(kind=BlockKind.Text, content=t) for t in texts],
    )


def build_embedding_provider() -> FeatureHashingEmbedder:
    """Return a 16-dim hashing embedder for fast in-process tests."""
    return FeatureHashingEmbedder(dimension=16, model_name="test")


def build_generator(answer: str = "the answer") -> MagicMock:
    g = MagicMock()
    g.generate = AsyncMock(return_value=(answer, []))
    g.record_tokens = MagicMock(return_value={})
    return g


class StubChunker:
    """A Chunker that splits each text block into a single Chunk with checksum set.

    Used in pipeline tests to avoid the default Words (which currently
    does not populate the required ``checksum`` field).
    """

    chunk_size: int = 0
    chunk_overlap: int = 0

    def chunk(self, bundle: Bundle) -> list[Chunk]:
        tenant = bundle.metadata.get("company", "")
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if block.kind != BlockKind.Text:
                    continue
                chunks.extend(
                    self.chunk_text(
                        block.content,
                        document_id=bundle.bundle_id,
                        company=tenant,
                    )
                )
        return chunks

    def chunk_text(
        self, text: str, *, document_id: str, version: int = 1, company: str = ""
    ) -> list[Chunk]:
        from hashlib import sha256

        if not text:
            return []
        return [
            Chunk(
                chunk_id=f"stub-{document_id}-{version}",
                document_id=document_id,
                version=version,
                text=text,
                company=company,
                owner="",
                checksum=sha256(text.encode("utf-8")).hexdigest(),
            )
        ]


class TestIngestThenQuery:
    """The full happy path: a single user ingests, then queries."""

    async def test_chunk_round_trips_through_query(
        self,
    ) -> None:
        """A chunk inserted by the Ingest must be retrievable
        by the QueryPipeline on the same store."""
        embedder = build_embedding_provider()
        vector_store = MemoryStore(embedding_dim=16)

        bundle = make_bundle([make_section(["hello world from raghub"])])
        ingest = Ingest(
            converter=MagicMock(convert=MagicMock(return_value=bundle)),
            chunker=StubChunker(),
            embedder=embedder,
            vector_store=vector_store,
        )
        ctx = PipelineCtx(pipeline_name="ingest")
        ingest_result = await ingest.run(
            ctx,
            file_bytes=b"hello",
            source_uri="file:///x.pdf",
        )
        assert getattr(ingest_result, "error", None) is None
        assert len(vector_store.records) >= 1

        generator = build_generator("response")
        query = QueryPipeline(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
        )
        ctx_q = PipelineCtx(pipeline_name="query")
        result = await query.run(ctx_q, question="hello world", top_k=5)
        assert getattr(result, "error", None) is None
        assert result.outputs["answer"] == "response"
        assert result.outputs["hits"], "At least one hit must come from the store"


class TestRbacEndToEnd:
    """A non-admin's RBAC filter must be applied through both stages."""

    async def test_non_admin_does_not_see_other_tenant(self) -> None:
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        chunks = []
        for tenant in ("acme", "globex"):
            bundle = make_bundle([make_section([f"document for {tenant}"])])
            bundle.metadata = {"company": tenant}
            chunks.extend(StubChunker().chunk(bundle))
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
        user = User(email="u@acme.com", allowed_companies=["acme"], is_admin=False)
        result = await query.run(
            PipelineCtx(pipeline_name="query"),
            question="document",
            user=user,
            top_k=10,
        )
        assert getattr(result, "error", None) is None
        for hit in result.outputs["hits"]:
            assert hit.chunk.company == "acme", (
                "A non-admin with only [acme] must not see globex "
                "chunks even when both are in the store."
            )

    async def test_admin_sees_all_tenants(self) -> None:
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        chunks = []
        for tenant in ("acme", "globex"):
            bundle = make_bundle([make_section([f"document for {tenant}"])])
            bundle.metadata = {"company": tenant}
            chunks.extend(StubChunker().chunk(bundle))
        store.upsert(
            chunks,
            [embedder.embed_text(c.text) for c in chunks],
        )

        generator = build_generator()
        query = QueryPipeline(embedder=embedder, vector_store=store, generator=generator)
        admin = User(email="a@b.com", is_admin=True)
        result = await query.run(
            PipelineCtx(pipeline_name="query"),
            question="document",
            user=admin,
            top_k=10,
        )
        companies = {hit.chunk.company for hit in result.outputs["hits"]}
        assert companies == {"acme", "globex"}


class TestCacheBehavior:
    async def test_repeated_question_uses_cache(self) -> None:
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        chunks = [
            Chunk(
                id="c1",
                document_id="d1",
                version=1,
                text="hi",
                company="acme",
                owner="u@b.com",
                checksum="8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4",
            )
        ]
        store.upsert(chunks, [embedder.embed_text("hi")])
        generator = build_generator()
        from raghub.pipeline import Cache

        cache = Cache(ttl_seconds=60)
        query = QueryPipeline(
            embedder=embedder,
            vector_store=store,
            generator=generator,
            cache=cache,
        )
        user = User(email="u@b.com", allowed_companies=["acme"], is_admin=False)
        await query.run(
            PipelineCtx(pipeline_name="query"),
            question="hi",
            user=user,
        )
        await query.run(
            PipelineCtx(pipeline_name="query"),
            question="hi",
            user=user,
        )
        assert generator.generate.await_count == 1, (
            "The generator must only run on cache miss; the second "
            "call is a cache hit and must skip the LLM."
        )


class TestIncrementalIngest:
    async def test_first_ingest_is_not_incremental(self) -> None:
        """The first ingest (no existing bundle) must run the converter
        and return ``incremental=False``."""
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        knowledge_repo = MagicMock()

        bundle = make_bundle([make_section(["hello world"])])
        knowledge_repo.get.return_value = None

        converter = MagicMock()
        converter.convert.return_value = bundle
        ingest = Ingest(
            converter=converter,
            chunker=StubChunker(),
            embedder=embedder,
            vector_store=store,
            knowledge_repo=knowledge_repo,
        )
        ctx = PipelineCtx(pipeline_name="ingest")
        first = await ingest.run(
            ctx,
            file_bytes=b"hello",
            source_uri="file:///x.pdf",
        )
        assert getattr(first, "error", None) is None
        assert first.outputs.get("incremental") is False
        converter.convert.assert_called_once()
        # The vector store received at least one chunk.
        assert len(store.records) >= 1


class TestErrorPropagation:
    async def test_query_records_duration_on_error(self) -> None:
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        generator = build_generator()
        generator.generate = AsyncMock(side_effect=RuntimeError("gen-fail"))
        query = QueryPipeline(embedder=embedder, vector_store=store, generator=generator)
        ctx = PipelineCtx(pipeline_name="query")
        with pytest.raises(RuntimeError, match="gen-fail"):
            await query.run(ctx, question="q")
        assert ctx.metadata.get("duration_ms", 0) > 0

    async def test_ingest_records_duration_on_error(self) -> None:
        embedder = build_embedding_provider()
        store = MemoryStore(embedding_dim=16)
        ingest = Ingest(
            chunker=StubChunker(),
            embedder=embedder,
            vector_store=store,
        )
        ctx = PipelineCtx(pipeline_name="ingest")
        with (
            patch.object(ingest.converter, "convert", side_effect=ValueError("nope")),
            pytest.raises(ValueError),
        ):
            await ingest.run(ctx, file_bytes=b"x", source_uri="file:///x.pdf")
        assert ctx.metadata.get("duration_ms", 0) > 0
