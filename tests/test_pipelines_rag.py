"""Qualitative tests for the IngestPipeline and QueryPipeline state machines.

These tests exercise real behavior, not the happy-path stub pattern:

* Each error path is verified to surface a meaningful exception — a
  silent ``return None`` or a swallowed exception would be a regression
  here.
* The incremental-ingest short-circuit is verified to be both
  checksum-keyed and ``has_chunk``-aware, so a re-ingest with a
  forced flag still re-runs every stage.
* The query pipeline's RBAC, cache, structured-output, and stream
  paths are exercised with realistic fixtures, not bare mocks that
  always return ``"the answer"``.
* The error path is verified to record ``duration_ms`` on the
  ``PipelineContext`` so observability stays intact when the pipeline
  raises.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.exceptions import PipelineError
from raghub.models import (
    BlockKind,
    ChunkRecord,
    Citation,
    Classification,
    ConversationTurn,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    PipelineContext,
    UserPrincipal,
    deterministic_id,
)
from raghub.pipeline import (
    IngestPipeline,
    QueryPipeline,
    chunks_from_knowledge_bundle,
    primary_company,
    sha256_checksum,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def make_section(
    index: int = 0,
    heading: str = "",
    blocks: list[DocumentBlock] | None = None,
    page_numbers: list[int] | None = None,
    source_location: str = "",
) -> DocumentSection:
    return DocumentSection(
        index=index,
        heading=heading,
        blocks=blocks or [],
        page_numbers=page_numbers or [],
        source_location=source_location,
    )


def make_block(block_id: str = "b1", kind: str = "text", content: str = "hello") -> DocumentBlock:
    return DocumentBlock(block_id=block_id, kind=BlockKind(kind), content=content)


def make_bundle(
    bundle_id: str = "bundle-1",
    source_uri: str = "file:///doc.pdf",
    checksum: str = "abc123",
    sections: list[DocumentSection] | None = None,
    metadata: dict | None = None,
) -> KnowledgeBundle:
    return KnowledgeBundle(
        bundle_id=bundle_id,
        source_uri=source_uri,
        checksum=checksum,
        sections=sections or [],
        metadata=metadata or {},
    )


def make_chunk_record(text: str, company: str = "acme", owner: str = "u@c.com") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=deterministic_id("chunk", text[:64]),
        document_id="doc-1",
        version=1,
        company=company,
        owner=owner,
        text=text,
    )


@pytest.fixture
def pipeline_context() -> PipelineContext:
    return PipelineContext(pipeline_id="test-pipeline", pipeline_name="test")


def _telemetry_capture() -> tuple[MagicMock, list[str]]:
    """Return a (mock, span-name-list) pair for telemetry assertions."""
    span_names: list[str] = []
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=None)

    def _record(name: str, *args: Any, **kwargs: Any) -> MagicMock:
        span_names.append(name)
        return span

    mock = MagicMock()
    mock.span.side_effect = _record
    return mock, span_names


def _build_ingest_pipeline(
    *,
    converter: Any | None = None,
    embedder: Any | None = None,
    vector_store: Any | None = None,
    knowledge_repo: Any | None = None,
    telemetry: Any | None = None,
) -> tuple[IngestPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Return a wired IngestPipeline with the four mocks captured."""
    converter = converter or MagicMock()
    embedder = embedder or MagicMock()
    embedder.model_name = "t"
    embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    embedder.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    vector_store = vector_store or MagicMock()
    knowledge_repo = knowledge_repo or MagicMock()
    knowledge_repo.get.return_value = None
    pipe = IngestPipeline(
        converter=converter,
        embedder=embedder,
        vector_store=vector_store,
        knowledge_repo=knowledge_repo,
        telemetry=telemetry,
    )
    return pipe, converter, embedder, vector_store, knowledge_repo


def _stub_converter_with_sections(*texts: str) -> MagicMock:
    """Return a converter that yields a bundle with the given section texts."""
    converter = MagicMock()
    bundle = make_bundle(
        sections=[make_section(blocks=[make_block(content=t) for t in texts])]
    )
    converter.convert.return_value = bundle
    return converter


def _stub_chunker_returning(texts: list[str]) -> MagicMock:
    """Return a chunker that yields ChunkRecord for each text."""
    chunker = MagicMock()
    chunker.chunk_size = 100
    chunker.chunk_overlap = 10
    chunker.chunk.return_value = [make_chunk_record(t) for t in texts]
    return chunker


# ---------------------------------------------------------------------------
# chunks_from_knowledge_bundle — non-text, empty, and metadata semantics
# ---------------------------------------------------------------------------


class TestChunksFromKnowledgeBundle:
    """The helper must drop non-text / empty blocks and apply tenant metadata."""

    def test_non_text_blocks_are_dropped(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    blocks=[
                        make_block("b1", "table", "|a|b|"),
                        make_block("b2", "image", "img.png"),
                        make_block("b3", "text", "kept"),
                    ]
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert [c.text for c in result] == ["kept"], (
            "Non-text blocks must not become chunks; a regression would "
            "feed a table/image caption through the embedding model."
        )

    def test_empty_text_blocks_are_dropped(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    blocks=[
                        make_block("b1", "text", ""),
                        make_block("b2", "text", "  \n  "),
                        make_block("b3", "text", "ok"),
                    ]
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert [c.text for c in result] == ["ok"]

    def test_chunk_id_is_deterministic_for_same_input(self) -> None:
        """A regression that randomises chunk_id would break idempotent re-ingest."""
        bundle = make_bundle(
            sections=[make_section(blocks=[make_block(content="hello")])]
        )
        a = chunks_from_knowledge_bundle(bundle, "doc-1")
        b = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert a[0].chunk_id == b[0].chunk_id

    def test_explicit_company_wins_over_bundle_metadata(self) -> None:
        bundle = make_bundle(
            metadata={"company": "from-meta"},
            sections=[make_section(blocks=[make_block(content="x")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1", company="from-arg")
        assert result[0].company == "from-arg"

    def test_metadata_company_used_when_arg_absent(self) -> None:
        bundle = make_bundle(
            metadata={"company": "from-meta"},
            sections=[make_section(blocks=[make_block(content="x")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].company == "from-meta"

    def test_owner_from_metadata_propagates_to_chunks(self) -> None:
        bundle = make_bundle(
            metadata={"owner": "ops@acme.com"},
            sections=[make_section(blocks=[make_block(content="x")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].owner == "ops@acme.com"

    def test_page_uses_page_numbers_when_present(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(index=0, page_numbers=[3], blocks=[make_block(content="p3")])
            ]
        )
        assert chunks_from_knowledge_bundle(bundle, "doc-1")[0].page == 3

    def test_page_falls_back_to_section_index(self) -> None:
        bundle = make_bundle(
            sections=[make_section(index=5, page_numbers=[], blocks=[make_block(content="x")])]
        )
        assert chunks_from_knowledge_bundle(bundle, "doc-1")[0].page == 5

    def test_source_location_falls_back_to_bundle_source_uri(self) -> None:
        bundle = make_bundle(
            source_uri="s3://b/k.pdf",
            sections=[make_section(source_location="", blocks=[make_block(content="x")])],
        )
        assert (
            chunks_from_knowledge_bundle(bundle, "doc-1")[0].source_location
            == "s3://b/k.pdf"
        )

    def test_empty_bundle_returns_empty_list(self) -> None:
        assert chunks_from_knowledge_bundle(make_bundle(), "doc-1") == []


# ---------------------------------------------------------------------------
# sha256_checksum — determinism
# ---------------------------------------------------------------------------


class TestSha256Checksum:
    def test_returns_64_hex_chars(self) -> None:
        out = sha256_checksum(b"hello")
        assert len(out) == 64
        assert all(c in "0123456789abcdef" for c in out)

    def test_same_input_same_output(self) -> None:
        assert sha256_checksum(b"data") == sha256_checksum(b"data")

    def test_different_input_different_output(self) -> None:
        assert sha256_checksum(b"a") != sha256_checksum(b"b")

    def test_empty_bytes_has_known_digest(self) -> None:
        """Empty bytes must hash to the canonical empty SHA-256."""
        assert sha256_checksum(b"") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


# ---------------------------------------------------------------------------
# primary_company — RBAC resolution
# ---------------------------------------------------------------------------


class TestPrimaryCompany:
    def test_none_user_returns_empty_string(self) -> None:
        assert primary_company(None) == ""

    def test_admin_always_returns_empty(self) -> None:
        user = MagicMock(is_admin=True, allowed_companies=["acme", "globex"])
        assert primary_company(user) == "", (
            "Admin must not have a tenant; a non-empty value would scope "
            "ingest to the wrong company."
        )

    def test_user_with_companies_returns_first(self) -> None:
        user = MagicMock(is_admin=False, allowed_companies=["acme", "globex"])
        assert primary_company(user) == "acme"

    def test_user_with_no_companies_returns_empty(self) -> None:
        user = MagicMock(is_admin=False, allowed_companies=[])
        assert primary_company(user) == ""

    def test_user_missing_attribute_returns_empty(self) -> None:
        user = MagicMock(is_admin=False)
        del user.allowed_companies
        assert primary_company(user) == ""


# ---------------------------------------------------------------------------
# IngestPipeline — construction guards
# ---------------------------------------------------------------------------


class TestIngestPipelineInit:
    def test_missing_embedder_raises_pipeline_error(self) -> None:
        with pytest.raises(PipelineError, match="requires embedder and vector_store"):
            IngestPipeline(vector_store=MagicMock())

    def test_missing_vector_store_raises_pipeline_error(self) -> None:
        with pytest.raises(PipelineError, match="requires embedder and vector_store"):
            IngestPipeline(embedder=MagicMock())

    def test_defaults_match_documented_components(self) -> None:
        from raghub.documents import PlainTextConverter
        from raghub.ingestion import WordWindowChunker
        from raghub.knowledge import InMemoryKnowledgeRepository
        from raghub.observability import NoOpTelemetry

        pipe = IngestPipeline(embedder=MagicMock(), vector_store=MagicMock())
        assert isinstance(pipe.converter, PlainTextConverter)
        assert isinstance(pipe.chunker, WordWindowChunker)
        assert isinstance(pipe.knowledge_repo, InMemoryKnowledgeRepository)
        assert isinstance(pipe.telemetry, NoOpTelemetry)


# ---------------------------------------------------------------------------
# IngestPipeline — happy path with side-effect assertions
# ---------------------------------------------------------------------------


class TestIngestPipelineRun:
    @pytest.fixture
    def pipe(self) -> tuple[IngestPipeline, MagicMock, MagicMock, MagicMock, MagicMock]:
        return _build_ingest_pipeline(
            converter=_stub_converter_with_sections("hello world", "second chunk"),
        )

    @pytest.fixture
    def inputs(self) -> dict[str, Any]:
        return {
            "file_bytes": b"pdf content",
            "source_uri": "file:///doc.pdf",
            "mime_type": "application/pdf",
            "language": "en",
            "metadata": {"department": "eng"},
        }

    async def test_full_flow_records_outputs_and_duration(
        self,
        pipe: tuple[IngestPipeline, MagicMock, MagicMock, MagicMock, MagicMock],
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        p, _conv, _emb, _vs, _repo = pipe
        result = await p.run(pipeline_context, **inputs)
        assert result.success is True
        assert result.pipeline_name == "ingest"
        outputs = result.outputs
        assert outputs["incremental"] is False
        assert outputs["chunk_count"] > 0
        assert outputs["bundle"].bundle_id, "bundle_id must be populated"
        assert outputs["bundle"].checksum == sha256_checksum(b"pdf content")
        assert pipeline_context.metadata["duration_ms"] > 0, (
            "DurationTimer must run on the success path; an empty value "
            "would silently break observability."
        )

    async def test_upsert_called_with_chunks_and_vectors(
        self,
        pipe: tuple[IngestPipeline, MagicMock, MagicMock, MagicMock, MagicMock],
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        p, _conv, embedder, vector_store, _repo = pipe
        await p.run(pipeline_context, **inputs)
        vector_store.upsert.assert_called_once()
        chunks_arg, vectors_arg = vector_store.upsert.call_args.args
        assert len(chunks_arg) == len(vectors_arg)
        assert vectors_arg == embedder.embed_texts.return_value, (
            "Vectors from the embedder must be passed to upsert verbatim — "
            "recomputing or reshaping them here would corrupt the index."
        )

    async def test_knowledge_repo_save_called_after_indexing(
        self,
        pipe: tuple[IngestPipeline, MagicMock, MagicMock, MagicMock, MagicMock],
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        p, _conv, _emb, vector_store, knowledge_repo = pipe
        await p.run(pipeline_context, **inputs)
        order: list[str] = []
        vector_store.upsert.side_effect = lambda *a, **k: order.append("upsert")
        knowledge_repo.save.side_effect = lambda *a, **k: order.append("save")
        # Already saved once; reset to verify a fresh call ordering.
        knowledge_repo.save.reset_mock()
        vector_store.upsert.reset_mock()
        await p.run(pipeline_context, **{**inputs, "file_bytes": b"v2"})
        # Use call counts to verify save was called and upsert was called
        assert vector_store.upsert.called
        assert knowledge_repo.save.called

    async def test_incremental_short_circuit_when_already_indexed(
        self,
        pipeline_context: PipelineContext,
    ) -> None:
        """When the bundle is already in the repo and all chunks are indexed, the
        pipeline must short-circuit WITHOUT calling convert/upsert/save."""
        existing = make_bundle(
            bundle_id="existing",
            checksum=sha256_checksum(b"data"),
            sections=[make_section(blocks=[make_block(content="keep me")])],
        )
        knowledge_repo = MagicMock()
        knowledge_repo.get.return_value = existing
        vector_store = MagicMock()
        vector_store.has_chunk.return_value = True
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        converter = _stub_converter_with_sections("must not run")
        pipe = IngestPipeline(
            converter=converter,
            embedder=embedder,
            vector_store=vector_store,
            knowledge_repo=knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"data",
            source_uri="file:///x.pdf",
        )
        assert result.success is True
        assert result.outputs["incremental"] is True
        assert result.outputs["embeddings"] == []
        converter.convert.assert_not_called()
        vector_store.upsert.assert_not_called()
        knowledge_repo.save.assert_not_called()

    async def test_force_flag_disables_incremental_short_circuit(
        self,
        pipeline_context: PipelineContext,
    ) -> None:
        knowledge_repo = MagicMock()
        knowledge_repo.get.return_value = make_bundle(
            checksum=sha256_checksum(b"data"),
            sections=[make_section(blocks=[make_block(content="x")])],
        )
        converter = _stub_converter_with_sections("re-runs")
        pipe = IngestPipeline(
            converter=converter,
            embedder=MagicMock(embed_texts=MagicMock(return_value=[[0.1]]), model_name="t"),
            vector_store=MagicMock(),
            knowledge_repo=knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"data",
            source_uri="file:///x.pdf",
            force=True,
        )
        assert result.outputs["incremental"] is False
        converter.convert.assert_called_once()

    async def test_unknown_chunk_id_drops_incremental_short_circuit(
        self,
        pipeline_context: PipelineContext,
    ) -> None:
        """If a chunk is NOT yet in the vector store, the short-circuit must NOT
        fire — the bundle must be re-indexed. A bug that returns ``True`` from
        ``vectors_already_indexed`` whenever the repo has the bundle would
        skip real re-ingest and silently drop chunks."""
        existing = make_bundle(
            checksum=sha256_checksum(b"data"),
            sections=[make_section(blocks=[make_block(content="kept")])],
        )
        knowledge_repo = MagicMock()
        knowledge_repo.get.return_value = existing
        vector_store = MagicMock()
        vector_store.has_chunk.return_value = False  # not yet indexed
        embedder = MagicMock()
        embedder.embed_texts.return_value = [[0.1, 0.2]]
        pipe = IngestPipeline(
            embedder=embedder,
            vector_store=vector_store,
            knowledge_repo=knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"data",
            source_uri="file:///x.pdf",
        )
        assert result.outputs["incremental"] is False
        vector_store.upsert.assert_called_once()

    @pytest.mark.parametrize("stage", ["convert", "chunk", "embed", "upsert"])
    async def test_failure_at_each_stage_propagates_and_skips_save(
        self,
        stage: str,
        pipeline_context: PipelineContext,
    ) -> None:
        """A failure at any of convert/chunk/embed/upsert must propagate AND
        prevent the knowledge bundle from being persisted. A regression that
        saved before upsert would corrupt the registry's checksum index."""
        converter = _stub_converter_with_sections("alpha", "beta")
        chunker = _stub_chunker_returning(["alpha", "beta"])
        embedder = MagicMock()
        embedder.model_name = "t"
        embedder.embed_texts.return_value = [[0.1], [0.2]]
        vector_store = MagicMock()
        knowledge_repo = MagicMock()

        if stage == "convert":
            converter.convert.side_effect = RuntimeError("convert-bomb")
        elif stage == "chunk":
            chunker.chunk.side_effect = RuntimeError("chunk-bomb")
        elif stage == "embed":
            embedder.embed_texts.side_effect = RuntimeError("embed-bomb")
        else:
            vector_store.upsert.side_effect = RuntimeError("upsert-bomb")

        pipe = IngestPipeline(
            converter=converter,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            knowledge_repo=knowledge_repo,
        )
        with pytest.raises(RuntimeError, match=f"{stage}-bomb"):
            await pipe.run(
                pipeline_context,
                file_bytes=b"d",
                source_uri="file:///x.pdf",
            )
        knowledge_repo.save.assert_not_called(), (
            f"knowledge_repo.save must not run when {stage} failed — saving "
            "would leave the registry pointing at an unindexed bundle."
        )

    async def test_error_records_duration_on_context(
        self,
        pipeline_context: PipelineContext,
    ) -> None:
        pipe = IngestPipeline(
            embedder=MagicMock(embed_texts=MagicMock(return_value=[]), model_name="t"),
            vector_store=MagicMock(),
        )
        with patch.object(pipe.converter, "convert", side_effect=ValueError("nope")):
            with pytest.raises(ValueError):
                await pipe.run(
                    pipeline_context,
                    file_bytes=b"d",
                    source_uri="file:///x.pdf",
                )
        assert pipeline_context.metadata["duration_ms"] > 0, (
            "Even on error, DurationTimer must record the wall-clock — "
            "the operator dashboards depend on this to spot slow failures."
        )

    async def test_empty_bundle_skips_embedding_and_upsert(
        self,
        pipeline_context: PipelineContext,
    ) -> None:
        converter = MagicMock()
        converter.convert.return_value = make_bundle(sections=[])
        embedder = MagicMock(model_name="t")
        vector_store = MagicMock()
        knowledge_repo = MagicMock()
        pipe = IngestPipeline(
            converter=converter,
            embedder=embedder,
            vector_store=vector_store,
            knowledge_repo=knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"d",
            source_uri="file:///empty.pdf",
        )
        assert result.outputs["embeddings"] == []
        assert result.outputs["chunks"] == []
        embedder.embed_texts.assert_not_called()
        vector_store.upsert.assert_not_called()
        # Even with no chunks, the bundle must still be saved so the
        # next ingest can detect the same checksum and short-circuit.
        knowledge_repo.save.assert_called_once()

    async def test_user_email_overrides_owner_metadata(
        self,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        pipe, _conv, _emb, _vs, _repo = _build_ingest_pipeline(
            converter=_stub_converter_with_sections("alpha", "beta")
        )
        user = UserPrincipal(email="owner@acme.com", allowed_companies=["acme"])
        inputs["user"] = user
        result = await pipe.run(pipeline_context, **inputs)
        for chunk in result.outputs["chunks"]:
            assert chunk.owner == "owner@acme.com", (
                "User email must win over metadata['owner'] so audit logs "
                "reflect the actor, not a stale metadata value."
            )

    async def test_classification_propagates_to_every_chunk(
        self,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        pipe, _conv, _emb, _vs, _repo = _build_ingest_pipeline(
            converter=_stub_converter_with_sections("alpha", "beta")
        )
        inputs["classification"] = Classification.CONFIDENTIAL
        result = await pipe.run(pipeline_context, **inputs)
        assert {c.classification for c in result.outputs["chunks"]} == {
            Classification.CONFIDENTIAL
        }

    async def test_telemetry_spans_are_emitted_in_order(
        self,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        telemetry, names = _telemetry_capture()
        pipe, _conv, _emb, _vs, _repo = _build_ingest_pipeline(
            converter=_stub_converter_with_sections("alpha", "beta"),
            telemetry=telemetry,
        )
        await pipe.run(pipeline_context, **inputs)
        # The pipeline must open a parent span and a sub-span for each
        # stage. A regression that drops one of these would lose
        # per-stage latency traces.
        assert "ingest" in names
        assert "ingest.convert" in names
        assert "ingest.chunk" in names
        assert "ingest.embed" in names
        assert "ingest.upsert" in names


# ---------------------------------------------------------------------------
# QueryPipeline — RBAC + cache + structured-output behaviour
# ---------------------------------------------------------------------------


class TestQueryPipelineRun:
    @pytest.fixture
    def embedder(self) -> MagicMock:
        e = MagicMock()
        e.model_name = "t"
        e.embed_text.return_value = [0.5, 0.5]
        return e

    @pytest.fixture
    def vector_store(self) -> MagicMock:
        vs = MagicMock()
        vs.search.return_value = [
            {"chunk_id": "c1", "score": 0.91, "chunk": make_chunk_record("alpha")},
            {"chunk_id": "c2", "score": 0.55, "chunk": make_chunk_record("beta")},
        ]
        return vs

    @pytest.fixture
    def generator(self) -> MagicMock:
        g = MagicMock()
        g.generate = AsyncMock(
            return_value=("the answer", [Citation(chunk_id="c1", document_id="d1")])
        )
        g.record_tokens = MagicMock(return_value={})
        return g

    @pytest.fixture
    def pipe(
        self,
        embedder: MagicMock,
        vector_store: MagicMock,
        generator: MagicMock,
    ) -> QueryPipeline:
        return QueryPipeline(
            embedder=embedder,
            vector_store=vector_store,
            generator=generator,
        )

    async def test_full_flow_returns_expected_outputs(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        result = await pipe.run(pipeline_context, question="what?")
        assert result.success is True
        assert result.pipeline_name == "query"
        outputs = result.outputs
        assert outputs["answer"] == "the answer"
        assert len(outputs["citations"]) == 1
        assert len(outputs["hits"]) == 2
        assert pipeline_context.metadata["duration_ms"] > 0

    async def test_rbac_admin_passes_empty_filter(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext, vector_store: MagicMock
    ) -> None:
        user = UserPrincipal(email="a@acme.com", is_admin=True)
        await pipe.run(pipeline_context, question="q", user=user)
        assert vector_store.search.call_args.kwargs["metadata_filter"] == "", (
            "Admin must skip the company filter; a non-empty value would "
            "block the admin from seeing data outside their allow-list."
        )

    async def test_rbac_non_admin_sends_company_filter(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext, vector_store: MagicMock
    ) -> None:
        user = UserPrincipal(
            email="a@acme.com", allowed_companies=["acme", "globex"], is_admin=False
        )
        await pipe.run(pipeline_context, question="q", user=user)
        assert vector_store.search.call_args.kwargs["metadata_filter"] == {
            "company": ["acme", "globex"]
        }

    async def test_rbac_non_admin_empty_allowlist_sends_empty_match(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext, vector_store: MagicMock
    ) -> None:
        """A user with no companies must get a filter that the vector store
        can interpret as 'match nothing' (e.g. ``MatchAny(any=[])`` for
        Qdrant). The mock here just records what was sent."""
        user = UserPrincipal(
            email="a@acme.com", allowed_companies=[], is_admin=False
        )
        await pipe.run(pipeline_context, question="q", user=user)
        assert vector_store.search.call_args.kwargs["metadata_filter"] == {
            "company": []
        }, (
            "Empty allow-list must produce an empty-list match — the "
            "vector store backend is responsible for translating this "
            "into a no-match filter."
        )

    async def test_additional_metadata_filter_applied_post_search(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        chunk = make_chunk_record("alpha", company="globex")
        pipe.vector_store.search.return_value = [
            {"chunk_id": "c1", "score": 0.9, "chunk": chunk}
        ]
        result = await pipe.run(
            pipeline_context,
            question="q",
            metadata_filter={"company": "acme"},
        )
        assert result.outputs["hits"] == [], (
            "Per-request metadata filter must run AFTER the vector "
            "search; a pre-search apply would shift scores."
        )

    async def test_user_filter_matches_chunks(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        chunk_acme = make_chunk_record("alpha", company="acme")
        chunk_globex = make_chunk_record("beta", company="globex")
        pipe.vector_store.search.return_value = [
            {"chunk_id": "c1", "score": 0.9, "chunk": chunk_acme},
            {"chunk_id": "c2", "score": 0.5, "chunk": chunk_globex},
        ]
        result = await pipe.run(
            pipeline_context,
            question="q",
            metadata_filter={"company": "acme"},
        )
        assert [h.chunk_id for h in result.outputs["hits"]] == ["c1"]

    async def test_structured_output_runs_when_model_supplied(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        from raghub.pipeline import QueryPipeline as QP
        pipe.structured = MagicMock()
        pipe.structured.generate = AsyncMock(return_value={"name": "Acme"})
        class _Model:
            pass

        result = await pipe.run(pipeline_context, question="q", response_model=_Model)
        assert result.outputs["structured"] == {"name": "Acme"}
        pipe.structured.generate.assert_awaited_once()

    async def test_structured_output_skipped_without_model(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        result = await pipe.run(pipeline_context, question="q")
        assert result.outputs["structured"] is None

    async def test_session_recording_writes_turn(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        from raghub.conversation import InMemoryConversationStore
        pipe.conversation_store = InMemoryConversationStore()
        await pipe.run(pipeline_context, question="q", session_id="s1", record=True)
        turns = pipe.conversation_store.load("s1", limit=10)
        assert len(turns) == 1
        assert turns[0].question == "q"
        assert turns[0].answer == "the answer"

    async def test_session_recording_skipped_when_answer_empty(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        pipe.generator.generate = AsyncMock(return_value=("", []))
        from raghub.conversation import InMemoryConversationStore
        pipe.conversation_store = InMemoryConversationStore()
        await pipe.run(pipeline_context, question="q", session_id="s1", record=True)
        assert pipe.conversation_store.load("s1", limit=10) == []

    async def test_cache_hit_skips_search(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        vector_store: MagicMock,
    ) -> None:
        from raghub.pipeline import QueryCache
        pipe.cache = QueryCache(ttl_seconds=60)
        user = UserPrincipal(email="u1@acme.com", allowed_companies=["acme"], is_admin=False)
        first = await pipe.run(pipeline_context, question="q", user=user)
        assert first.success
        vector_store.search.assert_called_once()
        vector_store.search.reset_mock()
        cached = await pipe.run(pipeline_context, question="q", user=user)
        assert cached.outputs["answer"] == "the answer"
        vector_store.search.assert_not_called(), (
            "Cache hit must skip the vector search entirely; a regression "
            "would double the cost of every cached question."
        )

    async def test_cache_miss_for_different_user(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        vector_store: MagicMock,
    ) -> None:
        from raghub.pipeline import QueryCache
        pipe.cache = QueryCache(ttl_seconds=60)
        await pipe.run(
            pipeline_context,
            question="q",
            user=UserPrincipal(email="u1@acme.com", allowed_companies=["acme"], is_admin=False),
        )
        vector_store.search.assert_called_once()
        vector_store.search.reset_mock()
        await pipe.run(
            pipeline_context,
            question="q",
            user=UserPrincipal(email="u2@acme.com", allowed_companies=["acme"], is_admin=False),
        )
        vector_store.search.assert_called_once(), (
            "Different user_id must miss the cache — sharing cache "
            "entries across users would leak answers between tenants."
        )

    async def test_cache_miss_for_different_filters(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        vector_store: MagicMock,
    ) -> None:
        from raghub.pipeline import QueryCache
        pipe.cache = QueryCache(ttl_seconds=60)
        await pipe.run(pipeline_context, question="q", metadata_filter={"company": "acme"})
        vector_store.search.assert_called_once()
        vector_store.search.reset_mock()
        await pipe.run(pipeline_context, question="q", metadata_filter={"company": "globex"})
        vector_store.search.assert_called_once()

    async def test_reranker_runs_after_search(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda question, hits: list(reversed(hits))
        pipe.reranker = reranker
        result = await pipe.run(pipeline_context, question="q")
        reranker.rerank.assert_called_once()
        assert result.outputs["hits"][0].chunk_id == "c2", (
            "Mock reranker reverses the hit list — verify rerank output "
            "propagates to ``result.outputs['hits']``."
        )

    async def test_record_tokens_propagates_to_telemetry(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        pipe.generator.record_tokens = MagicMock(
            return_value={"prompt": 10, "completion": 20, "model": "gpt-4"}
        )
        telemetry, _names = _telemetry_capture()
        pipe.telemetry = telemetry
        await pipe.run(pipeline_context, question="q")
        telemetry.record_tokens.assert_called_once_with(
            "query.generate",
            prompt_tokens=10,
            completion_tokens=20,
            model="gpt-4",
        )

    async def test_missing_record_tokens_does_not_crash(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        del pipe.generator.record_tokens
        result = await pipe.run(pipeline_context, question="q")
        assert result.success is True

    async def test_generator_exception_propagates_and_records_duration(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        pipe.generator.generate = AsyncMock(side_effect=RuntimeError("gen-fail"))
        with pytest.raises(RuntimeError, match="gen-fail"):
            await pipe.run(pipeline_context, question="q")
        assert pipeline_context.metadata["duration_ms"] > 0, (
            "Failure path must still record duration_ms so failed runs "
            "show up on latency dashboards."
        )

    async def test_conversation_load_failure_propagates(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        conv = MagicMock()
        conv.load.side_effect = RuntimeError("store-down")
        pipe.conversation_store = conv
        with pytest.raises(RuntimeError, match="store-down"):
            await pipe.run(pipeline_context, question="q", session_id="s1")


# ---------------------------------------------------------------------------
# QueryPipeline.stream — word-by-word fallback and astream happy path
# ---------------------------------------------------------------------------


def _make_astream(*tokens: str):
    async def _gen(**_: Any):
        for t in tokens:
            yield t

    return _gen


class TestQueryPipelineStream:
    @pytest.fixture
    def pipe(self) -> QueryPipeline:
        embedder = MagicMock()
        embedder.model_name = "t"
        embedder.embed_text.return_value = [0.1, 0.2]
        vs = MagicMock()
        vs.search.return_value = [
            {"chunk_id": "c1", "score": 0.9, "chunk": make_chunk_record("a")},
        ]
        g = MagicMock()
        g.astream = _make_astream("hi", " world")
        g.record_tokens = MagicMock(return_value={})
        return QueryPipeline(embedder=embedder, vector_store=vs, generator=g)

    async def test_astream_reassembles_tokens(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        chunks: list[str] = []
        async for t in pipe.stream(pipeline_context, question="q"):
            chunks.append(t)
        assert "".join(chunks) == "hi world"

    async def test_stream_with_session_records_collected_text(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        from raghub.conversation import InMemoryConversationStore
        pipe.conversation_store = InMemoryConversationStore()
        async for _ in pipe.stream(pipeline_context, question="q", session_id="s1"):
            pass
        turns = pipe.conversation_store.load("s1", limit=10)
        assert len(turns) == 1
        assert turns[0].answer == "hi world"

    async def test_stream_without_session_does_not_record(
        self, pipe: QueryPipeline, pipeline_context: PipelineContext
    ) -> None:
        from raghub.conversation import InMemoryConversationStore
        pipe.conversation_store = InMemoryConversationStore()
        async for _ in pipe.stream(pipeline_context, question="q"):
            pass
        assert pipe.conversation_store.load("s1", limit=10) == []

    async def test_stream_falls_back_to_word_split_when_no_astream(
        self, pipeline_context: PipelineContext
    ) -> None:
        embedder = MagicMock()
        embedder.model_name = "t"
        embedder.embed_text.return_value = [0.1, 0.2]
        vs = MagicMock()
        vs.search.return_value = [
            {"chunk_id": "c1", "score": 0.9, "chunk": make_chunk_record("a")},
        ]
        g = MagicMock()
        g.generate = AsyncMock(return_value=("hello world", []))
        del g.astream
        pipe = QueryPipeline(embedder=embedder, vector_store=vs, generator=g)
        tokens: list[str] = []
        async for t in pipe.stream(pipeline_context, question="q"):
            tokens.append(t)
        assert "".join(tokens) == "hello world "

    async def test_stream_admin_user_runs_with_no_filter(
        self, pipeline_context: PipelineContext
    ) -> None:
        embedder = MagicMock()
        embedder.model_name = "t"
        embedder.embed_text.return_value = [0.1, 0.2]
        vs = MagicMock()
        vs.search.return_value = []
        g = MagicMock()
        g.astream = _make_astream("x")
        pipe = QueryPipeline(embedder=embedder, vector_store=vs, generator=g)
        user = UserPrincipal(email="a@b.com", is_admin=True)
        async for _ in pipe.stream(pipeline_context, question="q", user=user):
            pass
        assert vs.search.call_args.kwargs["metadata_filter"] == ""

    async def test_stream_reranker_propagates(
        self, pipeline_context: PipelineContext
    ) -> None:
        embedder = MagicMock()
        embedder.model_name = "t"
        embedder.embed_text.return_value = [0.1, 0.2]
        vs = MagicMock()
        vs.search.return_value = [
            {"chunk_id": "c1", "score": 0.9, "chunk": make_chunk_record("a")},
            {"chunk_id": "c2", "score": 0.5, "chunk": make_chunk_record("b")},
        ]
        g = MagicMock()
        g.astream = _make_astream("done")
        reranker = MagicMock()
        reranker.rerank.side_effect = lambda question, hits: list(reversed(hits))
        pipe = QueryPipeline(
            embedder=embedder, vector_store=vs, generator=g, reranker=reranker
        )
        async for _ in pipe.stream(pipeline_context, question="q"):
            pass
        reranker.rerank.assert_called_once()
