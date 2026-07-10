from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.exceptions import PipelineError
from raghub.models import (
    BlockKind,
    ChunkRecord,
    Citation,
    ConversationTurn,
    DocumentBlock,
    DocumentSection,
    KnowledgeBundle,
    PipelineContext,
    UserPrincipal,
    deterministic_id,
)
from raghub.pipelines.rag import (
    IngestPipeline,
    QueryPipeline,
    chunks_from_knowledge_bundle,
    primary_company,
    sha256_checksum,
)


# ---------------------------------------------------------------------------
# Helper factories
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


@pytest.fixture
def mock_converter() -> MagicMock:
    m = MagicMock()
    bundle = make_bundle(
        sections=[make_section(index=0, blocks=[make_block(content="hello world")])]
    )
    m.convert.return_value = bundle
    return m


@pytest.fixture
def mock_chunker() -> MagicMock:
    m = MagicMock()
    m.chunk_size = 100
    m.chunk_overlap = 10
    m.chunk.return_value = [
        make_chunk_record("hello world"),
        make_chunk_record("foo bar"),
    ]
    return m


@pytest.fixture
def mock_embedder() -> MagicMock:
    m = MagicMock()
    m.model_name = "test-model"
    m.embed_text.return_value = [0.1, 0.2, 0.3]
    m.embed_texts.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return m


@pytest.fixture
def mock_vector_store() -> MagicMock:
    m = MagicMock()
    m.search.return_value = [
        {"chunk_id": "c1", "score": 0.95, "chunk": make_chunk_record("hello")},
        {"chunk_id": "c2", "score": 0.80, "chunk": make_chunk_record("world")},
    ]
    return m


@pytest.fixture
def mock_knowledge_repo() -> MagicMock:
    m = MagicMock()
    m.get.return_value = None
    return m


@pytest.fixture
def mock_generator() -> MagicMock:
    m = MagicMock()
    m.generate = AsyncMock(return_value=("the answer", [Citation(chunk_id="c1", document_id="d1")]))
    m.record_tokens = MagicMock(return_value={"prompt": 10, "completion": 20, "model": "gpt-4"})
    return m


@pytest.fixture
def mock_reranker() -> MagicMock:
    m = MagicMock()
    m.rerank.side_effect = lambda question, hits: list(reversed(hits))
    return m


@pytest.fixture
def mock_structured() -> MagicMock:
    m = MagicMock()
    m.generate = AsyncMock(return_value={"name": "Acme", "revenue": 100})
    return m


@pytest.fixture
def mock_telemetry() -> MagicMock:
    m = MagicMock()
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=None)
    m.span.return_value = span
    return m


@pytest.fixture
def mock_conversation_store() -> MagicMock:
    m = MagicMock()
    m.load.return_value = [ConversationTurn(question="previous?", answer="previous!")]
    return m


# ---------------------------------------------------------------------------
# chunks_from_knowledge_bundle tests
# ---------------------------------------------------------------------------

class TestChunksFromKnowledgeBundle:
    def test_skips_non_text_blocks(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    blocks=[
                        make_block("b1", "table", "|a|b|"),
                        make_block("b2", "text", "hello"),
                        make_block("b3", "image", "img.png"),
                    ]
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_skips_empty_text_blocks(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    blocks=[
                        make_block("b1", "text", ""),
                        make_block("b2", "text", "  "),
                        make_block("b3", "text", "valid"),
                    ]
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert len(result) == 1
        assert result[0].text == "valid"

    def test_uses_bundle_metadata_company_when_not_provided(self) -> None:
        bundle = make_bundle(
            metadata={"company": "megacorp", "owner": "admin@m.com"},
            sections=[make_section(blocks=[make_block(content="data")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert len(result) == 1
        assert result[0].company == "megacorp"
        assert result[0].owner == "admin@m.com"

    def test_explicit_company_overrides_bundle_metadata(self) -> None:
        bundle = make_bundle(
            metadata={"company": "megacorp"},
            sections=[make_section(blocks=[make_block(content="data")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1", company="othercorp")
        assert result[0].company == "othercorp"

    def test_empty_bundle_yields_no_chunks(self) -> None:
        bundle = make_bundle(sections=[])
        assert chunks_from_knowledge_bundle(bundle, "doc-1") == []

    def test_page_numbers_used_for_page_field(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    index=0,
                    page_numbers=[3],
                    blocks=[make_block(content="page3 text")],
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].page == 3

    def test_section_index_fallback_when_no_page_numbers(self) -> None:
        bundle = make_bundle(
            sections=[
                make_section(
                    index=5,
                    page_numbers=[],
                    blocks=[make_block(content="no page num")],
                )
            ]
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].page == 5

    def test_source_location_fallback_to_bundle_source_uri(self) -> None:
        bundle = make_bundle(
            source_uri="s3://bucket/key",
            sections=[
                make_section(
                    source_location="",
                    blocks=[make_block(content="data")],
                )
            ],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].source_location == "s3://bucket/key"

    def test_tenant_company_empty_when_no_company_and_no_metadata(self) -> None:
        bundle = make_bundle(
            metadata={},
            sections=[make_section(blocks=[make_block(content="data")])],
        )
        result = chunks_from_knowledge_bundle(bundle, "doc-1")
        assert result[0].company == ""


# ---------------------------------------------------------------------------
# sha256_checksum tests
# ---------------------------------------------------------------------------

class TestSha256Checksum:
    def test_returns_hex_string(self) -> None:
        result = sha256_checksum(b"hello")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_deterministic(self) -> None:
        assert sha256_checksum(b"data") == sha256_checksum(b"data")

    def test_different_inputs_yield_different_hashes(self) -> None:
        assert sha256_checksum(b"a") != sha256_checksum(b"b")


# ---------------------------------------------------------------------------
# primary_company tests
# ---------------------------------------------------------------------------

class TestPrimaryCompany:
    def test_none_user_returns_empty(self) -> None:
        assert primary_company(None) == ""

    def test_admin_user_returns_empty(self) -> None:
        user = MagicMock(is_admin=True, allowed_companies=["acme"])
        assert primary_company(user) == ""

    def test_user_with_no_companies_returns_empty(self) -> None:
        user = MagicMock(is_admin=False, allowed_companies=[])
        assert primary_company(user) == ""

    def test_user_with_missing_allowed_companies_returns_empty(self) -> None:
        user = MagicMock(is_admin=False)
        del user.allowed_companies
        assert primary_company(user) == ""

    def test_user_with_allowed_companies_returns_first(self) -> None:
        user = MagicMock(is_admin=False, allowed_companies=["acme", "beta"])
        assert primary_company(user) == "acme"


# ---------------------------------------------------------------------------
# IngestPipeline — __init__
# ---------------------------------------------------------------------------

class TestIngestPipelineInit:
    def test_requires_embedder_and_vector_store(self) -> None:
        with pytest.raises(PipelineError, match="requires embedder and vector_store"):
            IngestPipeline()
        with pytest.raises(PipelineError, match="requires embedder and vector_store"):
            IngestPipeline(embedder=MagicMock())
        with pytest.raises(PipelineError, match="requires embedder and vector_store"):
            IngestPipeline(vector_store=MagicMock())

    def test_sets_defaults(self, mock_embedder: MagicMock, mock_vector_store: MagicMock) -> None:
        pipe = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)
        from raghub.converters.plaintext import PlainTextConverter
        from raghub.ingestion.chunkers.word_window import WordWindowChunker
        from raghub.knowledge.repository import InMemoryKnowledgeRepository
        from raghub.observability.noop import NoOpTelemetry
        assert isinstance(pipe.converter, PlainTextConverter)
        assert isinstance(pipe.chunker, WordWindowChunker)
        assert pipe.embedder is mock_embedder
        assert pipe.vector_store is mock_vector_store
        assert isinstance(pipe.knowledge_repo, InMemoryKnowledgeRepository)
        assert isinstance(pipe.telemetry, NoOpTelemetry)

    def test_accepts_explicit_dependencies(
        self,
        mock_converter: MagicMock,
        mock_chunker: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_knowledge_repo: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        pipe = IngestPipeline(
            converter=mock_converter,
            chunker=mock_chunker,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            knowledge_repo=mock_knowledge_repo,
            telemetry=mock_telemetry,
        )
        assert pipe.converter is mock_converter
        assert pipe.chunker is mock_chunker
        assert pipe.embedder is mock_embedder
        assert pipe.vector_store is mock_vector_store
        assert pipe.knowledge_repo is mock_knowledge_repo
        assert pipe.telemetry is mock_telemetry


# ---------------------------------------------------------------------------
# IngestPipeline — run
# ---------------------------------------------------------------------------

class TestIngestPipelineRun:
    @pytest.fixture
    def pipe(
        self,
        mock_converter: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_knowledge_repo: MagicMock,
        mock_telemetry: MagicMock,
    ) -> IngestPipeline:
        return IngestPipeline(
            converter=mock_converter,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            knowledge_repo=mock_knowledge_repo,
            telemetry=mock_telemetry,
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

    async def test_full_flow(
        self,
        pipe: IngestPipeline,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        result = await pipe.run(pipeline_context, **inputs)
        assert result.success is True
        assert result.pipeline_name == "ingest"
        outputs = result.outputs
        assert outputs["incremental"] is False
        assert outputs["chunk_count"] > 0
        assert outputs["bundle"].bundle_id is not None
        assert outputs["bundle"].checksum is not None
        assert len(outputs["embeddings"]) > 0
        assert pipeline_context.metadata["duration_ms"] > 0

    async def test_incremental_short_circuit(
        self,
        mock_converter: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_knowledge_repo: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        existing_bundle = make_bundle(
            bundle_id="existing-bundle",
            checksum=sha256_checksum(b"pdf content"),
            sections=[
                make_section(blocks=[make_block(content="existing text")])
            ],
        )
        mock_knowledge_repo.get.return_value = existing_bundle

        pipe = IngestPipeline(
            converter=mock_converter,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            knowledge_repo=mock_knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"pdf content",
            source_uri="file:///doc.pdf",
        )
        assert result.success is True
        assert result.outputs["incremental"] is True
        assert result.outputs["bundle"] is existing_bundle
        assert result.outputs["embeddings"] == []
        mock_converter.convert.assert_not_called()

    async def test_force_disables_incremental(
        self,
        mock_converter: MagicMock,
        mock_knowledge_repo: MagicMock,
        pipe: IngestPipeline,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        existing_bundle = make_bundle(checksum=sha256_checksum(b"pdf content"))
        mock_knowledge_repo.get.return_value = existing_bundle
        inputs["force"] = True
        result = await pipe.run(pipeline_context, **inputs)
        assert result.success is True
        assert result.outputs["incremental"] is False
        mock_converter.convert.assert_called_once()

    async def test_error_path(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        pipe = IngestPipeline(embedder=mock_embedder, vector_store=mock_vector_store)
        with patch.object(
            pipe.converter, "convert", side_effect=ValueError("boom")
        ):
            result = await pipe.run(
                pipeline_context,
                file_bytes=b"data",
                source_uri="file:///doc.pdf",
            )
        assert result.success is False
        assert "boom" in result.error

    async def test_missing_required_inputs(
        self,
        pipe: IngestPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        result = await pipe.run(pipeline_context)
        assert result.success is False
        assert "file_bytes" in result.error

    async def test_sets_duration_metadata_on_error(
        self,
        pipe: IngestPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        with patch.object(
            pipe.converter, "convert", side_effect=ValueError("fail")
        ):
            await pipe.run(
                pipeline_context,
                file_bytes=b"data",
                source_uri="file:///doc.pdf",
            )
        assert pipeline_context.metadata["duration_ms"] > 0

    async def test_user_overrides_chunk_owner(
        self,
        pipe: IngestPipeline,
        pipeline_context: PipelineContext,
        inputs: dict[str, Any],
    ) -> None:
        user = UserPrincipal(email="dev@acme.com", allowed_companies=["acme"])
        inputs["user"] = user
        result = await pipe.run(pipeline_context, **inputs)
        for chunk in result.outputs["chunks"]:
            assert chunk.owner == "dev@acme.com"

    async def test_empty_texts_no_embedding(
        self,
        mock_converter: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_knowledge_repo: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        bundle = make_bundle(sections=[])
        mock_converter.convert.return_value = bundle
        pipe = IngestPipeline(
            converter=mock_converter,
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            knowledge_repo=mock_knowledge_repo,
        )
        result = await pipe.run(
            pipeline_context,
            file_bytes=b"data",
            source_uri="file:///doc.pdf",
        )
        assert result.success is True
        assert result.outputs["embeddings"] == []
        mock_embedder.embed_texts.assert_not_called()
        mock_vector_store.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# QueryPipeline — __init__
# ---------------------------------------------------------------------------

class TestQueryPipelineInit:
    def test_requires_dependencies(self) -> None:
        with pytest.raises(TypeError):
            QueryPipeline()

    def test_sets_default_telemetry_and_conversation_store(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
        )
        from raghub.observability.noop import NoOpTelemetry
        assert isinstance(pipe.telemetry, NoOpTelemetry)
        from raghub.conversation.memory import InMemoryConversationStore
        assert isinstance(pipe.conversation_store, InMemoryConversationStore)

    def test_accepts_explicit_dependencies(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        mock_reranker: MagicMock,
        mock_structured: MagicMock,
        mock_telemetry: MagicMock,
        mock_conversation_store: MagicMock,
    ) -> None:
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
            reranker=mock_reranker,
            structured=mock_structured,
            telemetry=mock_telemetry,
            conversation_store=mock_conversation_store,
        )
        assert pipe.embedder is mock_embedder
        assert pipe.vector_store is mock_vector_store
        assert pipe.generator is mock_generator
        assert pipe.reranker is mock_reranker
        assert pipe.structured is mock_structured
        assert pipe.telemetry is mock_telemetry
        assert pipe.conversation_store is mock_conversation_store


# ---------------------------------------------------------------------------
# QueryPipeline — metadata_filter_for_user
# ---------------------------------------------------------------------------

class TestMetadataFilterForUser:
    @pytest.fixture
    def pipe(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
    ) -> QueryPipeline:
        return QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
        )

    def test_none_user_returns_empty(self, pipe: QueryPipeline) -> None:
        assert pipe.metadata_filter_for_user(None) == ""

    def test_admin_returns_empty(self, pipe: QueryPipeline) -> None:
        user = MagicMock(is_admin=True, allowed_companies=["acme"])
        assert pipe.metadata_filter_for_user(user) == ""

    def test_user_with_companies_returns_dict(self, pipe: QueryPipeline) -> None:
        user = MagicMock(is_admin=False, allowed_companies=["acme", "beta"])
        assert pipe.metadata_filter_for_user(user) == {"company": ["acme", "beta"]}

    def test_user_with_empty_companies_returns_empty_list_dict(
        self, pipe: QueryPipeline
    ) -> None:
        user = MagicMock(is_admin=False, allowed_companies=[])
        assert pipe.metadata_filter_for_user(user) == {"company": []}

    def test_user_without_allowed_companies_returns_empty_list_dict(
        self, pipe: QueryPipeline
    ) -> None:
        user = MagicMock(is_admin=False)
        assert pipe.metadata_filter_for_user(user) == {"company": []}


# ---------------------------------------------------------------------------
# QueryPipeline — run
# ---------------------------------------------------------------------------

class TestQueryPipelineRun:
    @pytest.fixture
    def pipe(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        mock_reranker: MagicMock,
        mock_structured: MagicMock,
        mock_telemetry: MagicMock,
        mock_conversation_store: MagicMock,
    ) -> QueryPipeline:
        return QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
            reranker=mock_reranker,
            structured=mock_structured,
            telemetry=mock_telemetry,
            conversation_store=mock_conversation_store,
        )

    async def test_full_flow(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            top_k=5,
        )
        assert result.success is True
        assert result.pipeline_name == "query"
        outputs = result.outputs
        assert outputs["answer"] == "the answer"
        assert len(outputs["citations"]) == 1
        assert len(outputs["hits"]) == 2
        assert pipeline_context.metadata["duration_ms"] > 0

    async def test_with_structured_output(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        class FakeModel:
            pass

        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            response_model=FakeModel,
        )
        assert result.success is True
        assert result.outputs["structured"] == {"name": "Acme", "revenue": 100}

    async def test_with_session_and_record(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        mock_conversation_store: MagicMock,
        mock_generator: MagicMock,
    ) -> None:
        mock_generator.record_tokens = MagicMock(return_value={"prompt": 5, "completion": 10, "model": "gpt-4"})
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            session_id="sess-1",
            record=True,
        )
        assert result.success is True
        assert len(result.outputs["history"]) == 1
        mock_conversation_store.append.assert_called_once()

    async def test_session_load_error_returns_empty_history(
        self,
        mock_conversation_store: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        mock_conversation_store.load.side_effect = RuntimeError("store down")
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
            conversation_store=mock_conversation_store,
        )
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            session_id="sess-1",
        )
        assert result.success is True
        assert result.outputs["history"] == []

    async def test_with_user_rbac(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        mock_vector_store: MagicMock,
    ) -> None:
        user = UserPrincipal(email="admin@acme.com", is_admin=True)
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            user=user,
        )
        assert result.success is True
        _, kwargs = mock_vector_store.search.call_args
        assert kwargs["metadata_filter"] == ""

    async def test_with_additional_metadata_filter(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
            metadata_filter={"company": "acme"},
        )
        assert result.success is True

    async def test_token_recording(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        mock_generator: MagicMock,
        mock_telemetry: MagicMock,
    ) -> None:
        mock_generator.record_tokens = MagicMock(return_value={"prompt": 10, "completion": 20, "model": "gpt-4"})
        pipe.telemetry = mock_telemetry
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
        )
        assert result.success is True
        mock_telemetry.record_tokens.assert_called_once_with(
            "query.generate",
            prompt_tokens=10,
            completion_tokens=20,
            model="gpt-4",
        )

    async def test_no_token_recording_when_not_available(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        del mock_generator.record_tokens
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
        )
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
        )
        assert result.success is True

    async def test_error_path(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        mock_generator.generate = AsyncMock(side_effect=ValueError("gen failed"))
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
        )
        result = await pipe.run(
            pipeline_context,
            question="what is acme?",
        )
        assert result.success is False
        assert "gen failed" in result.error

    async def test_sets_duration_on_error(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        mock_generator.generate = AsyncMock(side_effect=ValueError("fail"))
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
        )
        await pipe.run(pipeline_context, question="what?")
        assert pipeline_context.metadata["duration_ms"] > 0

    async def test_record_skipped_when_no_answer(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_generator: MagicMock,
        mock_conversation_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        mock_generator.generate = AsyncMock(return_value=("", []))
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=mock_generator,
            conversation_store=mock_conversation_store,
        )
        result = await pipe.run(
            pipeline_context,
            question="what?",
            session_id="sess-1",
            record=True,
        )
        assert result.success is True
        mock_conversation_store.append.assert_not_called()


# ---------------------------------------------------------------------------
# QueryPipeline — stream
# ---------------------------------------------------------------------------

class TestQueryPipelineStream:
    @pytest.fixture
    def pipe(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_reranker: MagicMock,
        mock_telemetry: MagicMock,
        mock_conversation_store: MagicMock,
    ) -> QueryPipeline:
        generator = MagicMock()
        generator.astream = _make_astream("hello", " world")
        generator.record_tokens = MagicMock(return_value={"prompt": 3, "completion": 7, "model": "gpt-4"})
        return QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            reranker=mock_reranker,
            telemetry=mock_telemetry,
            conversation_store=mock_conversation_store,
        )

    async def test_basic_stream(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
    ) -> None:
        tokens = []
        async for token in pipe.stream(pipeline_context, question="hi"):
            tokens.append(token)
        assert "".join(tokens) == "hello world"

    async def test_stream_with_session_records_turn(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        mock_conversation_store: MagicMock,
    ) -> None:
        tokens = []
        async for token in pipe.stream(
            pipeline_context, question="hi", session_id="sess-1"
        ):
            tokens.append(token)
        assert "".join(tokens) == "hello world"
        mock_conversation_store.append.assert_called_once()

    async def test_stream_without_session_does_not_record(
        self,
        mock_conversation_store: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        generator = MagicMock()
        generator.astream = _make_astream("only")
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            conversation_store=mock_conversation_store,
        )
        tokens = []
        async for token in pipe.stream(pipeline_context, question="hi"):
            tokens.append(token)
        assert "".join(tokens) == "only"
        mock_conversation_store.append.assert_not_called()

    async def test_stream_with_user_sets_span_attribute(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
        mock_telemetry: MagicMock,
    ) -> None:
        generator = MagicMock()
        generator.astream = _make_astream("ok")
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            telemetry=mock_telemetry,
        )
        user = UserPrincipal(email="u@a.com", is_admin=True)
        tokens = []
        async for token in pipe.stream(
            pipeline_context, question="hi", user=user
        ):
            tokens.append(token)
        # user span attribute should be set
        span = mock_telemetry.span.return_value
        span.set_attribute.assert_any_call("user_id", "u@a.com")

    async def test_stream_with_additional_metadata_filter(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        generator = MagicMock()
        generator.astream = _make_astream("resp")
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
        )
        tokens = []
        async for token in pipe.stream(
            pipeline_context,
            question="hi",
            metadata_filter={"company": "acme"},
        ):
            tokens.append(token)
        assert "".join(tokens) == "resp"

    async def test_stream_session_load_error_returns_empty_history(
        self,
        mock_conversation_store: MagicMock,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        mock_conversation_store.load.side_effect = RuntimeError("fail")
        generator = MagicMock()
        generator.astream = _make_astream("data")
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            conversation_store=mock_conversation_store,
        )
        tokens = []
        async for token in pipe.stream(
            pipeline_context, question="hi", session_id="sess-1"
        ):
            tokens.append(token)
        assert "".join(tokens) == "data"

    async def test_fallback_nonstreaming_generator(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        generator = MagicMock()
        generator.generate = AsyncMock(
            return_value=("hello world", [])
        )
        del generator.astream
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
        )
        tokens = []
        async for token in pipe.stream(pipeline_context, question="hi"):
            tokens.append(token)
        assert "".join(tokens) == "hello world "

    async def test_fallback_with_session_records(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_conversation_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        generator = MagicMock()
        generator.generate = AsyncMock(
            return_value=("answer text", [])
        )
        del generator.astream
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            conversation_store=mock_conversation_store,
        )
        tokens = []
        async for token in pipe.stream(
            pipeline_context, question="hi", session_id="sess-1"
        ):
            tokens.append(token)
        assert "".join(tokens) == "answer text "
        mock_conversation_store.append.assert_called_once()

    async def test_fallback_no_record_when_no_answer(
        self,
        mock_embedder: MagicMock,
        mock_vector_store: MagicMock,
        mock_conversation_store: MagicMock,
        pipeline_context: PipelineContext,
    ) -> None:
        generator = MagicMock()
        generator.generate = AsyncMock(return_value=("", []))
        del generator.astream
        pipe = QueryPipeline(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            generator=generator,
            conversation_store=mock_conversation_store,
        )
        tokens = []
        async for token in pipe.stream(
            pipeline_context, question="hi", session_id="sess-1"
        ):
            tokens.append(token)
        mock_conversation_store.append.assert_not_called()

    async def test_stream_token_recording(
        self,
        pipe: QueryPipeline,
        pipeline_context: PipelineContext,
        mock_telemetry: MagicMock,
    ) -> None:
        pipe.telemetry = mock_telemetry
        tokens = []
        async for token in pipe.stream(pipeline_context, question="hi"):
            tokens.append(token)
        assert "".join(tokens) == "hello world"
        mock_telemetry.record_tokens.assert_called_once()


def _make_astream(*tokens: str):
    """Return an async generator function that yields the given tokens."""
    async def astream_fn(**kwargs):
        for t in tokens:
            yield t
    return astream_fn
