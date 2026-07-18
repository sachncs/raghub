"""Comprehensive tests for ingestion pipeline components.

Covers:
- documents/chunker.py  : ChunkingPlan, extraction, chunk_words, build_chunk_records
- ingestion/service.py  : DocumentIngestionService, ingest flow, error handling
- ingestion/chunkers/chonkie.py : ChonkieChunker, build_chonkie_inner, factory
- ingestion/__init__.py : lazy module-level exports
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.documents.chunker import (
    ChunkingPlan,
    build_chunk_records,
    chunk_words,
    extract_pdf_metadata,
    extract_pdf_pages,
    extract_pdf_text,
    extract_text_from_content,
    normalize_text,
)
from raghub.ingestion import DocumentIngestionService, IngestionResult
from raghub.ingestion.chunkers.chonkie import (
    CHONKIE_AVAILABLE,
    ChonkieChunker,
    build_chonkie_chunker,
    build_chonkie_inner,
)
from raghub.models import (
    Classification,
    DocumentLifecycleStatus,
    DocumentRecord,
    UserPrincipal,
)

# =========================================================================
# documents/chunker.py
# =========================================================================


class TestChunkingPlan:
    def test_defaults(self) -> None:
        plan = ChunkingPlan()
        assert plan.chunk_size_words == 800
        assert plan.overlap_words == 100

    def test_custom_values(self) -> None:
        plan = ChunkingPlan(chunk_size_words=200, overlap_words=50)
        assert plan.chunk_size_words == 200
        assert plan.overlap_words == 50

    def test_frozen(self) -> None:
        plan = ChunkingPlan()
        with pytest.raises(AttributeError):
            plan.chunk_size_words = 999  # type: ignore[misc]


class TestExtractPdfPages:
    def test_normal_pdf(self) -> None:
        pdf_bytes = _make_pdf("Hello page one", "Page two content")
        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert pages[0][1].strip() == "Hello page one"
        assert pages[1][0] == 2
        assert pages[1][1].strip() == "Page two content"

    def test_single_page(self) -> None:
        pdf_bytes = _make_pdf("Only one page")
        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 1
        assert pages[0][0] == 1

    def test_empty_pdf(self) -> None:
        pdf_bytes = _make_pdf("")
        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 1
        assert pages[0] == (1, "")

    def test_page_numbers_are_1_based(self) -> None:
        pdf_bytes = _make_pdf("A", "B", "C")
        pages = extract_pdf_pages(pdf_bytes)
        assert [p[0] for p in pages] == [1, 2, 3]


class TestNormalizeText:
    def test_collapses_spaces(self) -> None:
        assert normalize_text("hello   world") == "hello world"

    def test_collapses_newlines_and_tabs(self) -> None:
        assert normalize_text("line1\n\tline2") == "line1 line2"

    def test_strips_whitespace(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_text("   \n\t  ") == ""


class TestChunkWords:
    def test_basic_chunking(self) -> None:
        text = "one two three four five six seven eight nine ten"
        plan = ChunkingPlan(chunk_size_words=4, overlap_words=1)
        chunks = chunk_words(text, plan)
        assert chunks == [
            "one two three four",
            "four five six seven",
            "seven eight nine ten",
        ]

    def test_no_overlap(self) -> None:
        text = "one two three four five six"
        plan = ChunkingPlan(chunk_size_words=3, overlap_words=0)
        chunks = chunk_words(text, plan)
        assert chunks == ["one two three", "four five six"]

    def test_overlap_equals_chunk_size_guarantees_progress(self) -> None:
        text = "a b c d e f"
        plan = ChunkingPlan(chunk_size_words=3, overlap_words=3)
        chunks = chunk_words(text, plan)
        assert chunks
        assert len(chunks) >= 1

    def test_short_text_no_chunks(self) -> None:
        plan = ChunkingPlan(chunk_size_words=10, overlap_words=2)
        assert chunk_words("", plan) == []

    def test_text_shorter_than_chunk_size(self) -> None:
        text = "hello world"
        plan = ChunkingPlan(chunk_size_words=10, overlap_words=2)
        chunks = chunk_words(text, plan)
        assert chunks == ["hello world"]

    def test_exact_chunk_size(self) -> None:
        text = "one two three"
        plan = ChunkingPlan(chunk_size_words=3, overlap_words=0)
        assert chunk_words(text, plan) == ["one two three"]

    def test_normalization_applied(self) -> None:
        text = "hello   world\nfoo\tbar"
        plan = ChunkingPlan(chunk_size_words=4, overlap_words=0)
        assert chunk_words(text, plan) == ["hello world foo bar"]

    def test_single_word(self) -> None:
        plan = ChunkingPlan(chunk_size_words=5, overlap_words=1)
        assert chunk_words("hello", plan) == ["hello"]

    def test_large_overlap_edge(self) -> None:
        text = "a b c d e f g h i j"
        plan = ChunkingPlan(chunk_size_words=5, overlap_words=10)
        chunks = chunk_words(text, plan)
        assert len(chunks) >= 2


class TestExtractPdfText:
    def test_returns_tuples(self) -> None:
        pdf_bytes = _make_pdf("Hello", "World")
        result = extract_pdf_text(pdf_bytes)
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[0][1] == "page 1"
        assert result[0][2].strip() == "Hello"
        assert result[1][0] == 2
        assert result[1][1] == "page 2"
        assert result[1][2].strip() == "World"


class TestExtractTextFromContent:
    def test_pdf_by_mime(self) -> None:
        pdf_bytes = _make_pdf("PDF content")
        result = extract_text_from_content(pdf_bytes, "doc.pdf", "application/pdf")
        assert len(result) == 1
        assert result[0][2].strip() == "PDF content"

    def test_pdf_by_extension(self) -> None:
        pdf_bytes = _make_pdf("PDF ext")
        result = extract_text_from_content(pdf_bytes, "doc.pdf", "application/octet-stream")
        assert len(result) == 1
        assert result[0][2].strip() == "PDF ext"

    def test_plain_text(self) -> None:
        result = extract_text_from_content(b"hello world", "readme.txt", "text/plain")
        assert result == [(0, "full file", "hello world")]

    def test_csv(self) -> None:
        result = extract_text_from_content(b"a,b,c\n1,2,3", "data.csv", "text/csv")
        assert result == [(0, "full file", "a,b,c\n1,2,3")]

    def test_image(self) -> None:
        result = extract_text_from_content(b"fake-image-data", "photo.png", "image/png")
        assert result == [(0, "image", "fake-image-data")]

    def test_docx_mime(self) -> None:
        result = extract_text_from_content(
            b"docx content",
            "report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert result == [(0, "document", "docx content")]

    def test_doc_mime(self) -> None:
        result = extract_text_from_content(b"doc content", "report.doc", "application/msword")
        assert result == [(0, "document", "doc content")]

    def test_xlsx_mime(self) -> None:
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        result = extract_text_from_content(b"sheet data", "data.xlsx", mime)
        assert result == [(0, "spreadsheet", "sheet data")]

    def test_xls_mime(self) -> None:
        result = extract_text_from_content(b"old sheet", "data.xls", "application/vnd.ms-excel")
        assert result == [(0, "spreadsheet", "old sheet")]

    def test_pptx_mime(self) -> None:
        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        result = extract_text_from_content(b"slide data", "deck.pptx", mime)
        assert result == [(0, "presentation", "slide data")]

    def test_ppt_mime(self) -> None:
        result = extract_text_from_content(
            b"old slides", "deck.ppt", "application/vnd.ms-powerpoint"
        )
        assert result == [(0, "presentation", "old slides")]

    def test_unknown_mime(self) -> None:
        result = extract_text_from_content(b"binary data", "file.bin", "application/octet-stream")
        assert result == [(0, "unknown", "binary data")]

    def test_utf8_decode_fallback(self) -> None:
        result = extract_text_from_content("héllo wörld".encode(), "notes.txt", "text/plain")
        assert result == [(0, "full file", "héllo wörld")]


class TestExtractPdfMetadata:
    def test_returns_metadata(self) -> None:
        pdf_bytes = _make_pdf("content")
        meta = extract_pdf_metadata(pdf_bytes)
        assert isinstance(meta, dict)
        for key in ("title", "author", "producer", "creator"):
            assert key in meta

    def test_malformed_pdf_returns_empty(self) -> None:
        meta = extract_pdf_metadata(b"not a pdf at all")
        assert meta == {}

    def test_empty_pdf_returns_defaults(self) -> None:
        pdf_bytes = _make_pdf("")
        meta = extract_pdf_metadata(pdf_bytes)
        assert isinstance(meta.get("title"), str)


class TestBuildChunkRecords:
    def test_non_pdf_creates_chunks(self) -> None:
        records = build_chunk_records(
            file_bytes=b"hello world foo bar baz",
            document_id="doc-1",
            version=1,
            company="Acme",
            owner="alice@acme.com",
            department="eng",
            classification=Classification.INTERNAL,
            embedding_model="text-embedding-3-small",
            plan=ChunkingPlan(chunk_size_words=3, overlap_words=0),
            mime_type="text/plain",
            file_name="notes.txt",
        )
        assert len(records) == 2
        for r in records:
            assert r.document_id == "doc-1"
            assert r.company == "Acme"
            assert r.owner == "alice@acme.com"
            assert r.department == "eng"
            assert r.classification == Classification.INTERNAL
            assert r.embedding_model == "text-embedding-3-small"
            assert r.version == 1
            assert r.metadata == {}

    def test_pdf_includes_metadata(self) -> None:
        pdf_bytes = _make_pdf("hello world")
        records = build_chunk_records(
            file_bytes=pdf_bytes,
            document_id="doc-pdf",
            version=1,
            company="Acme",
            owner="bob@acme.com",
            department="",
            classification=Classification.CONFIDENTIAL,
            embedding_model="test-model",
            plan=ChunkingPlan(chunk_size_words=10, overlap_words=0),
            mime_type="application/pdf",
            file_name="doc.pdf",
        )
        assert len(records) == 1
        assert records[0].document_id == "doc-pdf"
        # PDF metadata should be present (keys like title, author)
        assert isinstance(records[0].metadata, dict)

    def test_empty_file_returns_no_chunks(self) -> None:
        records = build_chunk_records(
            file_bytes=b"",
            document_id="doc-empty",
            version=1,
            company="A",
            owner="o@o.com",
            department="",
            classification=Classification.INTERNAL,
            embedding_model="m",
            plan=ChunkingPlan(chunk_size_words=10, overlap_words=0),
            mime_type="text/plain",
            file_name="empty.txt",
        )
        assert records == []

    def test_chunks_have_hashes(self) -> None:
        records = build_chunk_records(
            file_bytes=b"hello world foo bar baz",
            document_id="doc-1",
            version=1,
            company="Acme",
            owner="a@a.com",
            department="",
            classification=Classification.INTERNAL,
            embedding_model="m",
            plan=ChunkingPlan(chunk_size_words=3, overlap_words=0),
            mime_type="text/plain",
            file_name="notes.txt",
        )
        for r in records:
            expected_hash = hashlib.sha256(r.text.encode()).hexdigest()
            assert r.hash == expected_hash
            assert r.chunk_id is not None


# =========================================================================
# ingestion/service.py
# =========================================================================


@pytest.fixture
def mock_uow() -> MagicMock:
    uow = MagicMock()
    uow.document_repo = AsyncMock()
    uow.chunk_repo = AsyncMock()
    return uow


@pytest.fixture
def mock_embedding_provider() -> MagicMock:
    provider = MagicMock()
    provider.model_name = "test-model"
    provider.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
    return provider


@pytest.fixture
def mock_lifecycle() -> MagicMock:
    return MagicMock()


@pytest.fixture
def owner() -> UserPrincipal:
    return UserPrincipal(email="alice@acme.com")


@pytest.fixture
def service(
    mock_uow: MagicMock,
    mock_embedding_provider: MagicMock,
    mock_lifecycle: MagicMock,
) -> DocumentIngestionService:
    return DocumentIngestionService(
        uow=mock_uow,
        embedding_provider=mock_embedding_provider,
        lifecycle_manager=mock_lifecycle,
        plan=ChunkingPlan(chunk_size_words=800, overlap_words=100),
        max_upload_bytes=10_000_000,
        virus_scan_hook=None,
    )


class TestDocumentIngestionServiceInit:
    def test_default_virus_scan_hook(self) -> None:
        svc = DocumentIngestionService(
            uow=MagicMock(),
            embedding_provider=MagicMock(),
            lifecycle_manager=MagicMock(),
            plan=ChunkingPlan(),
            max_upload_bytes=1000,
        )
        assert callable(svc.virus_scan_hook)
        # no-op should not raise
        svc.virus_scan_hook(b"data")

    def test_custom_virus_scan_hook(self) -> None:
        hook = MagicMock()
        svc = DocumentIngestionService(
            uow=MagicMock(),
            embedding_provider=MagicMock(),
            lifecycle_manager=MagicMock(),
            plan=ChunkingPlan(),
            max_upload_bytes=1000,
            virus_scan_hook=hook,
        )
        svc.virus_scan_hook(b"data")
        hook.assert_called_once_with(b"data")


class TestSubmitAsync:
    def test_submit_without_background_service(
        self, service: DocumentIngestionService, owner: UserPrincipal
    ) -> None:
        job_id = service.submit_async(
            file_name="test.txt",
            file_bytes=b"hello",
            owner=owner,
            organization="Acme",
        )
        assert isinstance(job_id, str)

    def test_submit_with_background_service(
        self, service: DocumentIngestionService, owner: UserPrincipal
    ) -> None:
        bg = MagicMock()
        bg.submit.return_value = "bg-job-1"
        job_id = service.submit_async(
            file_name="test.txt",
            file_bytes=b"hello",
            owner=owner,
            organization="Acme",
            background_service=bg,
        )
        assert job_id == "bg-job-1"
        bg.submit.assert_called_once()


class TestIngest:
    @patch("raghub.documents.validation.validate_upload")
    async def test_happy_path(
        self,
        mock_validate: MagicMock,
        service: DocumentIngestionService,
        mock_uow: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_lifecycle: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        mock_uow.document_repo.get_by_checksum.return_value = None
        mock_validate.return_value = "text/plain"
        # Wire a stub pipeline that always succeeds; the wrapper now
        # delegates every real piece of work to the new IngestPipeline.
        from raghub.models import PipelineResult

        async def fake_run(_context: object, **_kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=True,
                outputs={
                    "bundle": None,
                    "chunks": [],
                    "chunk_count": 0,
                    "document_id": "doc-new",
                    "version": 1,
                    "incremental": False,
                },
            )

        service._pipeline = MagicMock()
        service._pipeline.run = fake_run  # type: ignore[assignment]
        text = "hello world foo bar baz"
        result = await service.ingest(
            file_name="notes.txt",
            file_bytes=text.encode(),
            owner=owner,
            organization="Acme",
        )
        assert isinstance(result, IngestionResult)
        assert result.document is not None
        assert result.chunk_ids is not None
        # Document is persisted at the wrapper level after a successful
        # pipeline run.
        mock_uow.document_repo.save.assert_called()

    @patch("raghub.documents.validation.validate_upload")
    async def test_dedup_ready_document_short_circuits(
        self,
        mock_validate: MagicMock,
        service: DocumentIngestionService,
        mock_uow: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        existing = DocumentRecord(
            checksum="abc",
            owner="alice@acme.com",
            organization="Acme",
            status=DocumentLifecycleStatus.READY,
            chunk_ids=["c1", "c2"],
        )
        mock_uow.document_repo.get_by_checksum.return_value = existing
        text = "hello world"
        result = await service.ingest(
            file_name="notes.txt",
            file_bytes=text.encode(),
            owner=owner,
            organization="Acme",
        )
        assert result.document is existing
        assert result.chunk_ids == ["c1", "c2"]
        # No further processing
        mock_uow.document_repo.save.assert_not_called()
        # Pipeline was not invoked on a dedup short-circuit.
        assert service._pipeline is None or service._pipeline.run.call_count == 0  # type: ignore[union-attr]

    @patch("raghub.documents.validation.validate_upload")
    async def test_dedup_non_ready_creates_new_version(
        self,
        mock_validate: MagicMock,
        service: DocumentIngestionService,
        mock_uow: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_lifecycle: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        from raghub.models import PipelineResult

        existing = DocumentRecord(
            checksum="abc",
            owner="alice@acme.com",
            organization="Acme",
            status=DocumentLifecycleStatus.FAILED,
        )
        mock_uow.document_repo.get_by_checksum.return_value = existing
        mock_validate.return_value = "text/plain"

        async def fake_run(_context: object, **_kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=True,
                outputs={
                    "bundle": None,
                    "chunks": [],
                    "chunk_count": 0,
                    "document_id": "doc-different",
                    "version": 2,
                    "incremental": False,
                },
            )

        service._pipeline = MagicMock()
        service._pipeline.run = fake_run  # type: ignore[assignment]
        text = "hello world foo bar baz"
        result = await service.ingest(
            file_name="notes.txt",
            file_bytes=text.encode(),
            owner=owner,
            organization="Acme",
        )
        assert result.document is not existing
        assert result.document.version == 2

    @patch("raghub.documents.validation.validate_upload")
    async def test_validation_failure_propagates(
        self,
        mock_validate: MagicMock,
        service: DocumentIngestionService,
        owner: UserPrincipal,
    ) -> None:
        from raghub.exceptions import DocumentError

        mock_validate.side_effect = DocumentError("Bad file")
        with pytest.raises(DocumentError, match="Bad file"):
            await service.ingest(
                file_name="bad.txt",
                file_bytes=b"x",
                owner=owner,
                organization="Acme",
            )

    @patch("raghub.documents.validation.validate_upload")
    async def test_virus_scan_rejection(
        self,
        mock_validate: MagicMock,
        mock_uow: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_lifecycle: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        def rejecting_hook(_: bytes) -> None:
            raise RuntimeError("Virus detected!")

        svc = DocumentIngestionService(
            uow=mock_uow,
            embedding_provider=mock_embedding_provider,
            lifecycle_manager=mock_lifecycle,
            plan=ChunkingPlan(),
            max_upload_bytes=10_000_000,
            virus_scan_hook=rejecting_hook,
        )
        mock_uow.document_repo.get_by_checksum.return_value = None

        with pytest.raises(RuntimeError, match="Virus detected!"):
            await svc.ingest(
                file_name="safe.txt",
                file_bytes=b"clean data here hello world foo bar",
                owner=owner,
                organization="Acme",
            )

    @patch("raghub.documents.validation.validate_upload")
    async def test_pipeline_failure_propagates_as_document_error(
        self,
        mock_validate: MagicMock,
        mock_uow: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_lifecycle: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        from raghub.exceptions import DocumentError
        from raghub.models import PipelineResult

        mock_uow.document_repo.get_by_checksum.return_value = None
        mock_validate.return_value = "text/plain"

        async def fake_run(_context: object, **_kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=False,
                error="Illegal transition",
            )

        svc = DocumentIngestionService(
            uow=mock_uow,
            embedding_provider=mock_embedding_provider,
            lifecycle_manager=mock_lifecycle,
            plan=ChunkingPlan(),
            max_upload_bytes=10_000_000,
        )
        svc._pipeline = MagicMock()
        svc._pipeline.run = fake_run  # type: ignore[assignment]
        with pytest.raises(DocumentError, match="Illegal transition"):
            await svc.ingest(
                file_name="test.txt",
                file_bytes=b"hello world foo bar",
                owner=owner,
                organization="Acme",
            )

    @patch("raghub.documents.validation.validate_upload")
    async def test_pipeline_failure_with_prior_failed_record_persists_status(
        self,
        mock_validate: MagicMock,
        mock_uow: MagicMock,
        mock_embedding_provider: MagicMock,
        mock_lifecycle: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        from raghub.exceptions import DocumentError
        from raghub.models import PipelineResult

        existing = DocumentRecord(
            checksum="abc",
            owner="alice@acme.com",
            organization="Acme",
            status=DocumentLifecycleStatus.FAILED,
        )
        mock_uow.document_repo.get_by_checksum.return_value = existing
        mock_validate.return_value = "text/plain"

        async def fake_run(_context: object, **_kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=False,
                error="Embedding failed",
            )

        svc = DocumentIngestionService(
            uow=mock_uow,
            embedding_provider=mock_embedding_provider,
            lifecycle_manager=mock_lifecycle,
            plan=ChunkingPlan(),
            max_upload_bytes=10_000_000,
        )
        svc._pipeline = MagicMock()
        svc._pipeline.run = fake_run  # type: ignore[assignment]
        with pytest.raises(DocumentError, match="Embedding failed"):
            await svc.ingest(
                file_name="test.txt",
                file_bytes=b"hello world foo bar",
                owner=owner,
                organization="Acme",
            )
        # Wrapper persists the failure on the prior record so callers
        # polling ``document_status`` see the latest error.
        assert mock_uow.document_repo.save.call_count >= 1
        last_saved = mock_uow.document_repo.save.call_args[0][0]
        assert last_saved.status == DocumentLifecycleStatus.FAILED
        assert "Embedding failed" in (last_saved.error or "")

    @patch("raghub.documents.validation.validate_upload")
    async def test_ingestion_result_fields(
        self,
        mock_validate: MagicMock,
        service: DocumentIngestionService,
        mock_uow: MagicMock,
        owner: UserPrincipal,
    ) -> None:
        from raghub.models import PipelineResult

        mock_uow.document_repo.get_by_checksum.return_value = None
        mock_validate.return_value = "text/plain"

        async def fake_run(_context: object, **_kwargs: object) -> PipelineResult:
            return PipelineResult(
                pipeline_id="i",
                pipeline_name="ingest",
                success=True,
                outputs={
                    "bundle": None,
                    "chunks": [],
                    "chunk_count": 0,
                    "document_id": "doc-1",
                    "version": 1,
                    "incremental": False,
                },
            )

        service._pipeline = MagicMock()
        service._pipeline.run = fake_run  # type: ignore[assignment]
        text = "hello world foo bar baz"
        result = await service.ingest(
            file_name="notes.txt",
            file_bytes=text.encode(),
            owner=owner,
            organization="Acme",
            department="eng",
            tags=["tag1"],
            classification=Classification.CONFIDENTIAL,
        )
        assert isinstance(result, IngestionResult)
        assert hasattr(result, "document")
        assert hasattr(result, "chunk_ids")
        assert isinstance(result.document, DocumentRecord)
        assert result.document.classification == Classification.CONFIDENTIAL


# =========================================================================
# ingestion/chunkers/chonkie.py
# =========================================================================


class TestBuildChonkieInner:
    def test_raises_when_chonkie_unavailable(self) -> None:
        with (
            patch("raghub.ingestion.chunkers.chonkie.CHONKIE_AVAILABLE", False),
            patch("raghub.ingestion.chunkers.chonkie.CHONKIE_MODULE", None),
        ):
            from raghub.exceptions import ConfigurationError

            with pytest.raises(ConfigurationError, match="not installed"):
                build_chonkie_inner(chunk_size=10, chunk_overlap=2, tokenizer="character")


class TestChonkieChunker:
    def test_init_raises_when_chonkie_unavailable(self) -> None:
        with patch("raghub.ingestion.chunkers.chonkie.CHONKIE_AVAILABLE", False):
            from raghub.exceptions import ConfigurationError

            with pytest.raises(ConfigurationError, match="not installed"):
                ChonkieChunker(chunk_size=10, chunk_overlap=2)

    def test_chonkie_text_chunks_type_error_fallback(self) -> None:
        """When inner(text) raises TypeError, fallback to .chunk()"""
        inner = MagicMock()
        inner.side_effect = TypeError("not callable")
        inner.chunk = MagicMock(return_value=["piece1", "piece2"])

        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = inner

        result = chunker.chonkie_text_chunks("some text")
        assert result == ["piece1", "piece2"]
        inner.chunk.assert_called_once_with("some text")

    def test_chonkie_text_chunks_split_text_fallback(self) -> None:
        inner = MagicMock()
        inner.side_effect = TypeError("not callable")
        inner.chunk = None
        inner.split_text = MagicMock(return_value=["a", "b"])

        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = inner

        result = chunker.chonkie_text_chunks("some text")
        assert result == ["a", "b"]
        inner.split_text.assert_called_once_with("some text")

    def test_chonkie_text_chunks_no_fallback_raises(self) -> None:
        inner = MagicMock()
        inner.side_effect = TypeError("not callable")
        inner.chunk = None
        inner.split_text = None

        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = inner

        with pytest.raises(TypeError):
            chunker.chonkie_text_chunks("some text")

    def test_chunk_skips_non_text_blocks(self) -> None:
        from raghub.models import BlockKind, DocumentBlock, DocumentSection, KnowledgeBundle

        inner = MagicMock(return_value=[MagicMock(text="chunk", id="c1")])
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = inner
        chunker.chonkie_text_chunks = MagicMock(return_value=[MagicMock(text="chunk", id="c1")])

        bundle = KnowledgeBundle(
            source_uri="file://x.txt",
            sections=[
                DocumentSection(
                    index=0,
                    blocks=[
                        DocumentBlock(kind=BlockKind.IMAGE, content="fig.png"),
                    ],
                )
            ],
        )
        result = chunker.chunk(bundle)
        assert result == []

    def test_chunk_handles_missing_text_attr(self) -> None:
        from raghub.models import BlockKind, DocumentBlock, DocumentSection, KnowledgeBundle

        class _StrFallback:
            def __str__(self) -> str:
                return "fallback text"

        piece = _StrFallback()
        inner = MagicMock(return_value=[piece])
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = inner
        chunker.chonkie_text_chunks = MagicMock(return_value=[piece])

        bundle = KnowledgeBundle(
            source_uri="file://x.txt",
            sections=[
                DocumentSection(
                    index=0,
                    blocks=[
                        DocumentBlock(kind=BlockKind.TEXT, content="hello world"),
                    ],
                )
            ],
        )
        result = chunker.chunk(bundle)
        assert len(result) == 1
        assert result[0].text == "fallback text"

    def test_chunk_handles_dict_pieces(self) -> None:
        from raghub.models import BlockKind, DocumentBlock, DocumentSection, KnowledgeBundle

        piece = {"text": "hello from dict", "id": "dict-id"}
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = None
        chunker.chonkie_text_chunks = MagicMock(return_value=[piece])

        bundle = KnowledgeBundle(
            source_uri="file://x.txt",
            sections=[
                DocumentSection(
                    index=0,
                    blocks=[
                        DocumentBlock(kind=BlockKind.TEXT, content="hello world"),
                    ],
                )
            ],
        )
        result = chunker.chunk(bundle)
        assert len(result) == 1
        assert result[0].text == "hello from dict"
        assert result[0].chunk_id == "dict-id"

    def test_chunk_text_handles_dict_pieces(self) -> None:
        piece = {"text": "dict text", "id": "d1"}
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = None
        chunker.chonkie_text_chunks = MagicMock(return_value=[piece])

        result = chunker.chunk_text("hello", document_id="doc1")
        assert len(result) == 1
        assert result[0].text == "dict text"
        assert result[0].chunk_id == "d1"

    def test_chunk_text_handles_missing_text_attr(self) -> None:
        class _StrOnly:
            def __str__(self) -> str:
                return "fallback"

        piece = _StrOnly()
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = None
        chunker.chonkie_text_chunks = MagicMock(return_value=[piece])

        result = chunker.chunk_text("hello", document_id="doc1")
        assert len(result) == 1
        assert result[0].text == "fallback"

    def test_chunk_text_generates_fallback_id(self) -> None:
        class _StrNoId:
            def __str__(self) -> str:
                return "text content"

        piece = _StrNoId()
        chunker = ChonkieChunker.__new__(ChonkieChunker)
        chunker.chunk_size = 10
        chunker.chunk_overlap = 2
        chunker.inner = None
        chunker.chonkie_text_chunks = MagicMock(return_value=[piece])

        result = chunker.chunk_text("hello", document_id="doc1", version=2)
        assert len(result) == 1
        assert result[0].chunk_id == "doc1:v2:0"


class TestBuildChonkieChunker:
    def test_auto_prefers_chonkie(self) -> None:
        if not CHONKIE_AVAILABLE:
            pytest.skip("chonkie not installed")
        chunker = build_chonkie_chunker("auto")
        assert isinstance(chunker, ChonkieChunker)

    def test_explicit_chonkie(self) -> None:
        if not CHONKIE_AVAILABLE:
            pytest.skip("chonkie not installed")
        chunker = build_chonkie_chunker("chonkie")
        assert isinstance(chunker, ChonkieChunker)

    def test_explicit_chonkie_unavailable(self) -> None:
        with patch("raghub.ingestion.chunkers.chonkie.CHONKIE_AVAILABLE", False):
            from raghub.exceptions import ConfigurationError

            with pytest.raises(ConfigurationError, match="not installed"):
                build_chonkie_chunker("chonkie")

    def test_auto_falls_back_to_word_window(self) -> None:
        with patch("raghub.ingestion.chunkers.chonkie.CHONKIE_AVAILABLE", False):
            from raghub.ingestion.chunkers.word_window import WordWindowChunker

            chunker = build_chonkie_chunker("auto")
            assert isinstance(chunker, WordWindowChunker)

    def test_explicit_word_window(self) -> None:
        from raghub.ingestion.chunkers.word_window import WordWindowChunker

        chunker = build_chonkie_chunker("word_window")
        assert isinstance(chunker, WordWindowChunker)

    def test_unknown_name_raises(self) -> None:
        from raghub.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown chunker"):
            build_chonkie_chunker("nonexistent")

    def test_kwargs_forwarded_to_chunker(self) -> None:
        if not CHONKIE_AVAILABLE:
            pytest.skip("chonkie not installed")
        chunker = build_chonkie_chunker("auto", chunk_size=100, chunk_overlap=10)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 10


# =========================================================================
# ingestion/__init__.py
# =========================================================================


class TestIngestionInit:
    def test_lazy_import_service(self) -> None:
        from raghub.ingestion import DocumentIngestionService

        assert DocumentIngestionService is not None

    def test_lazy_import_result(self) -> None:
        from raghub.ingestion import IngestionResult

        assert IngestionResult is not None

    def test_getattr_valid_names(self) -> None:
        from raghub.ingestion import __getattr__

        svc = __getattr__("DocumentIngestionService")
        assert svc is DocumentIngestionService
        res = __getattr__("IngestionResult")
        assert res is IngestionResult

    def test_getattr_invalid_name_raises(self) -> None:
        from raghub.ingestion import __getattr__

        with pytest.raises(AttributeError, match="has no attribute"):
            __getattr__("NonExistent")

    def test__all__export(self) -> None:
        from raghub.ingestion import __all__

        assert "DocumentIngestionService" in __all__
        assert "IngestionResult" in __all__


# =========================================================================
# Helpers
# =========================================================================


def _make_pdf(*page_texts: str) -> bytes:
    """Create a minimal PDF in memory with one or more pages."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in page_texts:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def service_factory(
    mock_uow: MagicMock,
    mock_embedding_provider: MagicMock,
    mock_lifecycle: MagicMock,
) -> DocumentIngestionService:
    """Create a basic service from mocks."""
    return DocumentIngestionService(
        uow=mock_uow,
        embedding_provider=mock_embedding_provider,
        lifecycle_manager=mock_lifecycle,
        plan=ChunkingPlan(chunk_size_words=800, overlap_words=100),
        max_upload_bytes=10_000_000,
        virus_scan_hook=None,
    )
