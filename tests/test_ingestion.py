"""Comprehensive tests for ingestion pipeline components.

Covers:

* :class:`ChunkingPlan` — defaults, custom values.
* :func:`normalize_text` — whitespace collapsing.
* :func:`chunk_words` — basic chunking, overlap, edge cases.
* :func:`extract_pdf_pages` — PDF text extraction by page.
* :func:`extract_pdf_text` — (page, location, text) tuples.
* :class:`Ingestor` — service construction + ingest() happy path.
* :class:`IngestionResult` — model fields.
* :class:`Batch` — background submit.
* :class:`WordChunker` — deterministic chunker behavior.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from raghub.embedder import Hasher
from raghub.ingest import (
    Batch,
    IngestionResult,
    Ingestor,
    WordChunker,
)
from raghub.lifecycle import ChunkingPlan, chunk_words, normalize_text


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


# =========================================================================
# documents/chunker.py — ChunkingPlan + helpers
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


class TestExtractPdfPages:
    def test_normal_pdf(self) -> None:
        pdf_bytes = _make_pdf("Hello page one", "Page two content")
        from raghub.lifecycle import extract_pdf_pages

        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert pages[0][1].strip() == "Hello page one"
        assert pages[1][0] == 2
        assert pages[1][1].strip() == "Page two content"

    def test_single_page(self) -> None:
        from raghub.lifecycle import extract_pdf_pages

        pdf_bytes = _make_pdf("Only one page")
        pages = extract_pdf_pages(pdf_bytes)
        assert len(pages) == 1
        assert pages[0][0] == 1

    def test_page_numbers_are_1_based(self) -> None:
        from raghub.lifecycle import extract_pdf_pages

        pdf_bytes = _make_pdf("A", "B", "C")
        pages = extract_pdf_pages(pdf_bytes)
        assert [p[0] for p in pages] == [1, 2, 3]


class TestExtractPdfText:
    def test_returns_page_location_and_text(self) -> None:
        from raghub.lifecycle import extract_pdf_text

        pdf_bytes = _make_pdf("Hello", "World")
        result = extract_pdf_text(pdf_bytes)
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[0][1] == "page 1"
        assert "Hello" in result[0][2]


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

    def test_overlap_equals_chunk_size_makes_progress(self) -> None:
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
        chunks = chunk_words(text, plan)
        assert chunks == ["hello world foo bar"]


# =========================================================================
# WordChunker — deterministic chunker contract
# =========================================================================


class TestWordChunker:
    def test_word_chunker_rejects_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            WordChunker(chunk_size=4, chunk_overlap=4)

    def test_word_chunker_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            WordChunker(chunk_size=0, chunk_overlap=0)


# =========================================================================
# Ingestor + IngestionResult — service layer
# =========================================================================


class TestIngestionResult:
    def test_required_fields(self) -> None:
        from raghub.models import DocumentRecord

        doc = DocumentRecord(
            document_id="d1",
            version=1,
            checksum="abc",
            owner="u@x.com",
            organization="acme",
        )
        result = IngestionResult(
            document=doc,
            chunk_ids=["c1", "c2"],
        )
        assert result.document == doc
        assert result.chunk_ids == ["c1", "c2"]


class TestIngestorInit:
    def test_build_pipeline_returns_ingest_pipeline(self) -> None:
        from raghub.pipeline import IngestPipeline

        mock_uow = MagicMock()
        mock_uow.vector_store = MagicMock()
        embedder = Hasher(dimension=4, model_name="test")
        ingestor = Ingestor(
            uow=mock_uow,
            embedding_provider=embedder,
            lifecycle_manager=MagicMock(),
            max_upload_bytes=10_000,
        )
        pipeline = ingestor.build_pipeline()
        assert isinstance(pipeline, IngestPipeline)
        assert pipeline.embedder is embedder


# =========================================================================
# Batch — background ingestion
# =========================================================================


class TestBatch:
    def test_submit_returns_job_id(self) -> None:
        batch = Batch(max_workers=1)
        try:

            def noop():
                return None

            job_id = batch.submit(noop)
            assert isinstance(job_id, str)
            assert len(job_id) > 0
            import time

            time.sleep(0.1)
            assert batch.get_status(job_id) == "completed"
        finally:
            batch.shutdown()

    def test_submit_async_runs_callable(self) -> None:
        batch = Batch(max_workers=1)
        try:

            def add(a: int, b: int) -> int:
                return a + b

            job_id = batch.submit(add, 2, 3)
            import time

            time.sleep(0.1)
            status = batch.get_status(job_id)
            assert status == "completed"
        finally:
            batch.shutdown()

    def test_batch_shutdown_is_idempotent(self) -> None:
        batch = Batch(max_workers=1)
        batch.shutdown()
        batch.shutdown()

    def test_submit_after_shutdown_raises(self) -> None:
        batch = Batch(max_workers=1)
        batch.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            batch.submit(lambda: None)
