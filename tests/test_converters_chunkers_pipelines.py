"""Tests for the converter + chunker + ingestion pipelines."""

from __future__ import annotations

import asyncio

import pytest

from raghub.converters.markdown import normalise_markdown
from raghub.converters.plaintext import PlainTextConverter
from raghub.embeddings.hashing import HashingEmbeddingProvider
from raghub.ingestion.chunkers.word_window import WordWindowChunker
from raghub.knowledge.repository import InMemoryKnowledgeRepository
from raghub.models import (
    BlockKind,
    DocumentBlock,
    KnowledgeBundle,
    PipelineContext,
)
from raghub.pipelines.rag import IngestPipeline, QueryPipeline
from raghub.generation.generator import DefaultGenerator
from raghub.llm.heuristic import HeuristicLLMProvider
from raghub.vectorstore.memory import InMemoryVectorStore


def test_normalise_markdown_extracts_tables() -> None:
    """Markdown tables become TABLE blocks."""
    md = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    bundle = normalise_markdown(md, source_uri="x")
    kinds = [b.kind for b in bundle.sections[0].blocks]
    assert BlockKind.TABLE in kinds


def test_plain_text_converter_round_trip() -> None:
    """PlainTextConverter builds a single-section bundle."""
    bundle = PlainTextConverter().convert(
        source_uri="x", file_bytes=b"hello world", mime_type="text/plain"
    )
    assert bundle.sections
    assert any(b.content == "hello world" for b in bundle.sections[0].blocks)


def test_word_window_chunkerword_window_chunks() -> None:
    """WordWindowChunker splits text on whitespace boundaries."""
    chunker = WordWindowChunker(chunk_size=5, chunk_overlap=2)
    text = " ".join(f"word{i}" for i in range(20))
    chunks = chunker.chunk_text(text, document_id="d1", version=1, company="o", owner="u")
    assert len(chunks) > 1
    assert all(c.document_id == "d1" for c in chunks)


def test_word_window_chunker_chunk_bundle() -> None:
    """WordWindowChunker.chunk walks bundle.sections/blocks."""
    chunker = WordWindowChunker(chunk_size=4, chunk_overlap=1)
    bundle = KnowledgeBundle(
        source_uri="file://x",
        sections=[
            {
                "index": 0,
                "blocks": [{"kind": "text", "content": "alpha beta gamma delta epsilon"}],
            }
        ],
    )
    chunks = chunker.chunk(bundle)
    assert chunks
    assert chunks[0].document_id == bundle.bundle_id


def test_ingest_pipeline_end_to_end() -> None:
    """The ingest pipeline produces chunks and embeddings."""
    bundle = KnowledgeBundle(
        source_uri="file://x",
        sections=[
            {"index": 0, "blocks": [{"kind": "text", "content": "alpha beta gamma delta"}]}
        ],
    )
    embedder = HashingEmbeddingProvider(dimension=16, model_name="hashing-test")
    store = InMemoryVectorStore()
    pipeline = IngestPipeline(
        converter=PlainTextConverter(),
        chunker=WordWindowChunker(chunk_size=4, chunk_overlap=1),
        embedder=embedder,
        vector_store=store,
        knowledge_repo=InMemoryKnowledgeRepository(),
    )
    result = asyncio.run(
        pipeline.run(
            PipelineContext(pipeline_name="ingest"),
            file_bytes=b"alpha beta gamma delta epsilon",
            source_uri="file://x",
            mime_type="text/plain",
        )
    )
    assert result.success, result.error
    assert result.outputs["chunk_count"] >= 1
    assert len(result.outputs["embeddings"]) == result.outputs["chunk_count"]


def test_query_pipeline_returns_answer() -> None:
    """The query pipeline returns an answer + citations."""
    embedder = HashingEmbeddingProvider(dimension=16, model_name="hashing-test")
    store = InMemoryVectorStore()
    llm = HeuristicLLMProvider()
    pipeline = QueryPipeline(
        embedder=embedder,
        vector_store=store,
        generator=DefaultGenerator(llm=llm),
    )
    # Seed the store directly.
    from raghub.models import ChunkRecord

    store.upsert(
        [
            ChunkRecord(
                chunk_id="c1",
                document_id="d1",
                version=1,
                text="revenue is high",
                company="o",
                owner="u",
            )
        ],
        [embedder.embed_text("revenue is high")],
    )
    result = asyncio.run(
        pipeline.run(PipelineContext(pipeline_name="query"), question="revenue")
    )
    assert result.success, result.error
    assert result.outputs["answer"]
    assert result.outputs["hits"]
