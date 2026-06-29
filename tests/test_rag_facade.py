"""Tests for the RAG facade and the plugin registry."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from raghub import RAG
from raghub.plugins.registry import PluginRegistry


def test_rag_default_construction() -> None:
    """A facade can be built with no arguments (offline-only defaults)."""
    rag = RAG()
    assert rag.health()["status"] == "ok"


def test_rag_from_config(tmp_path: Path) -> None:
    """A facade can be built from a YAML config."""
    cfg = tmp_path / "rag.yaml"
    cfg.write_text(
        "environment: development\nchunk_size_words: 200\nchunk_overlap_words: 10\n",
        encoding="utf-8",
    )
    rag = RAG.from_config(cfg)
    assert rag.settings.environment == "development"
    assert rag.settings.chunk_size_words == 200


def test_rag_ingest_query_smoke() -> None:
    """Smoke: ingest plain text and ask a question."""
    from raghub.converters.plaintext import PlainTextConverter
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    rag = RAG()
    # Use the plain-text converter + a tiny chunker so this smoke
    # test works without a real PDF or LLM endpoint.
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.chunker = WordWindowChunker(chunk_size=5, chunk_overlap=1)
    rag.ingest_pipeline.chunker = rag.chunker
    rag.ingest(b"revenue grew by 10% in Q3", source_uri="mem://text")
    response = rag.query("revenue")
    assert response.answer is not None


def test_rag_ingest_rejects_empty_bytes() -> None:
    """Empty bytes raise a clear ``RagHubError``."""
    import asyncio

    from raghub.exceptions import RagHubError

    rag = RAG()
    with pytest.raises(RagHubError, match="empty bytes"):
        rag.ingest(b"", source_uri="mem://empty")
    with pytest.raises(RagHubError, match="empty bytes"):
        asyncio.run(rag.aingest(b"", source_uri="mem://empty"))


def test_rag_evaluate_calls_evaluator() -> None:
    """The evaluate() helper returns a list of EvaluationResult."""
    rag = RAG()

    async def _factory(example):
        return example.get("answer", "")

    results = rag.evaluate(
        benchmark="financebench",
        response_factory=_factory,
        examples=[{"id": "0", "question": "x", "answer": "y"}],
    )
    assert results
    assert results[0].benchmark == "financebench"


def test_rag_evaluate_unknown_benchmark() -> None:
    """An unknown benchmark raises ConfigurationError."""
    from raghub.exceptions import ConfigurationError

    rag = RAG()
    with pytest.raises(ConfigurationError):
        rag.evaluate(benchmark="wat", examples=[])


def test_rag_shutdown_issafe_call() -> None:
    """Calling shutdown() twice should be safe."""
    rag = RAG()
    rag.shutdown()
    rag.shutdown()


def test_plugin_registry_records() -> None:
    """A registry stores registrations by category."""
    reg = PluginRegistry()
    reg.register_factory("noop", lambda: None)
    assert reg.factories["noop"]() is None
    assert reg.discover_entrypoints(group="raghub.plugins.does.not.exist") == 0
