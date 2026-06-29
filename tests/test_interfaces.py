"""Tests for the Protocol/ABC interfaces."""

from __future__ import annotations

from raghub.interfaces.chunker import Chunker
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.generator import Generator
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.interfaces.observability import (
    Logger,
    Metrics,
    Span,
    TelemetryProvider,
)
from raghub.interfaces.retrieval import Reranker
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.interfaces.vectorstore import VectorStore


def test_chunker_protocol_has_chunk_methods() -> None:
    """The :class:`Chunker` protocol defines ``chunk`` and ``chunk_text``."""

    assert hasattr(Chunker, "chunk")
    assert hasattr(Chunker, "chunk_text")


def test_knowledge_repository_protocol_has_save_get() -> None:
    """The :class:`KnowledgeRepository` protocol defines ``save``, ``get``,
    ``list_by_source`` and ``delete``."""

    for name in ("save", "get", "list_by_source", "delete"):
        assert hasattr(KnowledgeRepository, name)


def test_logger_protocol_has_info_warning_error() -> None:
    """The :class:`Logger` protocol defines the three log methods."""

    assert hasattr(Logger, "info")
    assert hasattr(Logger, "warning")
    assert hasattr(Logger, "error")


def test_metrics_protocol_has_record_latency_increment() -> None:
    """The :class:`Metrics` protocol defines ``record_latency`` and
    ``increment``."""

    assert hasattr(Metrics, "record_latency")
    assert hasattr(Metrics, "increment")


def test_span_protocol_has_end_and_set_attribute() -> None:
    """The :class:`Span` protocol defines ``end`` and ``set_attribute``."""

    assert hasattr(Span, "end")
    assert hasattr(Span, "set_attribute")


def test_telemetry_provider_is_runtime_checkable() -> None:
    """The :class:`TelemetryProvider` Protocol can be used as a structural
    type-check at runtime."""

    assert hasattr(TelemetryProvider, "start_span")
    assert hasattr(TelemetryProvider, "end_span")
    assert hasattr(TelemetryProvider, "record_tokens")


def test_structured_output_provider_protocol() -> None:
    """The :class:`StructuredOutputProvider` protocol defines
    ``generate`` and ``astream``."""

    assert hasattr(StructuredOutputProvider, "generate")
    assert hasattr(StructuredOutputProvider, "astream")


def test_generator_protocol_defines_methods() -> None:
    """The :class:`Generator` protocol defines ``generate`` and ``astream``."""

    assert hasattr(Generator, "generate")
    assert hasattr(Generator, "astream")


def test_reranker_protocol_defines_method() -> None:
    """The :class:`Reranker` protocol defines ``rerank``."""

    assert hasattr(Reranker, "rerank")


def test_embedding_provider_protocol() -> None:
    """The :class:`EmbeddingProvider` protocol defines ``embed_text`` and
    ``embed_texts``."""

    assert hasattr(EmbeddingProvider, "embed_text")
    assert hasattr(EmbeddingProvider, "embed_texts")


def test_vector_store_protocol_defines_methods() -> None:
    """The :class:`VectorStore` protocol defines the standard CRUD shape."""

    for name in (
        "create_collection",
        "insert",
        "upsert",
        "delete",
        "delete_document",
        "search",
        "hybrid_search",
        "optimize",
        "health",
    ):
        assert hasattr(VectorStore, name), name


def test_knowledge_repository_is_protocol() -> None:
    """The :class:`KnowledgeRepository` is declared as a Protocol."""

    assert getattr(KnowledgeRepository, "_is_protocol", False) is True
