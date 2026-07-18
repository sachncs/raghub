"""Smoke tests for the smaller modules that exist only for re-exports."""

from __future__ import annotations


def test_raghub_init_exposes_rag() -> None:
    """``raghub.RAG`` is the public facade class."""
    from raghub import RAG

    assert RAG is not None
    assert hasattr(RAG, "from_config")
    assert hasattr(RAG, "ingest")
    assert hasattr(RAG, "query")
    assert hasattr(RAG, "delete")
    assert hasattr(RAG, "sync_index")


def test_raghub_api_init_lazy_creates_app() -> None:
    """``raghub.api.create_app`` is lazily importable."""
    from raghub.api import create_app

    assert callable(create_app)


def test_raghub_observability_init() -> None:
    """The observability package re-exports the key helpers."""
    from raghub.observability import (
        LoguruTelemetryProvider,
        NoOpTelemetry,
        RedactingTelemetry,
        build_logger,
    )

    assert NoOpTelemetry is not None
    assert RedactingTelemetry is not None
    assert LoguruTelemetryProvider is not None
    assert callable(build_logger)


def test_raghub_telemetry_init() -> None:
    """The telemetry package re-exports Langfuse + NoopSpan."""
    from raghub.telemetry import LangfuseTelemetryProvider, NoopSpan

    assert LangfuseTelemetryProvider is not None
    assert NoopSpan is not None


def test_raghub_knowledge_init() -> None:
    """The knowledge package re-exports OKF helpers and the manifest."""
    from raghub.knowledge import (
        InMemoryKnowledgeRepository,
        SourceManifest,
        from_okf,
        loads,
        to_okf,
    )

    assert InMemoryKnowledgeRepository is not None
    assert SourceManifest is not None
    assert callable(from_okf)
    assert callable(loads)
    assert callable(to_okf)


def test_raghub_pipelines_init() -> None:
    """The pipelines package re-exports the standard pipelines."""
    from raghub.pipelines import IngestPipeline, QueryPipeline

    assert IngestPipeline is not None
    assert QueryPipeline is not None


def test_raghub_structured_init() -> None:
    """The structured package re-exports the Instructor provider."""
    from raghub.structured import InstructorStructuredOutputProvider

    assert InstructorStructuredOutputProvider is not None


def test_raghub_documents_init() -> None:
    """The documents package re-exports the chunker and lifecycle."""
    from raghub.documents import ChunkingPlan, DocumentLifecycleManager

    assert ChunkingPlan is not None
    assert DocumentLifecycleManager is not None


def test_raghub_retrieval_init() -> None:
    """The retrieval package re-exports the reranker."""
    from raghub.retrieval import IdentityReranker

    assert IdentityReranker is not None


def test_raghub_prompts_init() -> None:
    """The prompts package re-exports the canonical builder."""
    from raghub.prompts import (
        PromptBuilder,
        PromptConfig,
        SYSTEM_PROMPT_TEMPLATE,
        TokenCounter,
    )

    assert PromptBuilder is not None
    assert PromptConfig is not None
    assert TokenCounter is not None
    assert isinstance(SYSTEM_PROMPT_TEMPLATE, str)


def test_raghub_config_init() -> None:
    """The config package re-exports ``AppSettings`` and ``load_settings``."""
    from raghub.config import AppSettings, load_settings

    assert AppSettings is not None
    assert callable(load_settings)


def test_raghub_auth_init() -> None:
    """The auth package re-exports the user store and RBAC service."""
    from raghub.auth import SqliteUserStore

    assert SqliteUserStore is not None


def test_raghub_domain_init() -> None:
    """The domain package re-exports the legacy domain classes."""
    from raghub.domain import document, session

    assert document.Document is not None
    assert session.SessionRecord is not None


def test_raghub_vectorstore_init() -> None:
    """The vector store package re-exports ``InMemoryVectorStore`` and friends."""
    from raghub.vectorstore.memory import InMemoryVectorStore
    from raghub.vectorstore.qdrant import QdrantVectorStore
    from raghub.vectorstore.zvec import ZvecVectorStore

    assert InMemoryVectorStore is not None
    assert QdrantVectorStore is not None
    assert ZvecVectorStore is not None


def test_raghub_interfaces_init_has_submodules() -> None:
    """The interfaces package exposes all submodules."""
    from raghub.interfaces import chunker, converter, embeddings, evaluation
    from raghub.interfaces import generator, knowledge, llm, observability
    from raghub.interfaces import pipeline, plugin, prompts, retrieval
    from raghub.interfaces import storage, structured, vectorstore, workers

    for module in (
        chunker,
        converter,
        embeddings,
        evaluation,
        generator,
        knowledge,
        llm,
        observability,
        pipeline,
        plugin,
        prompts,
        retrieval,
        storage,
        structured,
        vectorstore,
        workers,
    ):
        assert module is not None


def test_raghub_models_init_exposes_canonical_names() -> None:
    """``raghub.models`` re-exports every spec-mandated model name."""
    from raghub import models

    for name in (
        "Document",
        "DocumentBlock",
        "DocumentSection",
        "KnowledgeBundle",
        "Chunk",
        "Embedding",
        "Citation",
        "SearchResult",
        "CanonicalQuery",
        "CanonicalResponse",
        "PipelineContext",
        "PipelineResult",
        "EvaluationResult",
        "deterministic_id",
    ):
        assert hasattr(models, name), f"missing {name}"
