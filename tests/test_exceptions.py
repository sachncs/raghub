"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from raghub.errors import (
    AgentBudgetExceeded,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConversionError,
    DocumentError,
    DynamicRagError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    GraphUnavailableError,
    IndexingError,
    IngestionError,
    KnowledgeError,
    LLMError,
    PipelineError,
    PromptError,
    RagHubError,
    RerankerError,
    RetrievalError,
    StorageError,
    ToolError,
    TransformError,
    VectorStoreError,
    WebSearchError,
)


@pytest.mark.parametrize(
    "cls",
    [
        ConfigurationError,
        ConversionError,
        KnowledgeError,
        IngestionError,
        EmbeddingError,
        VectorStoreError,
        RetrievalError,
        GenerationError,
        PipelineError,
        EvaluationError,
    ],
)
def test_spec_exceptions_subclass_raghub_error(cls) -> None:
    """Every spec exception descends from RagHubError."""
    assert issubclass(cls, RagHubError)


@pytest.mark.parametrize(
    "cls",
    [
        AuthenticationError,
        AuthorizationError,
        DocumentError,
        IndexingError,
        PromptError,
        LLMError,
        StorageError,
    ],
)
def test_legacy_exceptions_subclass_dynamic_rag_error(cls) -> None:
    """Legacy names keep working under DynamicRagError."""
    assert issubclass(cls, DynamicRagError)
    assert issubclass(cls, RagHubError)


def test_generation_error_is_llm_error_subclass() -> None:
    """GenerationError is the new name; LLMError is the legacy alias."""
    assert issubclass(LLMError, RagHubError)
    assert issubclass(GenerationError, RagHubError)


@pytest.mark.parametrize(
    "cls",
    [
        ToolError,
        AgentBudgetExceeded,
        WebSearchError,
        RerankerError,
        GraphUnavailableError,
        TransformError,
    ],
)
def test_phase1_exceptions_subclass_raghub_error(cls) -> None:
    """Phase 1.5 advanced-RAG exceptions descend from RagHubError."""
    assert issubclass(cls, RagHubError)
