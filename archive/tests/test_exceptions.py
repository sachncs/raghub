"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from raghub.errors import (
    AgentBudgetError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConversionError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    GraphUnavailableError,
    IngestionError,
    KnowledgeError,
    PipelineError,
    RagHubError,
    RerankerError,
    RetrievalError,
    ToolError,
    TransformError,
    VectorStoreError,
    WebSearchError,
)


@pytest.mark.parametrize(
    "cls",
    [
        AuthenticationError,
        AuthorizationError,
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
def test_canonical_exceptions_subclass_raghub_error(cls) -> None:
    """Every canonical exception descends from RagHubError."""
    assert issubclass(cls, RagHubError)


@pytest.mark.parametrize(
    "cls",
    [
        ToolError,
        AgentBudgetError,
        WebSearchError,
        RerankerError,
        GraphUnavailableError,
        TransformError,
    ],
)
def test_phase1_exceptions_subclass_raghub_error(cls) -> None:
    """Phase 1.5 advanced-RAG exceptions descend from RagHubError."""
    assert issubclass(cls, RagHubError)
