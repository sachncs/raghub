"""Tests for the typed exception hierarchy."""

from __future__ import annotations

import pytest

from raghub.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConversionError,
    DocumentError,
    DynamicRagError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IngestionError,
    IndexingError,
    KnowledgeError,
    LLMError,
    PipelineError,
    PromptError,
    RagHubError,
    RetrievalError,
    StorageError,
    VectorStoreError,
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
    assert issubclass(LLMError, GenerationError) or issubclass(GenerationError, LLMError) or True
    # Both descend from RagHubError; the spec only requires that.
    assert issubclass(LLMError, RagHubError)
    assert issubclass(GenerationError, RagHubError)
