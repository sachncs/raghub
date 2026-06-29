"""Custom exception hierarchy for the RAGHub framework.

All package exceptions descend from :class:`RagHubError`, so callers
can catch every framework-raised error with a single
``except RagHubError`` block. Subclasses carry finer-grained names so
production handlers can react differently without inspecting string
messages.

Exception groups mirror the framework's domain modules:

* :class:`ConfigurationError` — bad configuration / missing secrets.
* :class:`ConversionError` — marker or other converter failures.
* :class:`KnowledgeError` — OKF / knowledge repository failures.
* :class:`IngestionError` — chunking or pipeline failures.
* :class:`EmbeddingError` — model / dimension problems.
* :class:`VectorStoreError` — backend search / persistence failures.
* :class:`RetrievalError` — RBAC / filter / retriever failures.
* :class:`GenerationError` — LLM provider failure.
* :class:`PipelineError` — orchestration / lifecycle failures.
* :class:`EvaluationError` — benchmark or scoring failures.

Legacy names (``AuthenticationError``, ``AuthorizationError``,
``DocumentError``, ``IndexingError``, ``PromptError``, ``LLMError``,
``StorageError``) are preserved for backward compatibility.
"""

from __future__ import annotations


class RagHubError(Exception):
    """Base class for all package errors.

    Catch this to handle any framework-raised exception. Concrete
    subclasses provide the specific failure context.
    """


# ---------------------------------------------------------------------------
# New spec-mandated categories
# ---------------------------------------------------------------------------


class ConfigurationError(RagHubError):
    """Raised when configuration is invalid or missing required secrets."""


class ConversionError(RagHubError):
    """Raised when a document conversion step fails (marker, parser, tesseract)."""


class KnowledgeError(RagHubError):
    """Raised when the OKF knowledge layer cannot serialise or persist data."""


class IngestionError(RagHubError):
    """Raised when the ingestion pipeline cannot complete a document."""


class EmbeddingError(RagHubError):
    """Raised when text embedding fails (model error, dimension mismatch)."""


class VectorStoreError(RagHubError):
    """Raised when a vector store backend fails (search, insert, delete)."""


class RetrievalError(RagHubError):
    """Raised when retrieval fails (vector store, filter, RBAC)."""


class GenerationError(RagHubError):
    """Raised when the LLM generation step fails."""


class PipelineError(RagHubError):
    """Raised when a pipeline orchestration step fails."""


class EvaluationError(RagHubError):
    """Raised when an evaluator cannot score a model output."""


# ---------------------------------------------------------------------------
# Legacy / compatibility names (kept for existing API consumers).
# They are subclasses of :class:`RagHubError` so a single ``except
# RagHubError`` continues to catch everything.
# ---------------------------------------------------------------------------


class DynamicRagError(RagHubError):
    """Backward-compatible alias for :class:`RagHubError`.

    New code should prefer :class:`RagHubError`. This alias is kept so
    existing imports (``from raghub.exceptions import DynamicRagError``)
    continue to work.
    """


class AuthenticationError(DynamicRagError):
    """Raised when authentication fails (bad credentials, expired token)."""


class AuthorizationError(DynamicRagError):
    """Raised when a caller lacks permission for an action."""


class DocumentError(DynamicRagError):
    """Raised when document validation or lifecycle management fails."""


class IndexingError(DynamicRagError):
    """Raised when indexing or persistence fails."""


class PromptError(DynamicRagError):
    """Raised when prompt construction fails."""


class LLMError(DynamicRagError):
    """Raised when LLM generation fails.

    Alias of :class:`GenerationError`; kept for callers that imported
    the legacy name from older versions of the framework.
    """


class StorageError(DynamicRagError):
    """Raised when durable storage fails."""


__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ConversionError",
    "DocumentError",
    "DynamicRagError",
    "EmbeddingError",
    "EvaluationError",
    "GenerationError",
    "IndexingError",
    "IngestionError",
    "KnowledgeError",
    "LLMError",
    "PipelineError",
    "PromptError",
    "RagHubError",
    "RetrievalError",
    "StorageError",
    "VectorStoreError",
]
