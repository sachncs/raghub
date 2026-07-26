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
    """Raised when a document conversion step fails (marker, parser, tesseract).

    Examples:
        * Marker cannot parse a malformed PDF.
        * A plain-text converter rejects empty bytes.
    """


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


class ToolError(RagHubError):
    """Raised when an agent tool cannot complete or returns malformed output."""


class AgentBudgetExceeded(RagHubError):
    """Raised when the agent loop exhausts its step / wall-clock / token budget.

    The exception carries the partial planner trace so callers can surface
    progress to the user even though no final answer was produced.
    """


class WebSearchError(RagHubError):
    """Raised when a web search tool fails (network, missing dep, parse error)."""


class RerankerError(RagHubError):
    """Raised when a reranker cannot score or rank the candidate list."""


class GraphUnavailableError(RagHubError):
    """Raised when a graph-backed feature (RAPTOR / GraphRAG) is requested
    but the required dependency (sklearn / igraph / leidenalg) is missing.
    """


class TransformError(RagHubError):
    """Raised when a query transform (HyDE / multi-query / decompose) fails."""


# ---------------------------------------------------------------------------
# Legacy / compatibility names (kept for existing API consumers).
# They are subclasses of :class:`RagHubError` so a single
# ``except RagHubError`` continues to catch everything.
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

    Examples:
        * Network timeout against the upstream provider.
        * Malformed model response.
    """


class StorageError(DynamicRagError):
    """Raised when persistent storage fails (disk full, permission denied)."""


class ValidationError(DynamicRagError):
    """Raised when caller-supplied input fails validation."""


class RateLimitError(DynamicRagError):
    """Raised when a per-caller rate limit is exceeded."""


class TelemetryError(RagHubError):
    """Raised when a telemetry provider fails.

    Production code should catch this at the boundary and continue
    without telemetry; the framework treats telemetry as non-essential.
    """


class OptionalDependencyMissing(ImportError):
    """Raised when an optional runtime dependency is not installed.

    Subclasses :class:`ImportError` so existing handlers that catch
    ``ImportError`` continue to work, but adds a structured ``hint``
    field that tells callers how to install the missing package.
    """

    def __init__(self, package: str, hint: str) -> None:
        """Build a structured import error.

        Args:
            package: The distribution name that was not found
                (e.g. ``"qdrant-client"``).
            hint: A user-friendly install command
                (e.g. ``"pip install qdrant-client"``).
        """
        super().__init__(f"{hint}; the {package!r} distribution is not installed")
        self.package = package
        self.hint = hint


class PipelineFailed(RagHubError):
    """Raised by an orchestration pipeline when a step fails irrecoverably.

    Carries the offending step name and partial result so callers can
    resume or surface a useful error to the user.
    """

    def __init__(self, step: str, message: str, partial: object | None = None) -> None:
        """Build a structured pipeline failure.

        Args:
            step: The pipeline step that failed.
            message: Human-readable failure description.
            partial: Optional partial result captured before the failure.
        """
        super().__init__(f"pipeline step {step!r} failed: {message}")
        self.step = step
        self.partial = partial


class TokenBudgetExceeded(RagHubError):
    """Raised when an operation consumes more tokens than its budget allows."""


class StreamingFormatError(GenerationError):
    """Raised when SSE stream formatting fails (invalid event payload)."""


class CacheMiss(KeyError):
    """Raised by ``cache.get_or_raise()`` when the key is absent.

    Subclasses :class:`KeyError` for compatibility with
    ``__contains__`` checks while carrying a richer message.
    """


__all__ = [
    "AgentBudgetExceeded",
    "AuthenticationError",
    "AuthorizationError",
    "CacheMiss",
    "ConfigurationError",
    "ConversionError",
    "DocumentError",
    "DynamicRagError",
    "EmbeddingError",
    "EvaluationError",
    "GenerationError",
    "GraphUnavailableError",
    "IndexingError",
    "IngestionError",
    "KnowledgeError",
    "LLMError",
    "OptionalDependencyMissing",
    "PipelineError",
    "PipelineFailed",
    "PromptError",
    "RagHubError",
    "RateLimitError",
    "RerankerError",
    "RetrievalError",
    "StorageError",
    "StreamingFormatError",
    "TelemetryError",
    "TokenBudgetExceeded",
    "ToolError",
    "TransformError",
    "ValidationError",
    "VectorStoreError",
    "WebSearchError",
]
