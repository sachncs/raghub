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
* :class:`IngestionError` — chunking, pipeline, or input-validation failures.
* :class:`EmbeddingError` — model / dimension problems.
* :class:`VectorStoreError` — backend search / persistence failures.
* :class:`RetrievalError` — RBAC / filter / retriever failures.
* :class:`GenerationError` — LLM provider failure.
* :class:`PipelineError` — orchestration / lifecycle failures.
* :class:`EvaluationError` — benchmark or scoring failures.
* :class:`AuthenticationError` / :class:`AuthorizationError` —
    credential and permission failures (canonical; used by the API
    layer to map to 401 / 403).
* :class:`RagHubError` — generic catch-all for uncategorised failures
    (e.g. atomic-write failures, generic validation).
"""

from __future__ import annotations

__all__ = [
    "AgentBudgetExceeded",
    "AuthenticationError",
    "AuthorizationError",
    "CacheMiss",
    "ConfigurationError",
    "ConversionError",
    "EmbeddingError",
    "EvaluationError",
    "GenerationError",
    "GraphUnavailableError",
    "IngestionError",
    "KnowledgeError",
    "MissingDep",
    "PipelineError",
    "PipelineFailed",
    "RagHubError",
    "RerankerError",
    "RetrievalError",
    "StreamingFormatError",
    "TelemetryError",
    "TokenBudgetExceeded",
    "ToolError",
    "TransformError",
    "VectorStoreError",
    "WebSearchError",
]


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


class AuthenticationError(RagHubError):
    """Raised when authentication fails (bad credentials, expired token).

    Canonical exception used by the API layer to map auth failures to
    HTTP 401. Import from :mod:`raghub.errors`.
    """


class AuthorizationError(RagHubError):
    """Raised when a caller lacks permission for an action.

    Canonical exception used by the API layer to map permission
    failures to HTTP 403. Import from :mod:`raghub.errors`.
    """


class TelemetryError(RagHubError):
    """Raised when a telemetry provider fails.

    Production code should catch this at the boundary and continue
    without telemetry; the framework treats telemetry as non-essential.
    """


class MissingDep(ImportError):
    """Raised when an optional runtime dependency is not installed.

    Subclasses :class:`ImportError` so existing handlers that catch
    ``ImportError`` continue to work, but adds a structured ``hint``
    field that tells callers how to install the missing package.
    """

    def __init__(self, package: str, hint: str) -> None:
        """Build a structured import error.

        Args:
            package: The distribution name that was not found
                (e.g. ``"chonkie"``).
            hint: A user-friendly install command
                (e.g. ``"pip install chonkie"``).
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

    Subclasses :class:`KeyError` so callers can catch either type with
    ``__contains__`` checks while carrying a richer message.
    """
