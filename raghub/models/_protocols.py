"""Protocol contracts (chunkers, providers, generators, retriever, etc.).

The contracts that other packages depend on; split out so that
``raghub.models`` does not become a god module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, TypedDict, runtime_checkable

from raghub.models._api import Pipeline, PipelineCtx, Result
from raghub.models._document import Bundle, Chunk, Citation, Hit
from raghub.models._identity import Turn
from raghub.types import JSONValue

__all__ = [
    "BackgroundWorker",
    "Chunker",
    "ConversationStore",
    "DocumentConverter",
    "DocumentRegistry",
    "EmbeddingProvider",
    "Evaluator",
    "GeneratorProtocol",
    "KnowledgeRepository",
    "LLMProvider",
    "Logger",
    "Metrics",
    "PipelineRunner",
    "Prompt",
    "RagComponents",
    "RagQueryRequest",
    "Reranker",
    "Retriever",
    "SessionStoreProtocol",
    "Span",
    "StructuredOutputProvider",
    "StructuredOutputResult",
    "TaskQueue",
    "TelemetryProvider",
    "VectorStore",
]


class Chunker(Protocol):
    """Splits a :class:`Bundle` (or raw text) into :class:`Chunk` records."""

    chunk_size: int
    chunk_overlap: int

    def chunk(self, bundle: Bundle) -> list[Chunk]:
        """Split a knowledge bundle into chunks."""
        ...

    def chunk_text(self, text: str, *, document_id: str, version: int = 1) -> list[Chunk]:
        """Split raw text into chunks."""
        ...


class DocumentConverter(Protocol):
    """Converts source bytes to a :class:`Bundle`."""

    def convert(
        self,
        *,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str = "",
        language: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Bundle:
        """Convert source bytes to a Bundle."""
        ...


class EmbeddingProvider(Protocol):
    """Embeds text into a fixed-dimensional vector."""

    model_name: str

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    async def aembed_text(self, text: str) -> list[float]:
        """Async variant of :meth:`embed_text`."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple text strings."""
        ...


class Evaluator(Protocol):
    """Scores model outputs against a benchmark dataset."""

    benchmark: str

    async def evaluate(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        response_factory: Any,
    ) -> list[Result]:
        """Score model outputs against a benchmark."""
        ...


class GeneratorProtocol(Protocol):
    """Generates an answer from retrieved context."""

    async def generate(
        self,
        *,
        question: str,
        context: Sequence[Hit],
        conversation: Sequence[Turn] = (),
    ) -> tuple[str, list[Citation]]:
        """Generate an answer from retrieved context."""
        ...

    async def astream(
        self,
        *,
        question: str,
        context: Sequence[Hit],
        conversation: Sequence[Turn] = (),
    ) -> AsyncIterator[str]:
        """Stream-generated answer tokens."""
        ...


class RagQueryRequest(TypedDict, total=False):
    """Optional inputs for :meth:`raghub.rag.RAG.aquery` and :meth:`astream`.

    All keys are optional except ``question``. Used to bundle the
    wide parameter set into a single argument so the facade methods
    stay readable.
    """

    question: str
    user: Any
    session_id: str
    top_k: int
    user_filter: dict[str, JSONValue]
    rbac_filter: dict[str, JSONValue]
    history: list[Turn]
    response_model: type
    record: bool
    tools_enabled: set[str]
    metadata: dict[str, JSONValue]
    resolved_config: dict[str, JSONValue]


class RagComponents(TypedDict, total=False):
    """Outputs of :func:`RAG.wire` — a typed dict for the components bag."""

    vector_store: Any
    embedder: Any
    llm: Any
    generator: Any
    reranker: Any
    retriever: Any
    agent: Any
    prompt: Any
    telemetry: Any
    long_context_pass: Any
    knowledge_repo: Any


class KnowledgeRepository(Protocol):
    """Persistence contract for knowledge-bundle storage (Stage 1.6)."""

    async def save(self, bundle: Bundle) -> None:
        """Persist a Bundle and return its id."""
        ...

    async def get(self, bundle_id: str) -> Bundle | None:
        """Look up a Bundle by id."""
        ...

    async def list_for_tenant(self, tenant_id: str | None, limit: int = 100) -> list[Bundle]:
        """Return Bundles for one tenant, newest first."""
        ...


class Logger(Protocol):
    """Logs structured messages with attribute kwarg binding."""

    def info(self, message: str, **kwargs: JSONValue) -> None: ...

    def warning(self, message: str, **kwargs: JSONValue) -> None: ...

    def error(self, message: str, **kwargs: JSONValue) -> None: ...


class Metrics(Protocol):
    """Counter + histogram + gauge surface for telemetry."""

    def increment(self, name: str, value: int = 1) -> None: ...

    def record_latency(self, name: str, value_ms: float) -> None: ...


@runtime_checkable
class Span(Protocol):
    """One timed sub-operation in a trace."""

    name: str
    attrs: dict[str, JSONValue]

    def set_attribute(self, key: str, value: JSONValue) -> None: ...

    def end(self) -> None: ...


class TelemetryProvider(Logger, Metrics, Protocol):
    """Composite telemetry surface combining Logger + Metrics + Span.

    Most call sites do not need to distinguish Logger from Metrics; the
    combined interface is enough.
    """

    def start_span(self, name: str, **attrs: JSONValue) -> Span:
        """Open a sub-operation span. Returned span ends on context exit."""
        ...

    def record_tokens(
        self, name: str, prompt_tokens: int, completion_tokens: int, model: str = ""
    ) -> None: ...


class StructuredOutputResult(TypedDict, total=False):
    """A single StructuredOutput result (matches the LLM tool-call shape)."""

    name: str
    arguments: dict[str, JSONValue]
    raw: str


class StructuredOutputProvider(Protocol):
    """Generates a StructuredOutput (tool call) instead of free-form text.

    Used by the RAG pipeline when the response_model is set.
    """

    name: str

    def get_schema(self) -> dict[str, Any]:
        """Return the JSON schema for the structured output."""
        ...

    async def generate(
        self,
        *,
        question: str,
        context: Sequence[Hit],
        schema: dict[str, Any],
    ) -> StructuredOutputResult:
        """Generate a structured output from ``question`` and ``context``."""
        ...


class VectorStore(Protocol):
    """Vector + metadata storage backend.

    Implementations: in-process (MemoryStore), SQLite (SqliteStore),
    and optional Postgres / pgvector.
    """

    name: str

    def create_collection(self) -> None:
        """Create the underlying collection (idempotent)."""
        ...

    def insert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        """Insert ``chunks`` with corresponding ``vectors``; return # inserted."""
        ...

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[Hit]:
        """Return top-k hits by similarity."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Delete chunks by id; return # deleted."""
        ...

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk for ``document_id``; return # deleted."""
        ...

    def health(self) -> dict[str, Any]:
        """Return backend-specific health snapshot."""
        ...

    def keyword_search(self, query: str, top_k: int = 5) -> list[Hit]:
        """Return keyword-only search hits (BM25 / FTS)."""
        ...


class Retriever(Protocol):
    """Performs vector + keyword retrieval and fuses the two channels.

    A retriever always returns a list of :class:`Hit` objects ordered
    by their final fused score (descending).
    """

    def retrieve(
        self,
        *,
        user: Any,
        question: str,
        top_k: int = 5,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[Hit]:
        """Run retrieval for ``question``; return fused hits."""
        ...


class Reranker(Protocol):
    """A reranker: reorder retrieval hits using a downstream signal.

    Implementations can be sync only (``rerank``), async only (or wrap a
    sync model with ``asyncio.run`` to expose ``arerank``), or both.
    """

    name: str

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Rerank ``hits`` for ``question`` synchronously; may block."""
        ...

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Asynchronously rerank ``hits`` for ``question``."""
        ...


class Prompt(Protocol):
    """Builds the prompt payload for the LLM provider."""

    def build_messages(self, request: Any) -> dict[str, Any]:
        """Build a structured message payload."""
        ...


class LLMProvider(Protocol):
    """Calls the underlying LLM with a message payload."""

    model_name: str

    async def generate(self, request: Any) -> tuple[str, dict[str, Any]]:
        """Generate the answer and return (text, usage)."""
        ...


class PipelineRunner(Protocol):
    """Runs a pipeline given its inputs."""

    name: str

    async def run(self, context: PipelineCtx, **inputs: Any) -> Pipeline:
        """Run the pipeline and return the result."""
        ...


class BackgroundWorker(Protocol):
    """Runs tasks in the background (threadpool or persistent queue)."""

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Submit ``fn`` for background execution; return a handle."""
        ...


class TaskQueue(Protocol):
    """Persistent task queue (SQLite / Redis / SQS)."""

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue a named task; return its job id."""
        ...


class DocumentRegistry(Protocol):
    """The interface used by the ingest pipeline for documents."""

    async def save(self, document: Any) -> None:
        """Persist a new or updated document."""
        ...


class ConversationStore(Protocol):
    """Persistent conversation-history store with sliding-window expiry."""

    async def get(self, session_id: str) -> Any:
        """Look up a session by id, sliding the expiry on access."""
        ...

    async def get_by_token(self, token: str) -> Any:
        """Look up a session by bearer token, sliding the expiry on access."""
        ...

    async def create(self, user_id: str) -> Any:
        """Create a fresh session for ``user_id``."""
        ...

    async def invalidate(self, token: str) -> None:
        """Remove the session for ``token``."""
        ...


class SessionStoreProtocol(Protocol):
    """Persistent store for :class:`Session` records (alias of ConversationStore)."""

    async def initialize(self) -> None:
        """Create the required tables."""
        ...

    async def create_from_record(self, session: Any) -> None:
        """Insert a session by-value (no fresh id)."""
        ...


class Plugin(Protocol):
    """A user-supplied extension point."""

    name: str
    type: str

    def register(self, registry: Any) -> None:
        """Register this plugin with ``registry``."""
        ...
