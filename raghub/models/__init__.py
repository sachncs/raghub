"""Domain, canonical, and transport dataclasses for raghub.

This module is the single source of truth for every persisted entity,
wire type, and request payload used by the framework. All models are
frozen dataclasses with explicit validation in ``__post_init__``.

Scope:

* **Domain** — runtime domain types (chunks, documents, users,
  sessions, turns, search results, lifecycle enums).
* **Canonical** — spec-mandated aliases plus the higher-level
  types (``Citation``, ``DocumentSection``, ``DocumentBlock``,
  ``Bundle``, ``PipelineCtx``, ``Pipeline``, ``Embedding``,
  ``Result``).
* **API** — the FastAPI request/response wire types.
* **Long-context** — ``RankedItem`` / ``RankedList`` for the
  second-pass LLM rerank.

Mapping (canonical ↔ domain):

* ``Document`` ↔ ``Document``
* ``Chunk`` ↔ ``Chunk``
* ``SearchResult`` ↔ ``Hit``
* ``Query`` ↔ ``SearchRequest``
* ``Response`` ↔ ``SearchResponse``

The :func:`deterministic_id` helper builds short stable ids for
newly-constructed dataclasses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from raghub.errors import VerificationError


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DocumentLifecycleStatus(StrEnum):
    """Document lifecycle states.

    Legal transitions are validated by the document state machine.
    ``Archived`` and ``Failed`` are terminal.
    """

    New = "NEW"
    Validating = "VALIDATING"
    Processing = "PROCESSING"
    Chunking = "CHUNKING"
    Embedding = "EMBEDDING"
    Indexing = "INDEXING"
    Ready = "READY"
    Updating = "UPDATING"
    Deleting = "DELETING"
    Archived = "ARCHIVED"
    Failed = "FAILED"


class Visibility(StrEnum):
    """Document visibility levels."""

    Private = "private"
    Organization = "organization"
    Public = "public"


class Classification(StrEnum):
    """Simplified data classification levels."""

    Internal = "internal"
    Confidential = "confidential"
    Restricted = "restricted"


class SessionKind(StrEnum):
    """Discriminator for :class:`Session` types."""

    Standard = "standard"
    Ephemeral = "ephemeral"
    Refresh = "refresh"


class DocType(StrEnum):
    """Discriminator for :class:`Document` types."""

    Pdf = "pdf"
    Markdown = "markdown"
    Html = "html"
    Text = "text"
    Csv = "csv"
    Json = "json"
    Unknown = "unknown"


class ChunkType(StrEnum):
    """Discriminator for :class:`Chunk` types."""

    Text = "text"
    Code = "code"
    Table = "table"
    Header = "header"
    ImageCaption = "image_caption"
    ListItem = "list_item"


class SectionType(StrEnum):
    """Discriminator for :class:`Section` types."""

    Text = "text"
    Table = "table"
    Figure = "figure"
    Code = "code"
    Reference = "reference"


class BlockType(StrEnum):
    """Discriminator for :class:`Block` types."""

    Text = "text"
    Table = "table"
    Figure = "figure"
    Code = "code"
    List = "list"
    Heading = "heading"


class CitationType(StrEnum):
    """Discriminator for :class:`Citation` types."""

    Direct = "direct"
    Paraphrase = "paraphrase"
    Inference = "inference"


class HitType(StrEnum):
    """Discriminator for :class:`Hit` types."""

    Dense = "dense"
    Sparse = "sparse"
    Hybrid = "hybrid"
    Keyword = "keyword"


class ResponseType(StrEnum):
    """Discriminator for :class:`Response` types."""

    Answer = "answer"
    Clarification = "clarification"
    Refusal = "refusal"
    Error = "error"


class BundleType(StrEnum):
    """Discriminator for :class:`Bundle` types."""

    Okf = "okf"
    Markdown = "markdown"
    Html = "html"
    Pdf = "pdf"


class PipelineType(StrEnum):
    """Discriminator for :class:`Pipeline` types."""

    Ingest = "ingest"
    Query = "query"
    Agent = "agent"
    Eval = "eval"


class JobType(StrEnum):
    """Discriminator for :class:`Job` types."""

    Ingest = "ingest"
    Eval = "eval"
    Reindex = "reindex"
    Export = "export"


class EventType(StrEnum):
    """Discriminator for :class:`Event` types."""

    Thought = "thought"
    ToolCall = "tool_call"
    ToolResult = "tool_result"
    AnswerChunk = "answer_chunk"
    Final = "final"


class UserKind(StrEnum):
    """Discriminator for :class:`User` types."""

    Standard = "standard"
    Admin = "admin"
    Service = "service"


class ManifestType(StrEnum):
    """Discriminator for :class:`Manifest` types."""

    Incremental = "incremental"
    Snapshot = "snapshot"


class EmbeddingType(StrEnum):
    """Discriminator for :class:`Embedding` types."""

    Dense = "dense"
    Sparse = "sparse"
    Colbert = "colbert"


class RankType(StrEnum):
    """Discriminator for :class:`RankedList` types."""

    Rrf = "rrf"
    CrossEncoder = "cross_encoder"
    Cohere = "cohere"


class ResultType(StrEnum):
    """Discriminator for :class:`Result` (eval)."""

    Passed = "passed"
    Failed = "failed"
    Errored = "errored"


class State(StrEnum):
    """Lifecycle state shared across entities with a state machine."""

    New = "new"
    Running = "running"
    Ready = "ready"
    Failed = "failed"
    Archived = "archived"


class Class(StrEnum):
    """Security classification shared across entities."""

    Public = "public"
    Internal = "internal"
    Restricted = "restricted"
    Confidential = "confidential"


class Access(StrEnum):
    """Visibility scope shared across entities."""

    Public = "public"
    Org = "org"
    Private = "private"


class BlockKind(StrEnum):
    """Coarse kinds of :class:`DocumentBlock`."""

    Text = "text"
    Table = "table"
    Equation = "equation"
    Image = "image"
    Code = "code"
    Metadata = "metadata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def deterministic_id(*parts: str, length: int = 16) -> str:
    """Build a short, deterministic id from a tuple of strings.

    SHA-256 of ``"\\x1f".join(parts)`` truncated to ``length`` hex
    characters. Re-indexing the same content yields the same id, which
    is the foundation of the incremental-indexing support.

    Args:
        *parts: Stable components (e.g. ``(source_uri, checksum)``).
        length: Length of the returned hex digest. Clamped to
            ``[8, 64]``; defaults to 16 characters.

    Returns:
        A lowercase hex string.

    """
    clamped = max(8, min(length, 64))
    joined = "\x1f".join(parts).encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(joined).hexdigest()
    return digest[:clamped]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ErrorInfo:
    """Structured error information shared across pipeline outputs."""

    kind: str = ""
    message: str = ""
    cause: str | None = None


# ---------------------------------------------------------------------------
# Identity domain
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class User:
    """Authenticated user principal."""

    id: str = field(default_factory=lambda: str(uuid4()))
    email: str = ""
    allowed_companies: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    is_admin: bool = False
    tool_settings: dict[str, Any] = field(default_factory=dict)
    type: UserKind = UserKind.Standard

    def __post_init__(self) -> None:
        if not self.id:
            raise VerificationError("User: empty id")
        if not self.email:
            raise VerificationError("User: empty email")


@dataclass(slots=True, frozen=True)
class Turn:
    """Single question-answer turn stored in session memory."""

    question: str = ""
    answer: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Session:
    """Session metadata and isolated conversational history."""

    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    token: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_seen_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    history: list[Turn] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)
    type: SessionKind = SessionKind.Standard

    def __post_init__(self) -> None:
        if not self.id:
            raise VerificationError("Session: empty id")
        if not self.token:
            raise VerificationError("Session: empty token")
        if not self.user_id:
            raise VerificationError("Session: empty user_id")


# ---------------------------------------------------------------------------
# Document domain
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Document:
    """Document data transfer object."""

    id: str = field(default_factory=lambda: str(uuid4()))
    version: int = 1
    checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    owner: str = ""
    organization: str = ""
    department: str = ""
    tags: list[str] = field(default_factory=list)
    classification: Classification = Classification.Internal
    visibility: Visibility = Visibility.Organization
    status: DocumentLifecycleStatus = DocumentLifecycleStatus.New
    filename: str = ""
    file_type: str = ""
    mime_type: str = ""
    chunk_count: int = 0
    chunks: list[str] = field(default_factory=list)
    error: str | None = None
    type: DocType = DocType.Unknown
    state: State = State.New

    def __post_init__(self) -> None:
        if not self.id:
            raise VerificationError("Document: empty id")
        if self.state == State.Failed and not self.error:
            raise VerificationError("Document: error required when state=FAILED")

    def verify(self) -> None:
        """Assert the document's lifecycle-state invariants.

        Raises:
            VerificationError: When ``state`` is ``Ready`` or
                ``Archived`` but no chunks are attached.

        """
        if self.state in {State.Ready, State.Archived} and not self.chunks:
            raise VerificationError("Document: chunks empty for non-FAILED state")


@dataclass(slots=True, frozen=True)
class Chunk:
    """Chunk metadata stored alongside the vector."""

    id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    version: int = 0
    page: int = 0
    source_location: str = ""
    section: str = ""
    company: str = ""
    owner: str = ""
    department: str = ""
    classification: Classification = Classification.Internal
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    embedding_model: str = ""
    checksum: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    type: ChunkType = ChunkType.Text
    tenant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise VerificationError("Chunk: empty id")
        if not self.text:
            raise VerificationError("Chunk: empty text")
        if not self.checksum:
            raise VerificationError("Chunk: empty checksum")
        if self.checksum != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise VerificationError("Chunk: checksum mismatch (expected sha256(text))")


@dataclass(slots=True, frozen=True)
class Hit:
    """A retrieved chunk with score and metadata."""

    chunk: Chunk
    score: float = 0.0
    type: HitType = HitType.Dense

    @property
    def chunk_id(self) -> str:
        """The id of the underlying chunk."""
        return self.chunk.id

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise VerificationError("Hit: empty chunk_id")


@dataclass(slots=True, frozen=True)
class SearchResult(Hit):
    """Alias for :class:`Hit`."""

    pass


@dataclass(slots=True, frozen=True)
class SearchRequest:
    """Search input to the retrieval pipeline."""

    user_id: str = ""
    question: str = ""
    session_id: str = ""
    top_k: int = 5


@dataclass(slots=True, frozen=True)
class Query(SearchRequest):
    """Alias for :class:`SearchRequest`."""

    pass


@dataclass(slots=True, frozen=True)
class SearchResponse:
    """Search output from the retrieval pipeline."""

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical spec-named models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DocumentBlock:
    """A single atom within a section: paragraph, table, image, equation."""

    block_id: str = field(default_factory=lambda: deterministic_id("block", str(uuid4())))
    kind: BlockKind = BlockKind.Text
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DocumentSection:
    """A logical section of a document — chapter, page, or slide."""

    section_id: str = field(default_factory=lambda: deterministic_id("section", str(uuid4())))
    index: int = 0
    heading: str = ""
    blocks: list[DocumentBlock] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    source_location: str = ""


@dataclass(slots=True, frozen=True)
class DocumentAlias(Document):
    """Alias for :class:`Document`."""

    pass


@dataclass(slots=True, frozen=True)
class ChunkAlias:
    """Alias placeholder for :class:`Chunk`."""

    pass


@dataclass(slots=True, frozen=True)
class Embedding:
    """A typed vector with provenance."""

    id: str = field(default_factory=lambda: str(uuid4()))
    target: str = ""
    model: str = ""
    dim: int = 0
    vector: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    type: EmbeddingType = EmbeddingType.Dense

    def __post_init__(self) -> None:
        if not self.id:
            raise VerificationError("Embedding: empty id")
        if not self.vector:
            raise VerificationError("Embedding: empty vector")


@dataclass(slots=True, frozen=True)
class Citation:
    """Provenance for a single answer span."""

    chunk: Chunk | None = None
    document_id: str = ""
    version: int = 1
    page: int = 0
    section: str = ""
    quote: str = ""
    score: float = 0.0
    source_uri: str = ""
    type: CitationType = CitationType.Direct

    def __post_init__(self) -> None:
        if not self.document_id:
            raise VerificationError("Citation: empty document_id")
        if self.score < 0.0:
            raise VerificationError(f"Citation: negative score ({self.score})")


@dataclass(slots=True, frozen=True)
class Citations:
    """The aggregate of citations on a :class:`Response`."""

    items: list[Citation] = field(default_factory=list)

    def verify(self, chunks: list | None = None) -> None:
        """Assert each citation's invariant and chunk membership.

        Args:
            chunks: The ``Response``'s ``source_chunks`` (or equivalent).
                When supplied, also asserts every citation's ``chunk.id``
                is in the chunks list.

        Raises:
            VerificationError: When any check fails.

        """
        for cit in self.items:
            cit.__post_init__()
        if chunks is not None:
            valid: set[str] = set()
            for c in chunks:
                cid = getattr(c, "id", None)
                if cid is None:
                    cid = getattr(c, "chunk_id", None)
                if cid is not None:
                    valid.add(str(cid))
            for cit in self.items:
                if cit.chunk is not None and cit.chunk.id not in valid:
                    raise VerificationError(
                        f"Citations: chunk_id {cit.chunk.id!r} not in source_chunks"
                    )


@dataclass(slots=True, frozen=True)
class Response:
    """Public response model with typed citations and source chunks."""

    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    source_chunks: list[Hit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    structured: dict[str, Any] | None = None
    transforms_applied: list[str] = field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = field(default_factory=list)
    type: ResponseType = ResponseType.Answer

    def __post_init__(self) -> None:
        if not self.answer and not self.citations:
            raise VerificationError("Response: empty answer and no citations")
        Citations(items=list(self.citations)).verify(chunks=list(self.source_chunks))

    def citations_aggregate(self) -> Citations:
        """Return the citations wrapped as a :class:`Citations` aggregate."""
        return Citations(items=list(self.citations))


@dataclass(slots=True, frozen=True)
class Bundle:
    """A persisted Open Knowledge Format bundle."""

    bundle_id: str = field(default_factory=lambda: deterministic_id("bundle", str(uuid4())))
    schema_version: str = "0.1"
    source_uri: str = ""
    checksum: str = ""
    language: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[DocumentSection] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class PipelineCtx:
    """Per-invocation state passed to every stage of a pipeline."""

    pipeline_id: str = field(default_factory=lambda: deterministic_id("pipeline", str(uuid4())))
    pipeline_name: str = "default"
    user: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class Pipeline:
    """Output of a pipeline run.

    The model uses an ``error: ErrorInfo | None`` discriminator rather
    than a boolean ``success`` flag: ``error is None`` means the run
    succeeded, ``error`` is set means it failed.
    """

    pipeline_id: str = ""
    pipeline_name: str = ""
    type: PipelineType = PipelineType.Ingest
    outputs: dict[str, Any] = field(default_factory=dict)
    error: ErrorInfo | None = None
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def succeeded(self) -> bool:
        """``True`` when the pipeline completed without error."""
        return self.error is None

    def __post_init__(self) -> None:
        if self.error is not None and not self.error.message:
            raise VerificationError("Pipeline: error.message required when error is set")


@dataclass(slots=True, frozen=True)
class Result:
    """Result of a single evaluation run on a benchmark example."""

    benchmark: str = ""
    example_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Long-context second-pass rerank models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RankedItem:
    """One ranked chunk in a :class:`RankedList` result."""

    id: str = ""
    score: float = 0.0
    chunk: Chunk | None = None
    rank: int = 0


@dataclass(slots=True, frozen=True)
class LongContextRankedItem:
    """A single re-ranked candidate produced by the long-context LLM."""

    chunk_id: str = ""
    score: float = 0.0
    rationale: str = ""


@dataclass(slots=True, frozen=True)
class RankedList:
    """Wrapper that lets structured-output providers validate the LLM output."""

    items: list[RankedItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AuthLoginRequest:
    """Login request payload."""

    email: str = ""
    password: str = ""


@dataclass(slots=True, frozen=True)
class AuthLoginResponse:
    """Login response payload."""

    session_token: str = ""
    user_email: str = ""
    allowed_companies: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class DocumentUploadResponse:
    """Upload response payload."""

    document_id: str = ""
    version: int = 0
    status: str = ""
    company: str = ""
    filename: str = ""


@dataclass(slots=True, frozen=True)
class QueryRequest:
    """Question answering payload."""

    question: str = ""
    tools_enabled: list[str] | None = None
    agent: bool | None = None
    web: bool | None = None
    graph: bool | None = None
    summaries: bool | None = None
    reranker: str | None = None
    long_context_pass: bool | None = None
    query_transforms: list[str] | None = None
    max_steps: int | None = None
    top_k: int | None = None


@dataclass(slots=True, frozen=True)
class QueryResponse:
    """Question answering response."""

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[dict[str, Any]] = field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = field(default_factory=list)
    transforms_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BatchIngestItem:
    """Result of ingesting a single file in a batch request."""

    filename: str = ""
    document_id: str = ""
    status: str = "ok"
    error: str = ""


@dataclass(slots=True, frozen=True)
class BatchIngestResponse:
    """Response from the batch-ingest endpoint."""

    documents: list[BatchIngestItem] = field(default_factory=list)


__all__ = [
    "Access",
    "AuthLoginRequest",
    "AuthLoginResponse",
    "BlockKind",
    "BlockType",
    "Bundle",
    "BundleType",
    "Citation",
    "Citations",
    "Chunk",
    "ChunkAlias",
    "ChunkType",
    "Class",
    "Classification",
    "DocType",
    "Document",
    "DocumentAlias",
    "DocumentBlock",
    "DocumentLifecycleStatus",
    "DocumentSection",
    "DocumentUploadResponse",
    "Embedding",
    "EmbeddingType",
    "ErrorInfo",
    "EventType",
    "Hit",
    "HitType",
    "JobType",
    "LongContextRankedItem",
    "ManifestType",
    "Pipeline",
    "PipelineCtx",
    "PipelineType",
    "Query",
    "QueryRequest",
    "QueryResponse",
    "RankType",
    "RankedItem",
    "RankedList",
    "Response",
    "ResponseType",
    "Result",
    "ResultType",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SectionType",
    "Session",
    "SessionKind",
    "State",
    "Turn",
    "User",
    "UserKind",
    "Visibility",
    "BatchIngestItem",
    "BatchIngestResponse",
    "deterministic_id",
]
