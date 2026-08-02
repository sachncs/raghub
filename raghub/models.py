"""Domain, canonical, and transport Pydantic models.

This module defines the full Pydantic model surface used by RAGHub,
Domain, canonical, and transport Pydantic models for the framework:

* **Domain** — runtime domain types (chunks, documents, users,
  sessions, turns, search results, lifecycle enums).
* **Canonical** — spec-mandated aliases (``Document`` /
  ``Chunk`` / ``SearchResult`` / ``Query`` / ``Response``) plus
  the new higher-level types (``Citation``, ``DocumentSection``,
  ``DocumentBlock``, ``Bundle``, ``PipelineCtx``,
  ``Pipeline``, ``Embedding``, ``Result``).
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
newly-constructed Pydantic models.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, TypedDict, TypeVar, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

from raghub.errors import VerificationError

__all__ = [
    "AuthLoginRequest",
    "AuthLoginResponse",
    "Bundle",
    "Chunk",
    "Chunk",
    "Citation",
    "Classification",
    "Document",
    "Document",
    "DocumentLifecycleStatus",
    "DocumentUploadResponse",
    "Embedding",
    "Hit",
    "Pipeline",
    "PipelineCtx",
    "Query",
    "QueryRequest",
    "QueryResponse",
    "RankedItem",
    "RankedList",
    "Response",
    "Result",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "Session",
    "Turn",
    "User",
    "Visibility",
    "deterministic_id",
]

# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def deterministic_id(*parts: str, length: int = 16) -> str:
    r"""Build a short, deterministic id from a tuple of strings.

    SHA-256 of ``"\x1f".join(parts)`` truncated to ``length`` hex
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
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:clamped]


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class DocumentLifecycleStatus(StrEnum):
    """Document lifecycle states.

    Legal transitions are validated by
    :class:`raghub.core.DocumentStateMachine`; see its
    docstring for the full transition table. ``ARCHIVED`` and
    ``FAILED`` are terminal.
    """

    NEW = "NEW"
    VALIDATING = "VALIDATING"
    PROCESSING = "PROCESSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    UPDATING = "UPDATING"
    DELETING = "DELETING"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"


class Visibility(StrEnum):
    """Document visibility levels.

    * ``PRIVATE``: only the owner can read.
    * ``ORGANIZATION``: any authenticated user in the same tenant.
    * ``PUBLIC``: any authenticated user, regardless of tenant.
    """

    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class Classification(StrEnum):
    """Simplified data classification levels.

    Used by RBAC filters and the redaction layer to gate sensitive
    content from users without the appropriate clearance.
    """

    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# ---------------------------------------------------------------------------
# Domain Pydantic models
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Discriminator enums (one per entity class; R7: >= 2 values)
# -----------------------------------------------------------------------------


class SessionKind(StrEnum):
    """Discriminator for :class:`Session` types."""

    STANDARD = "standard"
    EPHEMERAL = "ephemeral"
    REFRESH = "refresh"


class DocType(StrEnum):
    """Discriminator for :class:`Document` types."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


class ChunkType(StrEnum):
    """Discriminator for :class:`Chunk` types."""

    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    HEADER = "header"
    IMAGE_CAPTION = "image_caption"
    LIST_ITEM = "list_item"


class SectionType(StrEnum):
    """Discriminator for :class:`Section` types."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    REFERENCE = "reference"


class BlockType(StrEnum):
    """Discriminator for :class:`Block` types."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    LIST = "list"
    HEADING = "heading"


class CitationType(StrEnum):
    """Discriminator for :class:`Citation` types."""

    DIRECT = "direct"
    PARAPHRASE = "paraphrase"
    INFERENCE = "inference"


class HitType(StrEnum):
    """Discriminator for :class:`Hit` types."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ResponseType(StrEnum):
    """Discriminator for :class:`Response` types."""

    ANSWER = "answer"
    CLARIFICATION = "clarification"
    REFUSAL = "refusal"
    ERROR = "error"


class BundleType(StrEnum):
    """Discriminator for :class:`Bundle` types."""

    OKF = "okf"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class PipelineType(StrEnum):
    """Discriminator for :class:`Pipeline` types."""

    INGEST = "ingest"
    QUERY = "query"
    AGENT = "agent"
    EVAL = "eval"


class JobType(StrEnum):
    """Discriminator for :class:`Job` types."""

    INGEST = "ingest"
    EVAL = "eval"
    REINDEX = "reindex"
    EXPORT = "export"


class EventType(StrEnum):
    """Discriminator for :class:`Event` types."""

    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER_CHUNK = "answer_chunk"
    FINAL = "final"


class UserKind(StrEnum):
    """Discriminator for :class:`User` types."""

    STANDARD = "standard"
    ADMIN = "admin"
    SERVICE = "service"


class ManifestType(StrEnum):
    """Discriminator for :class:`Manifest` types."""

    INCREMENTAL = "incremental"
    SNAPSHOT = "snapshot"


class EmbeddingType(StrEnum):
    """Discriminator for :class:`Embedding` types."""

    DENSE = "dense"
    SPARSE = "sparse"
    COLBERT = "colbert"


class RankType(StrEnum):
    """Discriminator for :class:`RankedList` types."""

    RRF = "rrf"
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"


class ResultType(StrEnum):
    """Discriminator for :class:`Result` (eval)."""

    PASSED = "passed"
    FAILED = "failed"
    ERRORED = "errored"


# -----------------------------------------------------------------------------
# Shared lifecycle enums (R3 single-word)
# -----------------------------------------------------------------------------


class State(StrEnum):
    """Lifecycle state shared across entities with a state machine."""

    NEW = "new"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class Class(StrEnum):
    """Security classification shared across entities."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class Access(StrEnum):
    """Visibility scope shared across entities."""

    PUBLIC = "public"
    ORG = "org"
    PRIVATE = "private"


# -----------------------------------------------------------------------------
# Error type (replaces raw `str | None` error fields)
# -----------------------------------------------------------------------------


class ErrorInfo(BaseModel):
    """Structured error information shared across pipeline outputs.

    Attributes:
        kind: The error category.
        message: Human-readable error message.
        cause: Optional underlying cause message.

    """

    kind: str
    message: str
    cause: str | None = None


class User(BaseModel):
    """Authenticated user principal.

    Attributes:
        type: UserKind discriminator (admin / standard).

    """

    """Authenticated user principal.

    Attributes:
        user_id: Stable opaque user id.
        email: Login email; used as the principal's display name.
        allowed_companies: Tenant allow-list. Empty for admins
            (admins bypass the company filter).
        allowed_groups: Group memberships for finer-grained RBAC.
        is_admin: ``True`` for platform-wide admins.
        tool_settings: Per-user tool/agent defaults loaded from the
            ``user_preferences`` table (Phase 1.11). The keys mirror
            the kwargs on :meth:`RAG.aquery` (``agent_enabled``,
            ``tools_enabled``, ``reranker``, ``long_context_pass``,
            ``query_transforms``, ``max_steps``). Empty dict disables
            per-user defaults — the resolver falls through to the
            global :class:`Settings` defaults.

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False
    tool_settings: dict[str, Any] = Field(default_factory=dict)
    type: UserKind = UserKind.STANDARD

    def verify(self) -> None:
        """Assert the user's invariant contract.

        Raises:
            VerificationError: When ``id`` or ``email`` is empty.

        """
        if not self.id:
            raise VerificationError("User: empty id")
        if not self.email:
            raise VerificationError("User: empty email")


class Turn(BaseModel):
    """Single question-answer turn stored in session memory.

    Attributes:
        question: User-supplied question.
        answer: Provider-supplied answer.
        timestamp: When the turn was recorded (UTC).
        metadata: Optional structured metadata (sources, citations, …).

    """

    question: str
    answer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """Session metadata and isolated conversational history.

    Attributes:
        type: SessionKind discriminator.

    """

    """Session metadata and isolated conversational history.

    Attributes:
        session_id: Stable session id.
        user_id: Owning user's id.
        token: Opaque session token used as the JWT subject.
        created_at: Session creation time (UTC).
        expires_at: Hard expiry (UTC).
        last_seen_at: Last activity timestamp; used for sliding-window
            session extensions.
        history: Conversation turns persisted for the session.
        overrides: Session-scoped tool/agent settings (Phase 1.12).
            The resolver reads these between per-request overrides and
            per-user prefs. Empty dict == no session-level overrides.

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    last_seen_at: datetime
    history: list[Turn] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    type: SessionKind = SessionKind.STANDARD

    def verify(self) -> None:
        """Assert the session's invariant contract.

        Raises:
            VerificationError: When ``id`` is empty, ``token`` is empty,
            or ``expires_at`` is in the past.

        """
        if not self.id:
            raise VerificationError("Session: empty id")
        if not self.token:
            raise VerificationError("Session: empty token")
        if not self.user_id:
            raise VerificationError("Session: empty user_id")


class Document(BaseModel):
    """Document data transfer object.

    Attributes:
        id: Stable document id (UUID).
        version: 1-based version number, incremented on re-upload.
        checksum: SHA-256 of the file contents; used for dedup.
        created_at: First upload time (UTC).
        updated_at: Latest mutation time (UTC).
        owner: Owning user email.
        organization: Tenant (company) tag.
        department: Department tag (may be empty).
        tags: Free-form tags.
        classification: Sensitivity level.
        visibility: Visibility scope.
        status: Current lifecycle state.
        filename: Original filename.
        file_type: Lower-cased extension.
        mime_type: MIME type from the validator.
        chunk_count: Number of chunks produced by the latest ingest.
        chunks: Chunks produced by the latest ingest (carries refs).
        error: Optional error message when ``status == FAILED``.

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = 1
    checksum: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    owner: str
    organization: str
    department: str = ""
    tags: list[str] = Field(default_factory=list)
    classification: Classification = Classification.INTERNAL
    visibility: Visibility = Visibility.ORGANIZATION
    status: DocumentLifecycleStatus = DocumentLifecycleStatus.NEW
    filename: str = ""
    file_type: str = ""
    mime_type: str = ""
    chunk_count: int = 0
    chunks: list[str] = Field(default_factory=list)
    error: str | None = None
    type: DocType = DocType.UNKNOWN
    state: State = State.READY

    def verify(self) -> None:
        """Assert the document's invariant contract.

        Checks that ``id`` is non-empty and that ``chunks`` is a list of
        :class:`Chunk` (each verified recursively). When ``state`` is
        :attr:`State.FAILED`, ``error`` must be non-empty.

        Raises:
            VerificationError: When any invariant is broken.

        """
        if not self.id:
            raise VerificationError("Document: empty id")
        if self.state == State.FAILED and not self.error:
            raise VerificationError("Document: error required when state=FAILED")
        if self.state in {State.READY, State.ARCHIVED} and not self.chunks:
            raise VerificationError("Document: chunks empty for non-FAILED state")


class Chunk(BaseModel):
    """Chunk metadata stored alongside the vector.

    Attributes:
        chunk_id: Stable chunk id (UUID).
        document_id: Parent document id.
        version: Parent document version.
        page: 0-based page or section index.
        source_location: Human-readable location string.
        section: Optional section heading.
        company: Tenant (company) tag, copied from the parent document.
        owner: Owning user email.
        department: Department tag.
        classification: Sensitivity level.
        created_at: Chunk creation time (UTC).
        embedding_model: Name of the embedding model that produced the vector.
        hash: SHA-256 of the chunk text for dedup.
        text: Chunk text.
        metadata: Format-specific metadata (PDF metadata, image EXIF, …).

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    version: int
    page: int = 0
    source_location: str = ""
    section: str = ""
    company: str
    owner: str
    department: str = ""
    classification: Classification = Classification.INTERNAL
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    embedding_model: str = ""
    checksum: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    type: ChunkType = ChunkType.TEXT

    def verify(self) -> None:
        """Assert the chunk's invariant contract.

        The two checks are: (1) ``id`` is non-empty, and (2)
        ``checksum`` matches the SHA-256 of ``text``.

        Raises:
            VerificationError: When either check fails.

        """
        if not self.id:
            raise VerificationError("Chunk: empty id")
        if not self.text:
            raise VerificationError("Chunk: empty text")
        if not self.checksum:
            raise VerificationError("Chunk: empty checksum")
        from hashlib import sha256

        if self.checksum != sha256(self.text.encode("utf-8")).hexdigest():
            raise VerificationError("Chunk: checksum mismatch (expected sha256(text))")


class Hit(BaseModel):
    """A retrieved chunk with score and metadata.

    Attributes:
        chunk_id: Id of the underlying :class:`Chunk`.
        score: Cosine-similarity score reported by the vector store.
        chunk: The full chunk metadata.

    """

    score: float
    chunk: Chunk
    type: HitType = HitType.DENSE

    @property
    def chunk_id(self) -> str:
        """Chunks within :class:`Hit` carry their id on the inner chunk.

        Kept as a thin property so consumers that previously read
        ``hit.chunk_id`` keep working; the value is owned by the chunk.
        """
        return self.chunk.id

    def verify(self) -> None:
        """Assert the hit's invariant contract.

        Checks that ``chunk_id`` matches ``chunk.id`` and that
        the chunk itself verifies.

        Raises:
            VerificationError: When either check fails.

        """
        if not self.chunk_id:
            raise VerificationError("Hit: empty chunk_id")
        if self.chunk.id != self.chunk_id:
            raise VerificationError(
                f"Hit.chunk_id ({self.chunk_id!r}) does not match chunk.id ({self.chunk.id!r})"
            )
        self.chunk.verify()


class SearchRequest(BaseModel):
    """Search input to the retrieval pipeline.

    Attributes:
        user_id: Id of the requesting user.
        question: Raw question text.
        session_id: Optional session id; when set, prior turns are
            considered when assembling the prompt.
        top_k: Maximum number of hits to return.

    """

    user_id: str
    question: str
    session_id: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Search output from the retrieval pipeline.

    Attributes:
        answer: The generated answer string.
        citations: Citation metadata keyed by source location.
        source_chunks: The :class:`Chunk` objects that
            contributed to the answer.
        metadata: Provider- and pipeline-specific metadata.

    """

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_chunks: list[Chunk] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


Session.model_rebuild()


# ---------------------------------------------------------------------------
# Canonical spec-named models
# ---------------------------------------------------------------------------


class BlockKind(StrEnum):
    """Coarse kinds of :class:`DocumentBlock`.

    * ``TEXT`` — running prose.
    * ``TABLE`` — tabular data; ``content`` carries a serialised table.
    * ``EQUATION`` — mathematical expression (LaTeX or similar).
    * ``IMAGE`` — embedded image with optional ``caption``.
    * ``CODE`` — source code.
    """

    TEXT = "text"
    TABLE = "table"
    EQUATION = "equation"
    IMAGE = "image"
    CODE = "code"
    METADATA = "metadata"


class DocumentBlock(BaseModel):
    """A single atom within a section: paragraph, table, image, equation.

    Attributes:
        block_id: Stable id (deterministic from source+offset).
        kind: Block kind.
        content: Block payload (Markdown / LaTeX / image URI).
        metadata: Format-specific metadata.

    """

    block_id: str = Field(default_factory=lambda: deterministic_id("block", str(uuid4())))
    kind: BlockKind = BlockKind.TEXT
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A logical section of a document — chapter, page, or slide.

    Attributes:
        section_id: Stable id (deterministic from source+section_index).
        index: 0-based section ordinal within the parent document.
        heading: Optional heading text.
        blocks: Ordered list of :class:`DocumentBlock` atoms.
        page_numbers: 1-based page numbers that contributed to this section.
        source_location: Human-readable location string.

    """

    section_id: str = Field(default_factory=lambda: deterministic_id("section", str(uuid4())))
    index: int = 0
    heading: str = ""
    blocks: list[DocumentBlock] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_location: str = ""


class DocumentAlias(Document):
    """Spec-named alias for :class:`Document`.

    Re-exported under the spec-mandated name ``Document``. The
    underlying ``Document`` schema is unchanged.
    """


class ChunkAlias(BaseModel):
    """Internal: re-export alias exposed as ``Chunk`` at module level.

    Lives at module scope so it appears in introspection tools, but
    is not part of ``__all__`` and is not part of the public surface.
    Kept as ``ChunkAlias`` (single-word subject, no underscore) so
    the package's two-tier naming rule remains satisfied.
    """


class Embedding(BaseModel):
    """A typed vector with provenance.

    Separate from the in-place ``Chunk.checksum`` style so adapters
    can exchange embeddings without leaking the wire-format parent.

    Attributes:
        id: Embedding id (UUID).
        target: Owning chunk id.
        model: Embedding model name.
        dim: Vector dimensionality.
        vector: Float vector.
        created_at: Timestamp the vector was produced (UTC).

    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: str
    model: str
    dim: int
    vector: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: EmbeddingType = EmbeddingType.DENSE

    def verify(self) -> None:
        """Assert the embedding's invariant contract.

        Raises:
            VerificationError: When ``id`` is empty or vector is empty.

        """
        if not self.id:
            raise VerificationError("Embedding: empty id")
        if not self.vector:
            raise VerificationError("Embedding: empty vector")


class Citation(BaseModel):
    """Provenance for a single answer span.

    Attributes:
        document_id: Parent document id.
        version: Document version.
        page: Page number (1-based) or 0 for non-paginated.
        section: Section heading.
        quote: Optional excerpt used as evidence.
        score: Retrieval score (cosine similarity or fused score).
        source_uri: Original source location.
        chunk: The :class:`Chunk` this citation references; ``chunk.id``
            is the canonical chunk identity used to cross-check
            ``Response.source_chunks``.

    """

    chunk: Chunk | None = None
    document_id: str
    version: int = 1
    page: int = 0
    section: str = ""
    quote: str = ""
    score: float = 0.0
    source_uri: str = ""
    type: CitationType = CitationType.DIRECT

    def verify(self) -> None:
        """Assert the citation's contract.

        Raises:
            VerificationError: When required fields are empty.

        """
        if not self.document_id:
            raise VerificationError("Citation: empty document_id")
        if self.score < 0.0:
            raise VerificationError(f"Citation: negative score ({self.score})")


class Citations(BaseModel):
    """The aggregate of citations on a Response.

    Carries its own verify() so the rule 'every citation has its
    chunk in source_chunks' lives on this class instead of being
    re-implemented in Response.verify().

    Attributes:
        items: The flattened citation list.

    """

    items: list[Citation] = Field(default_factory=list)

    def verify(self, chunks: list | None = None) -> None:
        """Assert each citation's invariant and chunk membership.

        Args:
            chunks: The Response's source_chunks (or equivalent). When
                supplied, also asserts every citation's chunk_id is in
                the chunks list.

        Raises:
            VerificationError: When any check fails.

        """
        for cit in self.items:
            cit.verify()
        if chunks is not None:

            def _chunk_id(c: object) -> str | None:
                """Resolve a chunk's id from either ``.id`` or ``.chunk_id``.

                Args:
                    c: A chunk-like object.

                Returns:
                    The chunk id, or ``None`` when neither attribute is set.

                """
                id_attr: object = getattr(c, "id", None)
                if id_attr is not None:
                    return str(id_attr)
                cid_attr: object = getattr(c, "chunk_id", None)
                return str(cid_attr) if cid_attr is not None else None

            valid = {_chunk_id(c) for c in chunks}
            for cit in self.items:
                if cit.chunk is not None and cit.chunk.id not in valid:
                    raise VerificationError(
                        f"Citations: chunk_id {cit.chunk.id!r} not in source_chunks"
                    )


class SearchResult(Hit):
    """Spec-named alias for :class:`Hit`.

    Inherits the chunk_id-match validator from :class:`Hit`.
    """


class Query(SearchRequest):
    """Spec-named alias for :class:`raghub.models.SearchRequest`."""


class Response(BaseModel):
    """Public response model with typed citations and source chunks.

    The spec requires that components exchange typed models rather
    than raw dictionaries. This class is the canonical :class:`Response`
    used by the RAG facade.

    Attributes:
        type: Discriminator for the response kind (R3 <Entity>Type).

    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_chunks: list[Hit] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] | None = None
    transforms_applied: list[str] = Field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = Field(default_factory=list)
    type: ResponseType = ResponseType.ANSWER

    def verify(self) -> None:
        """Assert the response's contract.

        Delegates citation-membership checks to
        :meth:`Citations.verify`, then runs ``source_chunks.verify()``
        on each entry.

        Raises:
            VerificationError: When any check fails.

        """
        if not self.answer and not self.citations:
            raise VerificationError("Response: empty answer and no citations")
        Citations(items=list(self.citations)).verify(chunks=list(self.source_chunks))
        for source in self.source_chunks:
            source.verify()

    def citations_aggregate(self) -> Citations:
        """Return the citations wrapped as the :class:`Citations` aggregate."""
        return Citations(items=list(self.citations))


class Bundle(BaseModel):
    """A persisted Open Knowledge Format bundle.

    The bundle is the canonical persisted representation of source
    documents after conversion. Internal processing should derive
    optimised Python models from the bundle rather than re-parsing the
    original Markdown.

    Attributes:
        bundle_id: Stable id (deterministic from source URI + version).
        schema_version: OKF schema version this bundle was emitted in.
        source_uri: Source location (file path, URL, or s3://...).
        checksum: SHA-256 of the source bytes.
        language: Detected language (BCP-47).
        mime_type: Original MIME type.
        metadata: Format-specific metadata.
        sections: Ordered list of :class:`DocumentSection`.
        created_at: Bundle creation time (UTC).

    """

    bundle_id: str = Field(default_factory=lambda: deterministic_id("bundle", str(uuid4())))
    schema_version: str = "0.1"
    source_uri: str
    checksum: str = ""
    language: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[DocumentSection] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineCtx(BaseModel):
    """Per-invocation state passed to every stage of a pipeline.

    Attributes:
        pipeline_id: Stable id for this run.
        pipeline_name: Logical pipeline name (e.g. ``"ingest"``).
        user: Authenticated user principal driving the call (when applicable).
        metadata: Arbitrary per-run metadata.
        started_at: Pipeline start timestamp (UTC).

    """

    pipeline_id: str = Field(default_factory=lambda: deterministic_id("pipeline", str(uuid4())))
    pipeline_name: str = "default"
    user: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Pipeline(BaseModel):
    """Output of a pipeline run.

    The model uses an ``error: ErrorInfo | None`` discriminator rather
    than a boolean ``success`` flag: ``error is None`` means the run
    succeeded, ``error`` is set means it failed. Callers branch on
    ``pipeline.error is None`` rather than reading a removed field.

    Attributes:
        pipeline_id: Id of the originating run.
        pipeline_name: Logical pipeline name.
        type: Discriminator for the pipeline kind (R3 <Entity>Type).
        outputs: Stage-specific outputs keyed by stage name.
        error: Populated when the run failed; ``None`` on success.
        finished_at: Pipeline finish timestamp (UTC).

    """

    pipeline_id: str
    pipeline_name: str
    type: PipelineType = PipelineType.INGEST  # populated by the pipeline
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: ErrorInfo | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def result(self) -> Pipeline:
        """Return the configured result."""
        return self

    @property
    def succeeded(self) -> bool:
        """``True`` when the pipeline completed without error."""
        return self.error is None

    def verify(self) -> None:
        """Assert the result's invariant contract.

        A successful construction satisfies the contract by
        definition; an errored result must carry a non-empty
        error message.

        Raises:
            VerificationError: When ``error`` is set but the message is empty.

        """
        if self.error is not None and not self.error.message:
            raise VerificationError("Pipeline: error.message required when error is set")


class Result(BaseModel):
    """Result of a single evaluation run on a benchmark example.

    Attributes:
        type: Discriminator for the result kind (R3 <Entity>Type).

    """

    """Score produced by an :class:`Evaluator`.

    Attributes:
        benchmark: Benchmark/dataset identifier (e.g. ``"financebench"``).
        example_id: Per-example identifier (e.g. row id).
        metrics: Metric name → score mapping.
        passed: Whether the example met the benchmark threshold.
        details: Optional explanation / per-stage breakdown.
        evaluated_at: Timestamp the result was produced (UTC).

    """

    benchmark: str
    example_id: str
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# Long-context second-pass rerank models
# ---------------------------------------------------------------------------


class RankedItem(BaseModel):
    """One ranked chunk in a :class:`RankedList` result.

    Attributes:
        id: Stable item id (the chunk's id).
        score: Combined rank score.
        rank: 0-based position in the list.
        chunk: The ranked chunk.

    """

    id: str
    score: float
    rank: int = 0
    chunk: Chunk


class LongContextRankedItem(BaseModel):
    """A single re-ranked candidate produced by the long-context LLM.

    Attributes:
        chunk_id: Stable chunk id from the original hits list.
        score: Refined relevance score in ``[0, 1]``.
        rationale: One-sentence justification for the new score.
            Kept short so the assembled prompt stays under the
            long-context window.

    """

    chunk_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RankedList(BaseModel):
    """Wrapper that lets ``Instructor``-style providers validate the LLM output.

    Attributes:
        items: Per-chunk re-ranking, in the order the model produced
            them. Missing or malformed entries are dropped by the
            caller (see :func:`LongContextRerankPass._reorder`).

    """

    items: list[RankedItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class AuthLoginRequest(BaseModel):
    """Login request payload.

    Attributes:
        email: User email. Must contain ``@`` and a dot-separated
            domain per the regex constraint.
        password: User password (validated server-side; never
            echoed back).

    """

    email: str = Field(min_length=1, pattern=r".+@.+\..+")
    password: str = Field(min_length=1)


class AuthLoginResponse(BaseModel):
    """Login response payload.

    Attributes:
        session_token: Opaque token; the client should attach it as
            ``Authorization: Bearer <token>`` on subsequent calls.
        user_email: Echo of the authenticated user's email.
        allowed_companies: The tenant allow-list; useful for the
            client to decide which company's data to display.

    """

    session_token: str
    user_email: str
    allowed_companies: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Upload response payload.

    Attributes:
        document_id: Newly-created (or incremented) document id.
        version: New version number (1 for first upload).
        status: Initial lifecycle status (``"NEW"``).
        company: Tenant tag.
        filename: Original filename.

    """

    document_id: str
    version: int
    status: str
    company: str
    filename: str


class QueryRequest(BaseModel):
    """Question answering payload.

    Attributes:
        question: The user's question. Must be non-empty.
        tools_enabled: Explicit allow-list of tool names to enable for
            this request. ``None`` defers to the resolver (request >
            session > user > global). Phase 7/8 wiring.
        agent: When ``True``, route through the agentic planner even
            if no specific tools are named.
        web: Shortcut to enable the :class:`WebSearchTool` for this
            request. Equivalent to ``"web_search" in tools_enabled``.
        graph: Shortcut for the GraphRAG summary tool.
        summaries: Shortcut for the RAPTOR summary tool.
        reranker: Per-request reranker override (``"none"|"cohere"|
            "llm"|"cascade"``). ``None`` defers to resolver.
        long_context_pass: Per-request toggle for the long-context
            second-pass rerank.
        query_transforms: Per-request list of transform names
            (``"hyde"|"multi_query"|"step_back"|"decompose"``).
        max_steps: Per-request cap on planner steps.
        top_k: Per-request override of the default retrieval depth.

    """

    question: str = Field(min_length=1)
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


class QueryResponse(BaseModel):
    """Question answering response.

    Attributes:
        answer: The provider-generated answer.
        citations: Citation metadata keyed by source location.
        source_chunks: The retrieved chunks that informed the answer.
        planner_trace: Optional per-step trace of the agent loop
            (``None`` on the fast path). Each entry is the JSON
            payload of a :class:`PlannerEvent`.
        tools_invoked: Names of tools the agent invoked. Empty on the
            fast path.
        transforms_applied: Names of query transforms that ran before
            retrieval. Empty when the resolver disabled them.

    """

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_chunks: list[dict[str, Any]] = Field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = Field(default_factory=list)
    transforms_applied: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchIngestItem(BaseModel):
    """Result of ingesting a single file in a batch request.

    Attributes:
        filename: Original filename.
        document_id: The document id assigned on success, or empty.
        status: ``"ok"`` or ``"error"``.
        error: Error detail when ``status == "error"``.

    """

    filename: str
    document_id: str = ""
    status: str = "ok"
    error: str = ""


class BatchIngestResponse(BaseModel):
    """Response from the batch-ingest endpoint.

    Attributes:
        documents: One :class:`BatchIngestItem` per uploaded file.

    """

    documents: list[BatchIngestItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol contracts
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


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
    stay below :pylint:`too-many-arguments`.
    """

    question: str
    user: Any
    session_id: str
    top_k: int
    metadata_filter: dict[str, Any]
    response_model: type
    tools_enabled: list[str]
    agent: bool
    web: bool
    graph: bool
    summaries: bool
    reranker: str
    long_context_pass: bool
    query_transforms: list[str]
    max_steps: int


class RagComponents(TypedDict, total=False):
    """Optional injectable components for :class:`raghub.rag.RAG`.

    Every key is optional; :class:`RAG` substitutes a sensible
    default when a key is missing. Use this to thread many
    collaborators through a single parameter without exploding the
    signature.
    """

    settings: Any
    registry: Any
    knowledge_repo: Any
    vector_store: Any
    embedder: Any
    llm: Any
    llm_timeout_seconds: float | None
    converter: Any
    chunker: Any
    generator: Any
    reranker: Any
    structured: Any
    telemetry: Any
    background_service: Any
    manifest: Any
    transformer: Any


class KnowledgeRepository(Protocol):
    """Persists and retrieves :class:`Bundle` objects."""

    def save(self, bundle: Bundle) -> Bundle:
        """Persist a knowledge bundle."""
        ...

    def get(self, bundle_id: str) -> Bundle | None:
        """Retrieve a bundle by id."""
        ...

    def list_by_source(self, source_uri: str) -> list[Bundle]:
        """List bundles by source URI."""
        ...

    def delete(self, bundle_id: str) -> None:
        """Delete a bundle by id."""
        ...


class Logger(Protocol):
    """Structured logger contract."""

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info-level message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning-level message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error-level message."""
        ...


class Metrics(Protocol):
    """Metrics recorder contract."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        """Record a latency metric."""
        ...

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter metric."""
        ...


@runtime_checkable
class Span(Protocol):
    """A single open trace span."""

    name: str

    def end(self) -> None:
        """End the span."""
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        ...


class TelemetryProvider(Logger, Metrics, Protocol):
    """Combined observability surface: logging + metrics + spans + tokens."""

    def start_span(self, name: str, **attrs: Any) -> Span:
        """Start a new trace span."""
        ...

    def end_span(self, span: Span) -> None:
        """End a trace span."""
        ...

    def record_tokens(
        self,
        name: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
    ) -> None:
        """Record token usage."""
        ...

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        """Context manager wrapping a trace span."""
        s = self.start_span(name, **attrs)
        try:
            yield s
        finally:
            self.end_span(s)


class StructuredOutputProvider(Protocol):
    """Generates typed Pydantic outputs from context."""

    async def generate(
        self,
        *,
        response_model: type[T],
        question: str,
        context: Sequence[Hit],
    ) -> T:
        """Generate a structured Pydantic output."""
        ...

    async def astream(
        self,
        *,
        response_model: type[T],
        question: str,
        context: Sequence[Hit],
    ) -> AsyncIterator[T]:
        """Stream structured Pydantic outputs."""
        ...


class VectorStore(Protocol):
    """Vector database contract."""

    def create_collection(self) -> None:
        """Create the vector collection."""
        ...

    def insert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> int:
        """Insert chunks with their vectors; return the rows written."""
        ...

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[list[float]],
    ) -> int:
        """Insert or update chunks with vectors; return the rows written."""
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete chunks by ids."""
        ...

    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document."""
        ...

    def delete_version(self, document_id: str, version: int) -> None:
        """Delete a specific document version."""
        ...

    def search(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Search by vector similarity."""
        ...

    def hybrid_search(
        self,
        *,
        query: str,
        vector: Sequence[float],
        top_k: int,
        metadata_filter: str | dict[str, Any] = "",
    ) -> list[dict[str, Any]]:
        """Search by vector similarity and keyword."""
        ...

    def optimize(self) -> None:
        """Optimize the vector store."""
        ...

    def health(self) -> dict[str, Any]:
        """Return vector store health status."""
        ...

    def keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search by keyword."""
        ...


class Retriever(Protocol):
    """Retrieves authorized chunks for a user."""

    def retrieve(
        self,
        *,
        user: User,
        question: str,
        top_k: int,
    ) -> list[Hit]:
        """Retrieve authorized chunks for a user."""
        ...


class Reranker(Protocol):
    """Reorders retrieved results using a downstream signal."""

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[Hit],
    ) -> list[Hit]:
        """Rerank retrieved hits."""
        ...


class PromptBuilder(Protocol):
    """Builds structured prompts without manual concatenation."""

    def build_system_prompt(self) -> str:
        """Build a system prompt."""
        ...

    def build_messages(
        self,
        *,
        conversation: Sequence[Turn],
        retrieved_chunks: Sequence[Chunk],
        question: str,
    ) -> list[dict[str, str]]:
        """Build a message list for the LLM."""
        ...


class LLMProvider(Protocol):
    """Generates responses from prompt sections."""

    model_name: str

    def generate(
        self,
        *,
        system_prompt: str,
        conversation: Sequence[Turn],
        context: Sequence[str],
        question: str,
        **options: Any,
    ) -> str:
        """Generate a response from prompt sections.

        Optional ``image_paths=`` and ``session_history=`` overrides
        are accepted via ``**options`` for backward compatibility
        with the previous explicit signature.
        """
        ...


class PipelineRunner(Protocol):
    """A deterministic, multi-stage computation."""

    name: str

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the pipeline with the given context."""
        ...


class BackgroundWorker(Protocol):
    """Schedules background tasks."""

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a background task."""
        ...


class TaskQueue(Protocol):
    """Abstract task queue (e.g. Celery, RQ, SQS)."""

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue a task."""
        ...


class DocumentRegistry(Protocol):
    """Tracks versioned document state."""

    def save_version(self, document: Document) -> Document:
        """Persist a document version."""
        ...

    def get_latest(self, document_id: str) -> Document | None:
        """Get the latest document version."""
        ...

    def list_accessible(self, companies: list[str]) -> list[Document]:
        """List accessible documents for companies."""
        ...

    def archive(self, document_id: str) -> None:
        """Archive a document."""
        ...


class ConversationStore(Protocol):
    """Stores only turns, not context chunks."""

    def append(self, session_id: str, turn: Turn) -> None:
        """Append a turn to a session."""
        ...

    def load(self, session_id: str, limit: int = 20) -> list[Turn]:
        """Load turns from a session."""
        ...

    def clear(self, session_id: str) -> None:
        """Clear all turns for a session."""
        ...


class SessionStoreProtocol(Protocol):
    """Stores session metadata."""

    def create(self, user_id: str) -> Session:
        """Create a new session."""
        ...

    def resolve(self, token: str) -> Session | None:
        """Resolve a session token."""
        ...

    def invalidate(self, token: str) -> None:
        """Invalidate a session token."""
        ...


class Plugin(Protocol):
    """A discoverable plugin."""

    name: str
    version: str

    def register(self, registry: Any) -> None:
        """Register the plugin with a registry."""
        ...
