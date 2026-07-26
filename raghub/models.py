"""Domain, canonical, and transport Pydantic models.

This module defines the full Pydantic model surface used by RAGHub,
consolidated from three previously-separate files:

* **Domain** — runtime domain types (chunks, documents, users,
  sessions, turns, search results, lifecycle enums).
* **Canonical** — spec-mandated aliases (``Document`` /
  ``Chunk`` / ``SearchResult`` / ``Query`` / ``Response``) plus
  the new higher-level types (``Citation``, ``DocumentSection``,
  ``DocumentBlock``, ``KnowledgeBundle``, ``PipelineContext``,
  ``PipelineResult``, ``Embedding``, ``EvaluationResult``).
* **API** — the FastAPI request/response wire types.
* **Long-context** — ``RankedItem`` / ``RankedList`` for the
  second-pass LLM rerank.

Mapping (canonical ↔ domain):

* ``Document`` ↔ ``DocumentRecord``
* ``Chunk`` ↔ ``ChunkRecord``
* ``SearchResult`` ↔ ``RetrievalHit``
* ``Query`` ↔ ``SearchRequest``
* ``Response`` ↔ ``SearchResponse``

The :func:`deterministic_id` helper builds short stable ids for
newly-constructed Pydantic models.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Identity helpers
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
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:clamped]


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class DocumentLifecycleStatus(str, Enum):
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


class Visibility(str, Enum):
    """Document visibility levels.

    * ``PRIVATE``: only the owner can read.
    * ``ORGANIZATION``: any authenticated user in the same tenant.
    * ``PUBLIC``: any authenticated user, regardless of tenant.
    """

    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class Classification(str, Enum):
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


class UserPrincipal(BaseModel):
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

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False
    tool_settings: dict[str, Any] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
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


class SessionRecord(BaseModel):
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

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    last_seen_at: datetime
    history: list[ConversationTurn] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    """Document data transfer object.

    Attributes:
        document_id: Stable document id.
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
        chunk_ids: Chunk ids produced by the latest ingest.
        error: Optional error message when ``status == FAILED``.
    """

    document_id: str = Field(default_factory=lambda: str(uuid4()))
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
    chunk_ids: list[str] = Field(default_factory=list)
    error: str | None = None


DocumentVersion = DocumentRecord


class ChunkRecord(BaseModel):
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

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
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
    hash: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    """A retrieved chunk with score and metadata.

    Attributes:
        chunk_id: Id of the underlying :class:`ChunkRecord`.
        score: Cosine-similarity score reported by the vector store.
        chunk: The full chunk metadata.
    """

    chunk_id: str
    score: float
    chunk: ChunkRecord


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
        source_chunks: The :class:`ChunkRecord` objects that
            contributed to the answer.
        metadata: Provider- and pipeline-specific metadata.
    """

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_chunks: list[ChunkRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


SessionRecord.model_rebuild()


# ---------------------------------------------------------------------------
# Canonical spec-named models
# ---------------------------------------------------------------------------


class BlockKind(str, Enum):
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


class Document(DocumentRecord):
    """Spec-named alias for :class:`DocumentRecord`.

    Re-exported under the spec-mandated name ``Document``. The
    underlying ``DocumentRecord`` schema is unchanged.
    """


class Chunk(ChunkRecord):
    """Spec-named alias for :class:`ChunkRecord`."""


class Embedding(BaseModel):
    """A typed vector with provenance.

    Separate from the in-place ``ChunkRecord.hash`` style so adapters
    can exchange embeddings without leaking the wire-format parent.

    Attributes:
        chunk_id: Owning chunk id.
        model: Embedding model name.
        dim: Vector dimensionality.
        vector: Float vector.
        created_at: Timestamp the vector was produced (UTC).
    """

    chunk_id: str
    model: str
    dim: int
    vector: list[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(BaseModel):
    """Provenance for a single answer span.

    Attributes:
        chunk_id: Underlying chunk id.
        document_id: Parent document id.
        version: Document version.
        page: Page number (1-based) or 0 for non-paginated.
        section: Section heading.
        quote: Optional excerpt used as evidence.
        score: Retrieval score (cosine similarity or fused score).
        source_uri: Original source location.
    """

    chunk_id: str
    document_id: str
    version: int = 1
    page: int = 0
    section: str = ""
    quote: str = ""
    score: float = 0.0
    source_uri: str = ""


class SearchResult(BaseModel):
    """Spec-named alias for :class:`RetrievalHit`."""

    chunk_id: str
    score: float
    chunk: ChunkRecord


class Query(SearchRequest):
    """Spec-named alias for :class:`raghub.models.SearchRequest`."""


class Response(BaseModel):
    """Public response model with typed citations and source chunks.

    The spec requires that components exchange typed models rather
    than raw dictionaries. This class is the canonical :class:`Response`
    used by the RAG facade; the legacy
    :class:`raghub.models.SearchResponse` is retained for
    backwards compatibility.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    source_chunks: list[SearchResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structured: dict[str, Any] | None = None
    transforms_applied: list[str] = Field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = Field(default_factory=list)


class KnowledgeBundle(BaseModel):
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


class PipelineContext(BaseModel):
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


class PipelineResult(BaseModel):
    """Output of a successful pipeline run.

    Attributes:
        pipeline_id: Id of the originating run.
        pipeline_name: Logical pipeline name.
        success: Whether the pipeline completed without error.
        outputs: Stage-specific outputs keyed by stage name.
        error: Error message when ``success`` is ``False``.
        finished_at: Pipeline finish timestamp (UTC).
    """

    pipeline_id: str
    pipeline_name: str
    success: bool = True
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationResult(BaseModel):
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
            "bge"|"llm"|"cascade"``). ``None`` defers to resolver.
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
    citations: list[dict] = Field(default_factory=list)
    source_chunks: list[dict] = Field(default_factory=list)
    planner_trace: list[dict] | None = None
    tools_invoked: list[str] = Field(default_factory=list)
    transforms_applied: list[str] = Field(default_factory=list)


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