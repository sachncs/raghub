"""Core domain entities for the Dynamic RAG platform.

This module declares the project's Pydantic models: lifecycle
states, visibility levels, classifications, user principals, session
records, document records, chunk records, conversation turns, and
search request/response types.

The models double as the database schema (when persisted via the
SQLite repositories in :mod:`raghub.repositories`) and as the wire
format for the FastAPI surface (when serialised via
:mod:`raghub.models.api`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentLifecycleStatus(str, Enum):
    """Document lifecycle states.

    Legal transitions are validated by
    :class:`raghub.core.document_state.DocumentStateMachine`; see its
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


class UserPrincipal(BaseModel):
    """Authenticated user principal.

    Attributes:
        user_id: Stable opaque user id.
        email: Login email; used as the principal's display name.
        allowed_companies: Tenant allow-list. Empty for admins
            (admins bypass the company filter).
        allowed_groups: Group memberships for finer-grained RBAC.
        is_admin: ``True`` for platform-wide admins.
    """

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False


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
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    last_seen_at: datetime
    history: list["ConversationTurn"] = Field(default_factory=list)


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    embedding_model: str = ""
    hash: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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