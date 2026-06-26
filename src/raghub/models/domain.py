"""Core domain entities for the Dynamic RAG platform."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentLifecycleStatus(str, Enum):
    """Document lifecycle states."""

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
    """Document visibility levels."""

    PRIVATE = "private"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class Classification(str, Enum):
    """Simplified data classification levels."""

    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class UserPrincipal(BaseModel):
    """Authenticated user principal."""

    user_id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    allowed_companies: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    is_admin: bool = False


class SessionRecord(BaseModel):
    """Session metadata and isolated conversational history."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    token: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    last_seen_at: datetime
    history: list["ConversationTurn"] = Field(default_factory=list)


class DocumentVersion(BaseModel):
    """Versioned document aggregate."""

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
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ChunkRecord(BaseModel):
    """Chunk metadata stored alongside the vector."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    version: int
    page: int
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
    """Single question-answer turn stored in session memory."""

    question: str
    answer: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    """A retrieved chunk with score and metadata."""

    chunk_id: str
    score: float
    chunk: ChunkRecord


class SearchRequest(BaseModel):
    """Search input to the retrieval pipeline."""

    user_id: str
    question: str
    session_id: str
    top_k: int = 5


class SearchResponse(BaseModel):
    """Search output from the retrieval pipeline."""

    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_chunks: list[ChunkRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


SessionRecord.model_rebuild()

