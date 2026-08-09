"""Domain entity models (documents, chunks, retrieval results).

Includes the canonical :class:`Document`, :class:`Chunk`, :class:`Hit`,
:class:`SearchResult`, :class:`Embedding`, :class:`Citation`,
:class:`Citations`, :class:`DocumentBlock`, :class:`DocumentSection`,
:class:`Bundle`, and supporting alias types.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from raghub.errors import VerificationError
from raghub.models.enums import (
    Classification,
    DocumentLifecycleStatus,
    Visibility,
)
from raghub.types import JSONValue

__all__ = [
    "Bundle",
    "Chunk",
    "Citation",
    "Citations",
    "Document",
    "DocumentAlias",
    "DocumentBlock",
    "DocumentSection",
    "Embedding",
    "Hit",
    "SearchResult",
]


class Document(BaseModel):
    """A versioned document with its own metadata."""

    id: str
    version: int
    checksum: str
    created_at: datetime
    updated_at: datetime
    owner: str
    organization: str
    department: str = ""
    tags: list[str] = Field(default_factory=list)
    classification: Classification = Classification.Internal
    visibility: Visibility = Visibility.Organization
    status: DocumentLifecycleStatus = DocumentLifecycleStatus.New
    filename: str = ""
    file_type: str = ""
    mime_type: str = ""
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    error: str | None = None

    def verify(self) -> None:
        """Assert the document's invariant contract.

        Raises:
            VerificationError: When ``id``, ``owner``, ``organization``,
                or ``checksum`` is empty, or when ``updated_at`` is
                before ``created_at``.

        """
        if not self.id:
            raise VerificationError("Document: empty id")
        if not self.owner:
            raise VerificationError("Document: empty owner")
        if not self.organization:
            raise VerificationError("Document: empty organization")
        if not self.checksum:
            raise VerificationError("Document: empty checksum")
        if self.updated_at < self.created_at:
            raise VerificationError(
                "Document: updated_at must be >= created_at"
            )


class Chunk(BaseModel):
    """A retrieval unit carrying its own metadata and checksum."""

    id: str
    document_id: str
    version: int = 0
    page: int = 0
    source_location: str = ""
    section: str = ""
    company: str
    owner: str
    department: str = ""
    classification: Classification = Classification.Internal
    created_at: datetime
    embedding_model: str = ""
    hash: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def verify(self) -> None:
        """Assert the chunk's invariant contract.

        Raises:
            VerificationError: When ``id``, ``document_id``,
                ``company``, ``owner``, ``hash``, or ``text`` is empty.

        """
        if not self.id:
            raise VerificationError("Chunk: empty id")
        if not self.document_id:
            raise VerificationError("Chunk: empty document_id")
        if not self.company:
            raise VerificationError("Chunk: empty company")
        if not self.owner:
            raise VerificationError("Chunk: empty owner")
        if not self.hash:
            raise VerificationError("Chunk: empty hash")
        if not self.text:
            raise VerificationError("Chunk: empty text")


class Hit(BaseModel):
    """A single search result with score and chunk payload."""

    score: float
    chunk: Chunk


class SearchResult(Hit):
    """Canonical alias for :class:`Hit`. Same shape; clearer name in API
    responses where every entry is a *result* rather than a *hit*."""


class Embedding(BaseModel):
    """A chunk-embedding record indexed by chunk_id."""

    chunk_id: str
    vector: list[float]
    model: str

    def verify(self) -> None:
        """Assert the embedding's invariant contract.

        Raises:
            VerificationError: When ``chunk_id`` is empty or
                ``vector`` is empty.

        """
        if not self.chunk_id:
            raise VerificationError("Embedding: empty chunk_id")
        if not self.vector:
            raise VerificationError("Embedding: empty vector")


class Citation(BaseModel):
    """A citation entry pointing back to a chunk."""

    document_id: str
    version: int
    page: int = 0
    section: str = ""
    chunk_id: str


class Citations(BaseModel):
    """List wrapper for a :class:`Citation` array (typed for API responses)."""

    items: list[Citation]


class DocumentBlock(BaseModel):
    """A single atom within a section: paragraph, table, image, equation.

    Attributes:
        block_id: Stable id (deterministic from source+offset).
        kind: Block kind.
        content: Block payload (Markdown / LaTeX / image URI).
        metadata: Format-specific metadata.

    """

    block_id: str
    kind: Any = None  # BlockKind: concrete type comes from models.enums
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSection(BaseModel):
    """A section of a parsed document: a heading with N blocks."""

    section_id: str
    index: int
    heading: str
    blocks: list[DocumentBlock] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    source_location: str = ""


class DocumentAlias(Document):
    """Deprecated alias for :class:`Document` kept for backward compat."""


class ChunkAlias(BaseModel):
    """Deprecated alias for :class:`Chunk` kept for backward compat.

    New code should use :class:`Chunk` directly.
    """


class Bundle(BaseModel):
    """Output of an ingest run: a document plus its sections."""

    bundle_id: str
    schema_version: str
    source_uri: str
    checksum: str
    language: str = ""
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    sections: list[DocumentSection] = Field(default_factory=list)
    error: Any | None = None
    finished_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
