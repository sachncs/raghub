"""Discriminator and lifecycle enums for the raghub domain models.

This module was extracted from ``raghub.models`` to keep the model
file under a more manageable size. Every enum that previously lived
in ``raghub.models`` is re-exported here for backward compatibility.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class DocumentLifecycleStatus(StrEnum):
    """Document lifecycle states.

    Legal transitions are validated by
    :class:`raghub.core.DocumentStateMachine`; see its
    docstring for the full transition table. ``Archived`` and
    ``Failed`` are terminal.
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
    """Document visibility levels.

    * ``Private``: only the owner can read.
    * ``Organization``: any authenticated user in the same tenant.
    * ``Public``: any authenticated user, regardless of tenant.
    """

    Private = "private"
    Organization = "organization"
    Public = "public"


class Classification(StrEnum):
    """Simplified data classification levels.

    Used by RBAC filters and the redaction layer to gate sensitive
    content from users without the appropriate clearance.
    """

    Internal = "internal"
    Confidential = "confidential"
    Restricted = "restricted"


# ---------------------------------------------------------------------------
# Discriminator enums (one per entity class; R7: >= 2 values)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Shared lifecycle enums (R3 single-word)
# ---------------------------------------------------------------------------


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
    """Coarse kinds of :class:`DocumentBlock`.

    * ``TEXT`` — running prose.
    * ``TABLE`` — tabular data; ``content`` carries a serialised table.
    * ``EQUATION`` — mathematical expression (LaTeX or similar).
    * ``IMAGE`` — embedded image with optional ``caption``.
    * ``CODE`` — source code.
    """

    Text = "text"
    Table = "table"
    Equation = "equation"
    Image = "image"
    Code = "code"
    Metadata = "metadata"


__all__ = [
    "Access",
    "BlockKind",
    "BlockType",
    "BundleType",
    "ChunkType",
    "CitationType",
    "Class",
    "Classification",
    "DocType",
    "DocumentLifecycleStatus",
    "EmbeddingType",
    "EventType",
    "HitType",
    "JobType",
    "ManifestType",
    "PipelineType",
    "RankType",
    "ResponseType",
    "ResultType",
    "SectionType",
    "SessionKind",
    "State",
    "UserKind",
    "Visibility",
]
