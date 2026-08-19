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

Serialization API
-----------------

Every dataclass inherits from :class:`Snap`, which provides a small,
uniform ``dump / copy / verify`` surface backed by ``dataclasses``:

* :meth:`Snap.dump` returns a plain ``dict`` (or a JSON-safe ``dict``
  when ``mode="json"``).
* :meth:`Snap.copy` is a thin wrapper over :func:`dataclasses.replace`.
* :classmethod:`Snap.validate` constructs an instance from a dict,
  coercing primitives (ISO-8601 strings → ``datetime``, enum names →
  ``StrEnum``) and nested dicts into their dataclass types.
* :meth:`Snap.verify` re-runs the model's ``__post_init__`` invariants
  (frozen dataclasses guarantee no silent drift, so this is a
  defensive check used by stores before persisting).

The :func:`deterministic_id` helper builds short stable ids for
newly-constructed dataclasses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import typing
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast, get_args, get_origin
from uuid import uuid4

from raghub.errors import VerificationError

_TYPE_HINTS_CACHE: dict[type, dict[str, Any]] = {}


def resolve_hints(cls: type) -> dict[str, Any]:
    """Return ``cls``'s annotations, with ``from __future__`` strings resolved."""
    cached = _TYPE_HINTS_CACHE.get(cls)
    if cached is not None:
        return cached
    resolved = typing.get_type_hints(cls)
    _TYPE_HINTS_CACHE[cls] = resolved
    return resolved


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
    joined = "\x1f".join(parts).encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(joined).hexdigest()
    return digest[:clamped]


# ---------------------------------------------------------------------------
# Secret value type (defined after Snap so it can inherit the mixin)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Snap mixin: dump / copy / validate / verify
# ---------------------------------------------------------------------------


_JSON_SENTINELS: tuple[Any, ...] = (None,)


def json_safe(value: Any) -> Any:  # noqa: PLR0911 - one return per coercion branch
    """Recursively convert a value into a JSON-safe primitive.

    Each branch handles a different primitive / container shape; the
    function is intentionally written with one ``return`` per branch
    so the structure stays obvious in performance-sensitive serialise
    paths.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(cast(Any, value)))
    return value


def coerce_value(field_type: Any, value: Any) -> Any:  # noqa: PLR0911 - one return per coercion branch
    """Coerce ``value`` into ``field_type`` where possible.

    Handles the small set of coercions needed by RAGHub models:
    ISO-8601 strings → ``datetime``, enum names → ``StrEnum``,
    nested ``dict`` → dataclass, ``list[T]`` → ``list[T]``.

    Each branch returns immediately to keep the coercion logic
    readable; the function is intentionally structured as a
    discriminated ladder instead of a match statement because the
    cascading short-circuits benefit from the early return.
    """
    if value is None:
        return None
    origin = get_origin(field_type)
    args = get_args(field_type)
    if origin is list and args:
        return [coerce_value(args[0], item) for item in value]
    if origin is dict and args and value is not None:
        k_type, v_type = args
        return {k_type(k): coerce_value(v_type, v) for k, v in value.items()}
    if (
        isinstance(field_type, type)
        and dataclasses.is_dataclass(field_type)
        and isinstance(value, dict)
    ):
        coerced: dict[str, Any] = {}
        for f in fields(field_type):
            coerced[f.name] = (
                coerce_value(f.type, value.get(f.name)) if f.name in value else value.get(f.name)
            )
        return field_type(**coerced)
    if field_type is datetime and isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(field_type, type) and issubclass(field_type, StrEnum) and isinstance(value, str):
        return field_type(value)
    return value


class Snap:
    """Mixin that gives every model a ``dump / copy / verify`` API.

    Every frozen dataclass in this module inherits from :class:`Snap`,
    so callers can treat them uniformly:

    * :meth:`dump` — serialise the model to a ``dict``.
    * :meth:`copy` — return a shallow copy with selected fields updated.
    * :meth:`verify` — re-run ``__post_init__`` invariants (no-op when
      the model is unchanged, defensive check before persistence).
    * :meth:`validate` — build an instance from a serialised ``dict``.

    The mixin never mutates ``self``. Construction goes through the
    dataclass ``__init__``; coercion logic lives in :func:`coerce_value`.
    """

    def dump(self, mode: str = "default") -> dict[str, Any]:
        """Serialise the model to a ``dict``.

        Args:
            mode: ``"default"`` returns the raw :func:`dataclasses.asdict`
                representation. ``"json"`` recursively converts
                ``datetime`` → ISO-8601 strings and ``StrEnum`` → their
                string values, so the result is JSON-serialisable.

        Returns:
            A ``dict`` representation of the model.

        """
        if mode == "json":
            return cast(dict[str, Any], json_safe(dataclasses.asdict(cast(Any, self))))
        return cast(dict[str, Any], dataclasses.asdict(cast(Any, self)))

    def copy(self, **updates: Any) -> Any:
        """Return a shallow copy with the given fields replaced.

        Args:
            **updates: Field overrides applied via
                :func:`dataclasses.replace`.

        Returns:
            A new instance of the concrete model.

        """
        return dataclasses.replace(cast(Any, self), **updates)

    def verify(self) -> None:
        """Re-run the model's invariants.

        Default :meth:`Snap.verify` re-invokes ``__post_init__``.
        Models that already run their invariants there should rely
        on the inherited implementation; models with body-only
        verification can override.
        """
        self.__post_init__()  # type: ignore[attr-defined]

    @classmethod
    def validate(cls, data: dict[str, Any]) -> Any:
        """Build an instance from a serialised ``dict``.

        Args:
            data: The serialised representation produced by
                :meth:`dump` (in either ``"default"`` or ``"json"``
                ``mode``).

        Returns:
            A new instance of ``cls`` with invariants re-checked by
            ``__post_init__``.

        """
        coerced: dict[str, Any] = {}
        hints = resolve_hints(cls)
        for f in fields(cast(Any, cls)):
            if f.name not in data:
                continue
            coerced[f.name] = coerce_value(hints.get(f.name, f.type), data[f.name])
        return cls(**coerced)


# ---------------------------------------------------------------------------
# Secret value type
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Secret(Snap):
    """String value whose ``__repr__`` is masked.

    Used in place of Pydantic's ``SecretStr`` for credentials read
    from the environment so accidentally logging a settings dict never
    reveals the value. Equality and hashing compare the underlying
    string; rendering always returns ``Secret('***')``.

    Attributes:
        value: The cleartext credential.

    """

    value: str = ""

    def __repr__(self) -> str:
        return "Secret('***')"

    def __str__(self) -> str:
        return "***"

    def __bool__(self) -> bool:
        return bool(self.value)


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ErrorInfo(Snap):
    """Structured error information shared across pipeline outputs."""

    kind: str = ""
    message: str = ""
    cause: str | None = None


# ---------------------------------------------------------------------------
# Identity domain
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class User(Snap):
    """Authenticated user principal."""

    id: str = field(default_factory=lambda: str(uuid4()))
    email: str = ""
    allowed_companies: list[str] = field(default_factory=list)
    allowed_groups: list[str] = field(default_factory=list)
    is_admin: bool = False
    tool_settings: dict[str, Any] = field(default_factory=dict)
    type: UserKind = UserKind.Standard

    def __post_init__(self) -> None:
        """Validate that required identity fields are non-empty."""
        if not self.id:
            raise VerificationError("User: empty id")
        if not self.email:
            raise VerificationError("User: empty email")


@dataclass(slots=True, frozen=True)
class Turn(Snap):
    """Single question-answer turn stored in session memory."""

    question: str = ""
    answer: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Session(Snap):
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
        """Validate that required session identity fields are non-empty."""
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
class Document(Snap):
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
        """Validate the document id and the FAILED-state error payload."""
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
class Chunk(Snap):
    """Chunk metadata stored alongside the vector."""

    @classmethod
    def unsafe(  # noqa: PLR0913 - factory intentionally exposes every Chunk field
        cls,
        *,
        id: str = "",
        document_id: str = "",
        version: int = 0,
        page: int = 0,
        source_location: str = "",
        section: str = "",
        company: str = "",
        owner: str = "",
        department: str = "",
        classification: Classification = Classification.Internal,
        created_at: datetime = field(default_factory=lambda: datetime.now(UTC)),  # noqa: B008 - dataclass field is the documented default-factory pattern
        embedding_model: str = "",
        checksum: str = "",
        text: str = "",
        metadata: dict[str, Any] | None = None,
        type: ChunkType = ChunkType.Text,
        tenant_id: str | None = None,
    ) -> Chunk:
        """Construct a :class:`Chunk` skipping :py:meth:`__post_init__` validation.

        Used by tests that intentionally build a chunk whose checksum
        no longer matches ``sha256(text)`` so that :meth:`verify`
        can re-raise the underlying :class:`VerificationError`.
        """
        if metadata is None:
            metadata = {}
        instance = cls.__new__(cls)
        object.__setattr__(instance, "id", id)
        object.__setattr__(instance, "document_id", document_id)
        object.__setattr__(instance, "version", version)
        object.__setattr__(instance, "page", page)
        object.__setattr__(instance, "source_location", source_location)
        object.__setattr__(instance, "section", section)
        object.__setattr__(instance, "company", company)
        object.__setattr__(instance, "owner", owner)
        object.__setattr__(instance, "department", department)
        object.__setattr__(instance, "classification", classification)
        object.__setattr__(instance, "created_at", created_at)
        object.__setattr__(instance, "embedding_model", embedding_model)
        object.__setattr__(instance, "checksum", checksum)
        object.__setattr__(instance, "text", text)
        object.__setattr__(instance, "metadata", metadata)
        object.__setattr__(instance, "type", type)
        object.__setattr__(instance, "tenant_id", tenant_id)
        return instance

    """Chunk metadata stored alongside the vector."""

    _SKIP_VERIFY_FLAG = "_chunk_skip_verify"
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
        """Validate chunk id/text/checksum and the checksum-text alignment."""
        if not self.id:
            raise VerificationError("Chunk: empty id")
        if not self.text:
            raise VerificationError("Chunk: empty text")
        if not self.checksum:
            raise VerificationError("Chunk: empty checksum")
        if (
            self.checksum
            != hashlib.sha256(self.text.encode("utf-8", errors="surrogatepass")).hexdigest()
        ):
            raise VerificationError("Chunk: checksum mismatch (expected sha256(text))")

    def copy(self, **updates: Any) -> Chunk:
        """Return a copy with the given fields replaced.

        When ``text`` is updated without an explicit ``checksum``, the
        checksum is recomputed from the new text to keep the chunk's
        invariant satisfied.
        """
        if "text" in updates and "checksum" not in updates:
            updates["checksum"] = hashlib.sha256(
                updates["text"].encode("utf-8", errors="surrogatepass")
            ).hexdigest()
        return dataclasses.replace(self, **updates)


@dataclass(slots=True, frozen=True)
class Hit(Snap):
    """A retrieved chunk with score and metadata."""

    chunk: Chunk
    score: float = 0.0
    type: HitType = HitType.Dense

    @property
    def chunk_id(self) -> str:
        """The id of the underlying chunk."""
        return self.chunk.id

    def __post_init__(self) -> None:
        """Validate chunk_id and recursively verify the embedded Chunk."""
        if not self.chunk_id:
            raise VerificationError("Hit: empty chunk_id")
        # Recursively verify the inner chunk so a Hit surfaced from a
        # corrupted store fails fast at the Hit boundary.
        self.chunk.verify()

    def verify(self) -> None:
        """Verify both the Hit and the embedded chunk."""
        self.__post_init__()


@dataclass(slots=True, frozen=True)
class SearchResult(Hit):
    """Alias for :class:`Hit`."""

    pass


@dataclass(slots=True, frozen=True)
class SearchRequest(Snap):
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
class SearchResponse(Snap):
    """Search output from the retrieval pipeline."""

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical spec-named models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DocumentBlock(Snap):
    """A single atom within a section: paragraph, table, image, equation."""

    block_id: str = field(default_factory=lambda: deterministic_id("block", str(uuid4())))
    kind: BlockKind = BlockKind.Text
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class DocumentSection(Snap):
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
class ChunkAlias(Snap):
    """Alias placeholder for :class:`Chunk`."""

    pass


@dataclass(slots=True, frozen=True)
class Embedding(Snap):
    """A typed vector with provenance."""

    id: str = field(default_factory=lambda: str(uuid4()))
    target: str = ""
    model: str = ""
    dim: int = 0
    vector: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    type: EmbeddingType = EmbeddingType.Dense

    def __post_init__(self) -> None:
        """Validate that the embedding carries an id and a non-empty vector."""
        if not self.id:
            raise VerificationError("Embedding: empty id")
        if not self.vector:
            raise VerificationError("Embedding: empty vector")


@dataclass(slots=True, frozen=True)
class Citation(Snap):
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
        """Validate that the citation carries a non-empty document reference."""
        if not self.document_id:
            raise VerificationError("Citation: empty document_id")


@dataclass(slots=True, frozen=True)
class Citations(Snap):
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
class Response(Snap):
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
        """Validate answer/citation/sources and verify the citations aggregate."""
        if not self.answer and not self.citations:
            raise VerificationError("Response: empty answer and no citations")
        Citations(items=list(self.citations)).verify(chunks=list(self.source_chunks))

    def citations_aggregate(self) -> Citations:
        """Return the citations wrapped as a :class:`Citations` aggregate."""
        return Citations(items=list(self.citations))


@dataclass(slots=True, frozen=True)
class Bundle(Snap):
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
class PipelineCtx(Snap):
    """Per-invocation state passed to every stage of a pipeline.

    Attributes:
        pipeline_id: Stable id for this run.
        pipeline_name: Logical pipeline name (e.g. ``"ingest"``).
        user: Authenticated user principal driving the call.
        meta: Mutable :class:`PipelineMeta` updated by stages
            during the run (duration, resolved config, etc.).
        started_at: Pipeline start timestamp (UTC).

    """

    pipeline_id: str = field(default_factory=lambda: deterministic_id("pipeline", str(uuid4())))
    pipeline_name: str = "default"
    user: Any | None = None
    meta: Any = None  # PipelineMeta; Any avoids the import cycle
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a flat dict view of every field on ``meta``.

        Tests and lightweight callers often treat per-run metadata as
        a ``dict``; this property surfaces both typed attributes
        (e.g. ``duration_ms``) and the free-form ``meta.extra`` bag.
        """
        if self.meta is None:
            return {}
        snapshot: dict[str, Any] = {}
        if dataclasses.is_dataclass(self.meta):
            snapshot.update({f.name: getattr(self.meta, f.name) for f in fields(self.meta)})
        return snapshot

    def get(self, key: str, default: Any = None) -> Any:
        """Look up ``key`` in ``metadata`` with a default fallback."""
        return self.metadata.get(key, default)


@dataclass(slots=True, frozen=True)
class PipelineOutputs(Snap):
    """Typed output of a pipeline run.

    Each pipeline type fills the fields that apply; the rest are
    ``None``. The fields are intentionally a wide union so any
    pipeline can return its result through the same :class:`Pipeline`
    without coercing into a per-pipeline carrier class.

    Attributes:
        chunks: Chunks produced by the ingest pipeline.
        vectors: Vectors produced by the ingest pipeline (parallel to ``chunks``).
        chunks_written: Rows actually written by the vector store.
        answer: The query pipeline's answer string.
        hits: The query pipeline's retrieval hits.
        citations: The query pipeline's citation list.
        source_chunks: The query pipeline's :class:`Chunk` list.
        structured: The structured-output payload (or ``None``).
        history: Conversation history the answer was generated against.
        transforms_applied: Names of query transforms that ran.
        planner_trace: Agent loop planner events.
        tools_invoked: Agent loop tool-call names.
        agent_trace: Agent loop aggregate trace dict.
        extra: Pipeline-specific fields that don't warrant a typed
            attribute.

    """

    chunks: list[Chunk] | None = None
    vectors: list[list[float]] | None = None
    chunks_written: int | None = None
    answer: str | None = None
    hits: list[Hit] | None = None
    citations: list[Citation] | None = None
    source_chunks: list[Chunk] | None = None
    structured: dict[str, Any] | None = None
    history: list[Turn] | None = None
    transforms_applied: list[str] | None = None
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] | None = None
    agent_trace: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Pipeline(Snap):
    """Output of a pipeline run.

    The model uses an ``error: ErrorInfo | None`` discriminator rather
    than a boolean ``success`` flag: ``error is None`` means the run
    succeeded, ``error`` is set means it failed.
    """

    pipeline_id: str = ""
    pipeline_name: str = ""
    type: PipelineType = PipelineType.Ingest
    outputs: PipelineOutputs = field(default_factory=PipelineOutputs)
    error: ErrorInfo | None = None
    finished_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def succeeded(self) -> bool:
        """``True`` when the pipeline completed without error."""
        return self.error is None

    def __post_init__(self) -> None:
        """Validate that a Pipeline error carries a message when present."""
        if self.error is not None and not self.error.message:
            raise VerificationError("Pipeline: error.message required when error is set")

    def get(self, key: str, default: Any = None) -> Any:
        """Look up a key in the run output.

        Looks the key up first in ``outputs.extra`` so heterogeneous
        pipelines can stash payload without forcing a new field on the
        shared dataclass. Falls back to the typed
        :class:`PipelineOutputs` attributes (``answer`` / ``chunks`` /
        ``hits`` / ...) so callers can rely on either path.

        Args:
            key: The output key to look up.
            default: Returned when ``key`` is neither in ``extra`` nor
                a typed field on :class:`PipelineOutputs`.

        Returns:
            The resolved value, or ``default``.

        """
        if key in self.outputs.extra:
            return self.outputs.extra[key]
        if key in {f.name for f in fields(PipelineOutputs)}:
            return getattr(self.outputs, key)
        return default


@dataclass(slots=True, frozen=True)
class Result(Snap):
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
class RankedItem(Snap):
    """One ranked chunk in a :class:`RankedList` result."""

    id: str = ""
    score: float = 0.0
    chunk: Chunk | None = None
    rank: int = 0


@dataclass(slots=True, frozen=True)
class LongContextRankedItem(Snap):
    """A single re-ranked candidate produced by the long-context LLM."""

    chunk_id: str = ""
    score: float = 0.0
    rationale: str = ""


@dataclass(slots=True, frozen=True)
class RankedList(Snap):
    """Wrapper that lets structured-output providers validate the LLM output."""

    items: list[RankedItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AuthLoginRequest(Snap):
    """Login request payload."""

    email: str = ""
    password: str = ""


@dataclass(slots=True, frozen=True)
class AuthLoginResponse(Snap):
    """Login response payload."""

    session_token: str = ""
    user_email: str = ""
    allowed_companies: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class DocumentUploadResponse(Snap):
    """Upload response payload."""

    document_id: str = ""
    version: int = 0
    status: str = ""
    company: str = ""
    filename: str = ""


@dataclass(slots=True, frozen=True)
class QueryRequest(Snap):
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
class QueryResponse(Snap):
    """Question answering response."""

    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    source_chunks: list[dict[str, Any]] = field(default_factory=list)
    planner_trace: list[dict[str, Any]] | None = None
    tools_invoked: list[str] = field(default_factory=list)
    transforms_applied: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BatchIngestItem(Snap):
    """Result of ingesting a single file in a batch request."""

    filename: str = ""
    document_id: str = ""
    status: str = "ok"
    error: str = ""


@dataclass(slots=True, frozen=True)
class BatchIngestResponse(Snap):
    """Response from the batch-ingest endpoint."""

    documents: list[BatchIngestItem] = field(default_factory=list)


__all__ = [
    "Access",
    "AuthLoginRequest",
    "AuthLoginResponse",
    "BatchIngestItem",
    "BatchIngestResponse",
    "BlockKind",
    "BlockType",
    "Bundle",
    "BundleType",
    "Chunk",
    "ChunkAlias",
    "ChunkType",
    "Citation",
    "Citations",
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
    "PipelineOutputs",
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
    "deterministic_id",
]
