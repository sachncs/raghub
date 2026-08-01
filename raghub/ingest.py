"""Document ingestion workflows.

This module exposes four ingestion concerns:
in separate files:

* :class:`Ingestor` — synchronous ingestion over the
  canonical ingest pipeline (the public API / CLI callers
  both hit this).
* :class:`Batch` / :class:`Job` —
  thread-pool-backed fire-and-forget ingestion with status tracking.
* :class:`Resumable` — extends the
  background service with a SQLite ledger so jobs survive restarts.
* :class:`JobStore` — the SQLite-backed job ledger used
  by the resumable service.
* :class:`WordChunker` — the built-in overlap-aware chunker.
* :class:`Chonkie` — the Chonkie-backed chunker; supported
  strategies are recursive, token, sentence, semantic, late, table,
  code, slumber, neural.
* :func:`build_chonkie_chunker` — strategy-dispatch helper.

The :data:`__getattr__` cycle-breaker in the prior
``raghub.ingestion`` package has been dropped; everything is now
defined at module load.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from raghub.embedder import Embedder
from raghub.errors import (
    ConfigurationError,
    IngestionError,
)
from raghub.lifecycle import (
    ChunkingPlan,
    Lifecycle,
    PlainTextConverter,
    chunk_words,
    normalize_text,
    validate_upload,
)
from raghub.models import (
    Chunk,
    Chunker,
    Classification,
    Document,
    DocumentLifecycleStatus,
    Pipeline,
    PipelineCtx,
    User,
    deterministic_id,
)
from raghub.pipeline import Ingest
from raghub.repos import UnitOfWork
from raghub.utils import capture

__all__ = [
    "Batch",
    "Chonkie",
    "Ingestor",
    "Resumable",
    "WordChunker",
    "build_chonkie_chunker",
]

# ---------------------------------------------------------------------------
# Chunkers
# ---------------------------------------------------------------------------


chonkie, OptionalImportError = capture(__import__, "chonkie")
CHONKIE_AVAILABLE = OptionalImportError is None
CHONKIE_MODULE = chonkie if CHONKIE_AVAILABLE else None


class RAGHubGenie:
    """Adapter bridging raghub's LLMProvider to chonkie's Genie interface."""

    def __init__(self, llm_provider: Any) -> None:
        """Wrap an LLMProvider for chonkie's Genie interface."""
        self.llm = llm_provider

    def generate(self, prompt: str) -> str:
        """Generate a chunking response for ``prompt``."""
        return str(
            self.llm.generate(
                system_prompt="You are a text chunking assistant. Split the text at natural boundaries.",
                conversation=[],
                context=[],
                question=prompt,
            )
        )

    async def agenerate(self, prompt: str) -> str:
        """Generate a chunking response asynchronously."""
        return str(self.generate(prompt))


def build_refinery(context_size: int = 128, tokenizer: str = "character") -> Any:
    """Build an overlap refinery when supported."""
    if CHONKIE_MODULE is None:
        return None
    cls = getattr(CHONKIE_MODULE, "OverlapRefinery", None)
    if cls is None:
        return None
    refinery, error = capture(
        cls, tokenizer=tokenizer, context_size=context_size, merge=True, inplace=True
    )
    return None if isinstance(error, TypeError) else refinery


def apply_refinery(pieces: list[Any], refinery: Any) -> list[Any]:
    """Apply an available refinery to chunks."""
    if refinery is None or not pieces:
        return pieces
    result, error = capture(refinery, pieces)
    return pieces if error is not None else list(result)


def build_chonkie_inner(
    *,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: str = "character",
    chunker_name: str = "recursive",
    embedding_model: str = "minishlab/potion-base-8M",
    language: str = "auto",
    genie: Any = None,
) -> Any:
    """Build the best available Chonkie chunker for the configuration."""
    if not CHONKIE_AVAILABLE or CHONKIE_MODULE is None:
        raise ConfigurationError(
            "chonkie is not installed; install it via `pip install chonkie` or use WordChunker."
        )

    chunker_builders: dict[str, tuple[str, dict[str, Any]]] = {
        "token": (
            "TokenChunker",
            {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        ),
        "sentence": (
            "SentenceChunker",
            {"tokenizer": tokenizer, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        ),
        "recursive": ("RecursiveChunker", {"tokenizer": tokenizer, "chunk_size": chunk_size}),
        "semantic": (
            "SemanticChunker",
            {"embedding_model": embedding_model, "chunk_size": chunk_size, "threshold": 0.8},
        ),
        "late": ("LateChunker", {"embedding_model": embedding_model, "chunk_size": chunk_size}),
        "table": ("TableChunker", {"tokenizer": "row", "chunk_size": max(1, chunk_size // 100)}),
        "code": ("CodeChunker", {"language": language, "chunk_size": chunk_size}),
        "neural": ("NeuralChunker", {"min_characters_per_chunk": 24}),
        "slumber": (
            "SlumberChunker",
            {"genie": genie, "chunk_size": chunk_size, "candidate_size": 128},
        ),
    }

    auto_probe = ("RecursiveChunker", "TokenChunker", "SentenceChunker")

    if chunker_name == "auto":
        for cls_name in auto_probe:
            cls = getattr(CHONKIE_MODULE, cls_name, None)
            if cls is None:
                continue
            sig, signature_error = capture(inspect.signature, cls)
            if isinstance(signature_error, (TypeError, ValueError)):
                sig = None
            kwargs: dict[str, Any] = {}
            if sig is not None:
                params = sig.parameters
                for key, value in (
                    ("tokenizer", tokenizer),
                    ("tokenizer_or_token_counter", tokenizer),
                    ("chunk_size", chunk_size),
                    ("chunk_overlap", chunk_overlap),
                    ("return_type", "chunks"),
                ):
                    if key in params:
                        kwargs[key] = value
            inner, initialization_error = capture(cls, **kwargs)
            if initialization_error is None:
                return inner
            if not isinstance(initialization_error, TypeError):
                raise initialization_error
        raise ConfigurationError(
            "chonkie is installed but no documented chunker accepted the "
            "configuration; please check the installed chonkie version."
        )

    if chunker_name not in chunker_builders:
        raise ConfigurationError(f"Unknown chonkie chunker strategy: {chunker_name!r}")

    cls_name, kwargs = chunker_builders[chunker_name]
    cls = getattr(CHONKIE_MODULE, cls_name, None)
    if cls is None:
        raise ConfigurationError(
            f"chonkie chunker {cls_name!r} not available; "
            "install the required extra (e.g. `pip install chonkie[semantic]`)"
        )
    inner, initialization_error = capture(cls, **kwargs)
    if initialization_error is None:
        return inner
    if isinstance(initialization_error, ConfigurationError):
        raise initialization_error
    raise ConfigurationError(
        f"chonkie {cls_name} failed to initialize: {initialization_error}"
    ) from initialization_error


class Chonkie(Chunker):
    """Chonkie-backed chunker supporting all strategies."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        tokenizer: str = "character",
        chunker_name: str = "recursive",
        embedding_model: str = "minishlab/potion-base-8M",
        language: str = "auto",
        llm_provider: Any = None,
    ) -> None:
        """Initialise the Chonkie chunker.

        Args:
            chunk_size: Tokens per chunk.
            chunk_overlap: Token overlap.
            tokenizer: Tokenizer name (``"character"``, ``"gpt2"``, …).
            chunker_name: Chunking strategy.
            embedding_model: Model for semantic/late chunkers.
            language: Language for CodeChunker.
            llm_provider: raghub LLM provider for SlumberChunker.

        """
        if not CHONKIE_AVAILABLE:
            raise ConfigurationError(
                "chonkie is not installed; install it via `pip install chonkie` or use WordChunker."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        genie = None
        if chunker_name == "slumber":
            if llm_provider is None:
                raise ConfigurationError(
                    "SlumberChunker requires an LLM provider; pass llm_provider="
                )
            genie = RAGHubGenie(llm_provider)

        self.inner = build_chonkie_inner(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            chunker_name=chunker_name,
            embedding_model=embedding_model,
            language=language,
            genie=genie,
        )
        self.refinery = build_refinery(context_size=chunk_overlap, tokenizer=tokenizer)

    def chonkie_text_chunks(self, text: str) -> list[Any]:
        """Invoke the underlying Chonkie chunker; tolerate API drift."""
        pieces, invocation_error = capture(self.inner, text)
        if isinstance(invocation_error, TypeError):
            chunk = getattr(self.inner, "chunk", None) or getattr(self.inner, "split_text", None)
            if chunk is None:
                raise invocation_error
            pieces = chunk(text)
        elif invocation_error is not None:
            raise invocation_error
        return apply_refinery(pieces, self.refinery)

    def chonkie_batch_chunks(self, texts: list[str]) -> list[list[Any]]:
        """Chunk multiple texts at once via chonkie.Pipeline when available."""
        if not texts:
            return []
        return [self.chonkie_text_chunks(text) for text in texts]

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk a bundle via Chonkie."""
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if block.kind.value != "text":
                    continue
                pieces = self.chonkie_text_chunks(block.content)
                for piece in pieces:
                    text: str = (
                        getattr(piece, "text", None)
                        or (piece.get("text") if isinstance(piece, dict) else str(piece))
                        or ""
                    )
                    chunk_id = (
                        getattr(piece, "id", None)
                        or (piece.get("id") if isinstance(piece, dict) else None)
                        or f"{bundle.bundle_id}:{section.index}:{block.block_id}:{len(chunks)}"
                    )
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(
                                section.page_numbers[0] if section.page_numbers else section.index
                            ),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            text=text,
                            checksum=sha256(text.encode("utf-8")).hexdigest(),
                            metadata={
                                "chunker": "chonkie",
                                "strategy": getattr(self.inner, "__class__", type(None)).__name__,
                                "section_index": section.index,
                                "block_id": block.block_id,
                            },
                        )
                    )
        return chunks

    def chunk_text(
        self,
        text: str,
        *,
        document_id: str,
        version: int = 1,
        company: str = "",
        owner: str = "",
    ) -> list[Chunk]:
        """Chunk raw ``text`` via Chonkie."""
        pieces = self.chonkie_text_chunks(text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            text_value: str = (
                getattr(piece, "text", None)
                or (piece.get("text") if isinstance(piece, dict) else str(piece))
                or ""
            )
            chunk_id = (
                getattr(piece, "id", None)
                or (piece.get("id") if isinstance(piece, dict) else None)
                or f"{document_id}:v{version}:{i}"
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    text=text_value,
                    checksum=sha256(text_value.encode("utf-8")).hexdigest(),
                    metadata={
                        "chunker": "chonkie",
                        "strategy": getattr(self.inner, "__class__", type(None)).__name__,
                    },
                )
            )
        return chunks


class WordChunker(Chunker):
    """Overlap-aware word-window chunker."""

    chunk_size: int
    chunk_overlap: int

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        """Initialise the chunker.

        Args:
            chunk_size: Number of words per chunk.
            chunk_overlap: Overlap between consecutive chunks.

        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.plan = ChunkingPlan(chunk_size_words=chunk_size, overlap_words=chunk_overlap)

    def chunk(self, bundle: Any) -> list[Chunk]:
        """Chunk ``bundle`` into overlapping windows."""
        chunks: list[Chunk] = []
        for section in bundle.sections:
            for block in section.blocks:
                if block.kind.value != "text":
                    continue
                for text in self.word_window_chunks(block.content):
                    chunk_id = deterministic_id(
                        "chunk",
                        bundle.source_uri,
                        str(section.index),
                        block.block_id,
                        text[:64],
                    )
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            document_id=bundle.bundle_id,
                            version=1,
                            page=(
                                section.page_numbers[0] if section.page_numbers else section.index
                            ),
                            source_location=section.source_location or bundle.source_uri,
                            section=section.heading,
                            company="",
                            owner=bundle.metadata.get("owner", ""),
                            department=bundle.metadata.get("department", ""),
                            text=text,
                            checksum=sha256(text.encode("utf-8")).hexdigest(),
                            metadata={
                                "block_kind": "text",
                                "block_id": block.block_id,
                                "section_index": section.index,
                            },
                        )
                    )
        return chunks

    def chunk_text(
        self,
        text: str,
        *,
        document_id: str,
        version: int = 1,
        company: str = "",
        owner: str = "",
    ) -> list[Chunk]:
        """Chunk raw ``text`` (no bundle)."""
        result: list[Chunk] = []
        for chunk_text in self.word_window_chunks(text):
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(version),
                chunk_text[:64],
            )
            result.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=version,
                    company=company,
                    owner=owner,
                    text=chunk_text,
                    checksum=sha256(chunk_text.encode("utf-8")).hexdigest(),
                )
            )
        return result

    def word_window_chunks(self, text: str) -> list[str]:
        """Split ``text`` into overlapping windows."""
        return chunk_words(normalize_text(text), self.plan)


def build_chonkie_chunker(name: str = "auto", **kwargs: Any) -> Chunker:
    """Pick a chunker by name.

    Args:
        name: Chunker strategy.
        **kwargs: Forwarded to the underlying constructor.

    Returns:
        A configured :class:`Chunker`.

    Raises:
        ConfigurationError: When ``name`` is unknown or chonkie is
            explicitly requested but unavailable.

    """
    chonkie_names = {
        "auto",
        "recursive",
        "token",
        "sentence",
        "semantic",
        "late",
        "table",
        "code",
        "slumber",
        "neural",
    }
    if name in chonkie_names:
        if CHONKIE_AVAILABLE:
            return Chonkie(chunker_name=name, **kwargs)
        if name != "auto":
            raise ConfigurationError("chonkie is not installed")
    if name in ("chonkie", "word_window", "auto"):
        if name == "chonkie":
            if CHONKIE_AVAILABLE:
                return Chonkie(**kwargs)
            raise ConfigurationError("chonkie is not installed")
        return WordChunker(**kwargs)
    raise ConfigurationError(f"Unknown chunker: {name!r}")


# ---------------------------------------------------------------------------
# Ingestion service (synchronous wrapper over the ingest pipeline)
# ---------------------------------------------------------------------------


VirusScanHook = Callable[[bytes], None]


@dataclass
class IngestionResult:
    """The outcome of a successful ingestion.

    Attributes:
        document: The persisted :class:`Document` in its final
            status (``READY`` or a prior duplicate).
        chunks: The chunks that were indexed for this document.

    """

    document: Document
    chunks: list[str] = field(default_factory=list)


def record_from_pipeline(
    result: Pipeline,
    *,
    file_name: str,
    mime_type: str,
    owner: User,
    organization: str,
    classification: Classification,
    checksum: str,
    tags: list[str] | None,
) -> Document:
    """Project a :class:`Pipeline` into a :class:`Document`."""
    chunks = result.outputs.get("chunks") or []
    if chunks and isinstance(chunks[0], dict):
        chunk_records = [Chunk.model_validate(c) for c in chunks]
    else:
        chunk_records = list(chunks)
    bundle = result.outputs.get("bundle")
    document_id = str(result.outputs.get("document_id") or getattr(bundle, "bundle_id", "") or "")
    for chunk in chunk_records:
        if not chunk.document_id:
            chunk.document_id = document_id
    chunks = [c.id for c in chunk_records]
    return Document(
        id=document_id,
        version=int(result.outputs.get("version") or 1),
        checksum=checksum,
        owner=owner.email,
        organization=organization,
        tags=tags or [],
        classification=classification,
        status=DocumentLifecycleStatus.READY,
        filename=file_name,
        file_type=file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "",
        mime_type=mime_type,
        chunk_count=len(chunk_records),
        chunks=chunks,
    )


class Ingestor:
    """Thin wrapper over :class:`raghub.pipelines.rag.Ingest`.

    The service is constructed once and reused for many uploads. It is
    stateless apart from the wired collaborators, which makes it safe to
    share across concurrent coroutines as long as the underlying
    ``UnitOfWork`` is itself concurrent-safe.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        embedding_provider: Embedder,
        lifecycle_manager: Lifecycle,
        max_upload_bytes: int,
        virus_scan_hook: VirusScanHook | None = None,
        pipeline: Ingest | None = None,
        plan: object | None = None,
    ) -> None:
        """Initialise the service."""
        self.uow = uow
        self.embedding_provider = embedding_provider
        self.lifecycle_manager = lifecycle_manager
        self.max_upload_bytes = max_upload_bytes
        self.virus_scan_hook = virus_scan_hook or (lambda _: None)
        self.plan = plan
        self.make_pipeline: Ingest | None = pipeline

    def build_pipeline(self) -> Ingest:
        """Construct the default :class:`Ingest`."""
        return Ingest(
            converter=PlainTextConverter(),
            chunker=WordChunker(),
            embedder=self.embedding_provider,
            vector_store=self.uow.vector_store,
        )

    def submit_async(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: User,
        organization: str,
        department: str = "",
        tags: list[str] | None = None,
        classification: Classification = Classification.INTERNAL,
        background_service: Batch | None = None,
    ) -> str:
        """Submit ``ingest`` to a background thread pool."""
        svc = background_service or Batch()
        return svc.submit(
            self.ingest,
            file_name=file_name,
            file_bytes=file_bytes,
            owner=owner,
            organization=organization,
            department=department,
            tags=tags,
            classification=classification,
        )

    async def ingest(
        self,
        *,
        file_name: str,
        file_bytes: bytes,
        owner: User,
        organization: str,
        department: str = "",
        tags: list[str] | None = None,
        classification: Classification = Classification.INTERNAL,
    ) -> IngestionResult:
        """Run the canonical ingest pipeline for a single upload.

        Args:
            file_name: Original filename.
            file_bytes: Raw file content.
            owner: The uploading user principal.
            organization: Tenant (company) identifier.
            department: Optional department tag.
            tags: Optional tag list.
            classification: Sensitivity classification.

        Returns:
            An :class:`IngestionResult` carrying the final document
            record and chunk ids.

        Raises:
            IngestionError: If any ingestion stage fails. The document
                is left in ``FAILED`` state with the error message
                persisted.

        """
        mime_type = validate_upload(file_name, file_bytes, self.max_upload_bytes)
        self.virus_scan_hook(file_bytes)
        checksum = sha256(file_bytes).hexdigest()

        previous = await self.uow.document_repo.get_by_checksum(checksum)
        if previous is not None and previous.status == DocumentLifecycleStatus.READY:
            return IngestionResult(document=previous, chunks=list(previous.chunks))

        context = PipelineCtx(pipeline_name="ingest", metadata={"user_id": owner.email})
        if self.make_pipeline is None:
            self.make_pipeline = self.build_pipeline()
        result = await self.make_pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=file_name,
            mime_type=mime_type,
            metadata={
                "department": department,
                "tags": tags or [],
                "classification": classification.value,
            },
            user=owner,
            company=organization,
        )
        if result.error is not None:
            error_message = (result.error.message if result.error else None) or "ingestion failed"
            if previous is not None:
                previous.status = DocumentLifecycleStatus.FAILED
                previous.error = error_message if isinstance(error_message, str) else error_message.message
                await self.uow.document_repo.save(previous)
            raise IngestionError(error_message)

        record = record_from_pipeline(
            result,
            file_name=file_name,
            mime_type=mime_type,
            owner=owner,
            organization=organization,
            classification=classification,
            checksum=checksum,
            tags=tags,
        )
        await self.uow.document_repo.save(record)
        return IngestionResult(document=record, chunks=list(record.chunks))


# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------


class Job:
    """Lightweight value object tracking a single ingestion task.

    Attributes:
        job_id: Stable identifier returned by :meth:`submit`.
        status: One of ``"pending"``, ``"processing"``, ``"completed"``,
            ``"failed"``.
        result: The callable's return value on success, the stringified
            exception on failure, or ``None`` while pending.

    """

    def __init__(self, job_id: str, status: str, result: Any = None) -> None:
        """Initialise the job record."""
        self.job_id = job_id
        self.status = status
        self.result = result


class Batch:
    """Queues ingestion jobs for async processing.

    A thin wrapper around :class:`ThreadPoolExecutor` that adds job
    tracking. Construct once and reuse; constructing per-call does **not**
    reuse the underlying executor.

    Attributes:
        executor: Backing thread pool.
        jobs: Map from job id to :class:`Job`.
        closed: ``True`` after :meth:`shutdown` has been invoked.

    """

    def __init__(self, max_workers: int = 2) -> None:
        """Initialise the service with a thread pool."""
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: dict[str, Job] = {}
        self.closed = False

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit a callable for background execution."""
        if self.closed:
            raise RuntimeError("Batch is shut down")
        job_id = str(uuid4())
        self.jobs[job_id] = Job(job_id, "pending")
        self.executor.submit(self.run_job, job_id, fn, args, kwargs)
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute one queued job, including asyncio unwrapping."""
        job = self.jobs[job_id]
        job.status = "processing"
        result, error = capture(fn, *args, **kwargs)
        if error is None and asyncio.iscoroutine(result):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result, error = capture(loop.run_until_complete, result)
            loop.close()
        if error is not None:
            job.status = "failed"
            job.result = str(error)
            return
        job.status = "completed"
        job.result = result

    def get_status(self, job_id: str) -> str | None:
        """Return the current status for ``job_id``, or ``None`` if unknown."""
        job = self.jobs.get(job_id)
        return job.status if job else None

    def get_result(self, job_id: str) -> Any:
        """Return the stored result for ``job_id``, or ``None`` if unknown."""
        job = self.jobs.get(job_id)
        return job.result if job else None

    def shutdown(self, *, wait: bool = True) -> None:
        """Release the thread pool and refuse further submissions."""
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=wait)


# ---------------------------------------------------------------------------
# Persistent job store
# ---------------------------------------------------------------------------


class JobStore:
    """SQLite-backed job ledger.

    Records the lifecycle of every ingestion job so the application
    can resume after a crash. Records older than 24 hours are
    pruned lazily on save.
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialise the store."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert(self, job_id: str, status: str, result: Any = None) -> None:
        """Insert or update a job record."""
        encoded = (
            json.dumps(result) if result is not None and not isinstance(result, str) else result
        )
        self.conn.execute(
            """
            INSERT INTO ingestion_jobs (job_id, status, result, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET status = excluded.status, result = excluded.result
            """,
            (job_id, status, encoded, time.time()),
        )
        self.conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return the job record or ``None`` if unknown."""
        row = self.conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return {"job_id": row[0], "status": row[1], "result": row[2]}

    def all_jobs(self) -> Iterable[dict[str, Any]]:
        """Yield every persisted job."""
        for row in self.conn.execute(
            "SELECT job_id, status, result FROM ingestion_jobs"
        ).fetchall():
            yield {"job_id": row[0], "status": row[1], "result": row[2]}

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with suppress(Exception):
            self.conn.close()


# ---------------------------------------------------------------------------
# Resumable background ingestion
# ---------------------------------------------------------------------------


class Resumable(Batch):
    """Background ingestion with a persistent job ledger."""

    def __init__(self, *, db_path: str | Path, max_workers: int = 2) -> None:
        """Initialise the service."""
        super().__init__(max_workers=max_workers)
        self.store = JobStore(db_path)
        self.restore_from_store()

    def restore_from_store(self) -> None:
        """Reload prior job state into the in-memory map."""
        for record in self.store.all_jobs():
            self.jobs[record["job_id"]] = Job(
                job_id=record["job_id"],
                status=record["status"],
                result=record["result"],
            )

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> str:
        """Submit ``fn`` for background execution."""
        job_id = super().submit(fn, *args, **kwargs)
        self.store.upsert(job_id, "pending")
        return job_id

    def run_job(self, job_id: str, fn: Any, args: Any, kwargs: Any) -> None:
        """Execute a job, persisting status transitions."""
        super().run_job(job_id, fn, args, kwargs)
        job = self.jobs.get(job_id)
        if job is not None:
            self.store.upsert(job_id, job.status, job.result)

    def shutdown(self, *, wait: bool = False) -> None:
        """Flush the job store and shut down the executor."""
        if self.closed:
            return
        for job_id, job in list(self.jobs.items()):
            self.store.upsert(job_id, job.status, job.result)
        self.store.close()
        super().shutdown(wait=wait)
