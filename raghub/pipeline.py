"""Orchestration pipelines — ingest, query, agentic, cache, timing.

This is the **wiring** file. It composes every other module
(retrieval, embeddings, vector store, generation, knowledge,
observability, agent, conversation) into the public pipeline
surface. The file is large because the wiring itself is the
framework's value-add; co-locating every pipeline keeps the
single-responsibility split legible.

Section map:

* :class:`DurationTimer` — context manager that records pipeline
  wall-clock duration onto the ``PipelineCtx.metadata``.
* :class:`QueryCache` — TTL-based in-memory query cache.
* :class:`ConversationRouter` — facade over a pluggable conversation
  store.
* :class:`PipelineBuilder` — fluent builder for
  :class:`Pipeline` records.
* :class:`Ingest` — convert → chunk → embed → index.
* :class:`QueryPipeline` — embed → retrieve → rerank → generate.
* :class:`AgentPipeline` — agent-driven query pipeline.
* :func:`get_chunks` / :func:`primary_company` /
  :func:`sha256_checksum` — small ingest helpers.
* :func:`citations_from_trace` / :func:`hits_from_trace` —
  agent-trace → citation/hit coercion.
* :func:`canonical_filters` — flatten filters into hashable tuples.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
from types import TracebackType
from typing import Any, cast

from pydantic import ConfigDict
from tqdm import tqdm

from raghub.agent import Agent, AgentRequest, AgentTrace
from raghub.conv import Memory
from raghub.embedder import Embedder
from raghub.errors import PipelineError, VectorStoreError
from raghub.knowledge import MemoryRepo
from raghub.lifecycle import PlainTextConverter
from raghub.models import (
    Bundle,
    Chunk,
    Citation,
    Classification,
    EmbeddingProvider,
    ErrorInfo,
    GeneratorProtocol,
    Hit,
    Pipeline,
    PipelineCtx,
    PipelineRunner,
    Turn,
    User,
    VectorStore,
    deterministic_id,
)
from raghub.retry import retry as retry_sync
from raghub.telemetry import NoOpTelemetry


def awaitable(value: Any) -> Any:
    """Make ``await`` work for either sync return values or coroutines.

    Lifts a sync result into an inline coroutine so the query
    pipeline can drive both the async and the sync ``generate``
    path through the same call site.
    """
    if inspect.isawaitable(value):
        return value

    # Inline async coroutine factory (one level deep, no nested def).
    async def lift() -> Any:
        """Lift a sync return value into a coroutine."""
        await asyncio.sleep(0)
        return value

    return lift()


__all__ = [
    "AgentPipeline",
    "Cache",
    "Ingest",
    "QueryPipeline",
]

# ---------------------------------------------------------------------------
# DurationTimer
# ---------------------------------------------------------------------------


class DurationTimer(AbstractContextManager["DurationTimer"]):
    """Set ``context.metadata["duration_ms"]`` on exit."""

    def __init__(self, context: Any) -> None:
        """Store the context; the start time is captured on entry."""
        self.context = context
        self.start: float = 0.0

    def __enter__(self) -> DurationTimer:
        """Capture the start time and return ``self`` for ``as`` binding."""
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Record the elapsed milliseconds in ``context.metadata``."""
        self.context.metadata["duration_ms"] = (time.perf_counter() - self.start) * 1000.0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def canonical_filters(filters: dict[str, Any] | str | None) -> tuple[tuple[str, Any], ...]:
    """Flatten ``filters`` into a hashable tuple."""
    if filters is None:
        return ()
    if isinstance(filters, str):
        return (("raw", filters),)
    items = []
    for key, value in sorted(filters.items()):
        if isinstance(value, list):
            value = tuple(value)
        items.append((key, value))
    return tuple(items)


class Cache:
    """Simple TTL-based in-memory query cache."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        """Initialise the cache with a TTL in seconds."""
        self.ttl = ttl_seconds
        self.store: dict[tuple[Any, ...], tuple[float, Pipeline]] = {}

    @staticmethod
    def make_key(
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        **options: "JSONValue",
    ) -> tuple[Any, ...]:
        """Build the cache key for the given query context.

        Args:
            question: The query question.
            user_id: The caller user id.
            filters: Optional metadata filter.
            **options: Optional overrides (``top_k=``,
                ``response_model=``, ``session_id=``, ``history=``,
                ``scope=``).

        """
        top_k: int = options.get("top_k", 5)
        response_model: Any | None = options.get("response_model")
        session_id: str | None = options.get("session_id")
        history: Sequence[Any] = options.get("history", ())
        scope: Any = options.get("scope")
        model_key = ""
        if response_model is not None:
            model_key = (
                f"{response_model.__module__}.{response_model.__qualname__}"
                if isinstance(response_model, type)
                else str(response_model)
            )
        history_key = tuple(
            (
                turn.get("question", "")
                if isinstance(turn, dict)
                else getattr(turn, "question", ""),
                turn.get("answer", "") if isinstance(turn, dict) else getattr(turn, "answer", ""),
            )
            for turn in history
        )
        if isinstance(scope, dict):
            scope_key = canonical_filters(scope)
        elif isinstance(scope, list):
            scope_key = tuple(scope)
        else:
            scope_key = scope
        return (
            question,
            user_id or "",
            canonical_filters(filters),
            int(top_k),
            model_key,
            session_id or "",
            history_key,
            scope_key,
        )

    def get(
        self,
        question: str,
        user_id: str | None = None,
        filters: dict[str, Any] | str | None = None,
        **options: "JSONValue",
    ) -> Pipeline | None:
        """Return a cached :class:`Pipeline` or ``None``."""
        key = self.make_key(question, user_id, filters, **options)
        entry = self.store.get(key)
        if entry is None:
            return None
        timestamp, result = entry
        if time.monotonic() - timestamp > self.ttl:
            del self.store[key]
            return None
        return result

    def set(
        self,
        question: str,
        user_id: str | None,
        filters: dict[str, Any] | str | None,
        result: Pipeline,
        **options: "JSONValue",
    ) -> None:
        """Store a :class:`Pipeline` in the cache."""
        key = self.make_key(question, user_id, filters, **options)
        self.store[key] = (time.monotonic(), result)

    def clear(self) -> None:
        """Evict every cached entry."""
        self.store.clear()

    def invalidate(
        self,
        question: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Evict entries matching the given criteria."""
        if question is None and user_id is None:
            self.clear()
            return
        to_delete = [
            k
            for k in self.store
            if (question is None or k[0] == question) and (user_id is None or k[1] == user_id)
        ]
        for key in to_delete:
            del self.store[key]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Thin facade over a pluggable conversation store."""

    def __init__(self, store: Any) -> None:
        """Store the backing conversation store reference."""
        self.store = store

    def load_history(self, session_id: str | None, limit: int = 20) -> list[Turn]:
        """Return the recent turns for ``session_id``."""
        if not session_id:
            return []
        return cast(list[Turn], self.store.load(session_id, limit=limit))

    def record_turn(
        self,
        session_id: str | None,
        turn: Any,
        *,
        skip_when_empty: bool = True,
    ) -> bool:
        """Append ``turn`` to ``session_id`` when applicable."""
        if not session_id:
            return False
        if skip_when_empty and not getattr(turn, "answer", ""):
            return False
        self.store.append(session_id, turn)
        return True


# ---------------------------------------------------------------------------
# PipelineBuilder
# ---------------------------------------------------------------------------


class PipelineBuilder:
    """Fluent builder for :class:`Pipeline` records."""

    def __init__(self, context: PipelineCtx, pipeline_name: str) -> None:
        """Store the context and pipeline name for subsequent builds."""
        self.context = context
        self.pipeline_name = pipeline_name

    def success(self, outputs: dict[str, Any]) -> Pipeline:
        """Build a successful :class:`Pipeline` with ``outputs``."""
        return Pipeline(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            outputs=outputs,
        )

    def failure(self, error: str, outputs: dict[str, Any] | None = None) -> Pipeline:
        """Build a failed :class:`Pipeline` with ``error``."""
        return Pipeline(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            error=ErrorInfo(kind="ingestion", message=error),
            outputs=outputs or {},
        )


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------


def get_chunks(bundle: Bundle, document_id: str, company: str = "") -> list[Chunk]:
    """Materialise the :class:`Chunk` list for a bundle's sections."""
    chunks: list[Chunk] = []
    tenant_company = company or bundle.metadata.get("company", "")
    for section in bundle.sections:
        for block in section.blocks:
            if block.kind.value != "text":
                continue
            text = (block.content or "").strip()
            if not text:
                continue
            chunk_id = deterministic_id(
                "chunk",
                document_id,
                str(section.index),
                block.block_id,
                text[:64],
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    version=1,
                    page=(section.page_numbers[0] if section.page_numbers else section.index),
                    source_location=section.source_location or bundle.source_uri,
                    section=section.heading,
                    company=tenant_company,
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


def sha256_checksum(file_bytes: bytes) -> str:
    """SHA-256 of the raw file content."""
    return sha256(file_bytes).hexdigest()


def primary_company(user: Any) -> str:
    """Return the primary company for a :class:`User`."""
    if user is None:
        return ""
    companies = getattr(user, "allowed_companies", None) or []
    if getattr(user, "is_admin", False):
        return ""
    if not companies:
        return ""
    return str(companies[0])


@dataclass(frozen=True, slots=True)
class IngestResolvedMetadata:
    """Resolved per-request metadata for :meth:`Ingest.run`."""

    normalized_metadata: dict[str, Any]
    document_id: str
    version: int
    tenant_company: str
    owner: str
    classification: Classification
    mime_type: str
    language: str


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Per-request context passed through the :class:`QueryPipeline` helpers."""

    question: str
    top_k: int
    user_filter: dict[str, Any] | str
    user: Any | None
    session_id: str | None
    response_model: Any
    record: bool
    history: list[Turn]
    rbac_filter: dict[str, Any] | str
    user_id: str | None
    scope: tuple[bool, tuple[str, ...], tuple[str, ...]]


class Ingest(PipelineRunner):
    """Convert → chunk → embed → index pipeline."""

    name: str = "ingest"

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        **components: Any,
    ) -> None:
        """Initialise the ingest pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            **components: Optional collaborators — ``converter=``,
                ``chunker=``, ``knowledge_repo=``, ``telemetry=``,
                ``raptor=``, ``graph=``.

        """
        from raghub.ingest import WordChunker

        if embedder is None or vector_store is None:
            raise PipelineError("Ingest requires embedder and vector_store")
        self.converter = components.get("converter") or PlainTextConverter()
        self.chunker = components.get("chunker") or WordChunker()
        self.embedder = embedder
        self.vector_store = vector_store
        self.knowledge_repo = components.get("knowledge_repo") or MemoryRepo()
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        self.raptor = components.get("raptor")
        self.graph = components.get("graph")
        self.show_progress = True

    def vectors_already_indexed(self, chunks: list[Chunk]) -> bool:
        """Return ``True`` when every chunk already lives in the vector store."""
        if not chunks:
            return True
        has_chunk = getattr(self.vector_store, "has_chunk", None)
        if not callable(has_chunk):
            return True
        return all(bool(has_chunk(chunk.id)) for chunk in chunks)

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the ingest pipeline."""
        with DurationTimer(context):
            file_bytes: bytes = inputs["file_bytes"]
            source_uri: str = inputs["source_uri"]
            metadata_in = dict(inputs.get("metadata") or {})
            user: Any | None = inputs.get("user")
            force: bool = bool(inputs.get("force"))
            checksum = sha256_checksum(file_bytes)
            bundle_id = deterministic_id("bundle", source_uri, checksum)
            resolved = self.resolve_ingest_metadata(
                inputs=inputs,
                metadata=metadata_in,
                user=user,
                bundle_id=bundle_id,
            )

            with self.telemetry.span("ingest", source_uri=source_uri, bundle_id=bundle_id) as sp:
                sp.set_attribute("checksum", checksum)

                cached = self.maybe_cached_bundle(
                    context, force, bundle_id, checksum, resolved
                )
                if cached is not None:
                    return cached

                bundle = self.convert_bundle(
                    source_uri,
                    file_bytes,
                    resolved.mime_type,
                    resolved.language,
                    resolved.normalized_metadata,
                )
                bundle.bundle_id = bundle_id
                bundle.checksum = checksum
                bundle.metadata = {**bundle.metadata, **resolved.normalized_metadata}

                chunks = self.chunk_documents(bundle, resolved)
                vectors = self.embed_chunks(chunks)
                self.index_chunks(chunks, vectors)
                self.knowledge_repo.save(bundle)

                return self.build_ingest_result(
                    context, bundle, chunks, vectors, incremental=False
                )

    @staticmethod
    def resolve_ingest_metadata(
        *,
        inputs: dict[str, Any],
        metadata: dict[str, Any],
        user: Any | None,
        bundle_id: str,
    ) -> IngestResolvedMetadata:
        """Resolve the per-request ingest metadata (tenant, owner, classification)."""
        tenant_company = str(
            inputs.get("company") or primary_company(user) or metadata.get("company", "")
        )
        document_id = str(inputs.get("document_id") or bundle_id)
        version = int(inputs.get("version") or metadata.get("version") or 1)
        owner = str(
            inputs.get("owner") or getattr(user, "email", None) or metadata.get("owner", "")
        )
        classification = Classification(
            inputs.get("classification")
            or metadata.get("classification")
            or Classification.INTERNAL
        )
        return IngestResolvedMetadata(
            normalized_metadata={
                **metadata,
                "company": tenant_company,
                "owner": owner,
                "classification": classification.value,
                "document_id": document_id,
                "version": version,
            },
            document_id=document_id,
            version=version,
            tenant_company=tenant_company,
            owner=owner,
            classification=classification,
            mime_type=inputs.get("mime_type", ""),
            language=inputs.get("language", ""),
        )

    def maybe_cached_bundle(
        self,
        context: PipelineCtx,
        force: bool,
        bundle_id: str,
        checksum: str,
        resolved: IngestResolvedMetadata,
    ) -> Pipeline | None:
        """Return a cached ``Pipeline`` when the bundle is already indexed."""
        if force:
            return None
        existing = self.knowledge_repo.get(bundle_id)
        if existing is None or existing.checksum != checksum:
            return None
        prior_chunks = get_chunks(
            existing, resolved.document_id, company=resolved.tenant_company
        )
        if not self.vectors_already_indexed(prior_chunks):
            return None
        return Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "bundle": existing,
                "chunks": prior_chunks,
                "chunk_count": len(prior_chunks),
                "embeddings": [],
                "incremental": True,
            },
        )

    def convert_bundle(
        self,
        source_uri: str,
        file_bytes: bytes,
        mime_type: str,
        language: str,
        normalized_metadata: dict[str, Any],
    ) -> Bundle:
        """Run the converter and return the resulting :class:`Bundle`."""
        with self.telemetry.span("ingest.convert"):
            return self.converter.convert(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type,
                language=language,
                metadata=normalized_metadata,
            )

    def chunk_documents(
        self,
        bundle: Bundle,
        resolved: IngestResolvedMetadata,
    ) -> list[Chunk]:
        """Run the chunker and stamp per-chunk identity fields."""
        with self.telemetry.span("ingest.chunk"):
            raw_chunks = self.chunker.chunk(bundle)
            chunks: list[Chunk] = []
            for chunk in tqdm(
                raw_chunks,
                desc="Chunking",
                disable=not getattr(self, "show_progress", True),
                unit="chunk",
            ):
                chunk.document_id = resolved.document_id
                chunk.version = resolved.version
                chunk.company = resolved.tenant_company
                chunk.owner = resolved.owner
                chunk.classification = resolved.classification
                chunks.append(chunk)
        return chunks

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Embed every chunk's text and return the vectors in chunk order."""
        texts = [chunk.text for chunk in chunks]
        with self.telemetry.span("ingest.embed", count=len(texts)):
            return self.embedder.embed_texts(texts) if texts else []

    def index_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        """Write chunks to the vector store and (optionally) the derived indexes."""
        with self.telemetry.span("ingest.upsert", count=len(chunks)):
            if not chunks:
                return
            written = retry_sync(
                lambda: self.vector_store.upsert(chunks, vectors),
                max_retries=2,
                base_delay=0.5,
            )
            if written != len(chunks):
                raise VectorStoreError(
                    f"vector store wrote {written} of {len(chunks)} chunks"
                )
            if self.raptor is not None:
                with self.telemetry.span("ingest.raptor"):
                    self.raptor.add_chunks(chunks, vectors)
            if self.graph is not None:
                with self.telemetry.span("ingest.graph"):
                    self.graph.add_chunks(chunks, vectors)

    def build_ingest_result(
        self,
        context: PipelineCtx,
        bundle: Bundle,
        chunks: list[Chunk],
        vectors: list[list[float]],
        *,
        incremental: bool,
    ) -> Pipeline:
        """Wrap the ingest result in the framework's :class:`Pipeline` shape."""
        return Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "bundle": bundle,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "embeddings": vectors,
                "incremental": incremental,
            },
        )


# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------


class QueryPipeline(PipelineRunner):
    """Embed → retrieve → rerank → generate pipeline."""

    name: str = "query"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: GeneratorProtocol,
        **components: Any,
    ) -> None:
        """Initialise the query pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Answer generator.
            **components: Optional collaborators — ``reranker=``,
                ``structured=``, ``telemetry=``,
                ``conversation_store=``, ``cache=``,
                ``transformer=``, ``retrieval_pipeline=``,
                ``long_context_pass=``, ``agentic_pipeline=``.

        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = components.get("reranker")
        self.structured = components.get("structured")
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        conversation_store = components.get("conversation_store")
        if conversation_store is None:
            conversation_store = Memory()
        self.conversation_store = conversation_store
        self.cache = components.get("cache")
        self.transformer = components.get("transformer")
        self.retrieval_pipeline = components.get("retrieval_pipeline")
        self.long_context_pass = components.get("long_context_pass")
        self.agentic_pipeline = components.get("agentic_pipeline")

    @staticmethod
    def metadata_filter_for_user(user: Any) -> dict[str, Any] | str:
        """Derive a metadata filter for the vector store from a user."""
        if user is None:
            return ""
        if getattr(user, "is_admin", False):
            return ""
        companies = list(getattr(user, "allowed_companies", []) or [])
        return {"company": companies}

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the query pipeline."""
        with DurationTimer(context):
            return await self.run_inner(context, inputs)

    async def run_inner(
        self,
        context: PipelineCtx,
        inputs: dict[str, Any],
    ) -> Pipeline:
        """Body of :meth:`run` separated so the timing ``finally`` is obvious."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        response_model = inputs.get("response_model")
        record: bool = bool(inputs.get("record", True))
        tools_enabled: set[str] | None = inputs.get("tools_enabled")

        history: list[Turn] = []
        if session_id:
            history = self.conversation_store.load(session_id, limit=20)

        rbac_filter = self.metadata_filter_for_user(user)
        user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
        scope = QueryPipeline.scope_triple(user)

        query_ctx = QueryContext(
            question=question,
            top_k=top_k,
            user_filter=user_filter,
            user=user,
            session_id=session_id,
            response_model=response_model,
            record=record,
            history=history,
            rbac_filter=rbac_filter,
            user_id=user_id,
            scope=scope,
        )

        cached = self.maybe_cache_hit(query_ctx)
        if isinstance(cached, Pipeline):
            return cached

        agent_result = await self.maybe_dispatch_agentic(context, inputs, query_ctx, tools_enabled)
        if agent_result is not None:
            return agent_result

        return await self.run_query_leg(context, query_ctx)

    @staticmethod
    def scope_triple(user: Any) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
        """Build the cache scope tuple for ``user``."""
        return (
            bool(getattr(user, "is_admin", False)),
            tuple(sorted(str(value) for value in getattr(user, "allowed_companies", []) or [])),
            tuple(sorted(str(value) for value in getattr(user, "allowed_groups", []) or [])),
        )

    def maybe_cache_hit(self, ctx: QueryContext) -> Pipeline | None:
        """Return a cached ``Pipeline`` for the request, or ``None``."""
        if self.cache is None:
            return None
        cached = self.cache.get(
            ctx.question,
            ctx.user_id,
            ctx.user_filter,
            top_k=ctx.top_k,
            response_model=ctx.response_model,
            session_id=ctx.session_id,
            history=ctx.history,
            scope=ctx.scope,
        )
        return cached if isinstance(cached, Pipeline) else None

    async def maybe_dispatch_agentic(
        self,
        context: PipelineCtx,
        inputs: dict[str, Any],
        ctx: QueryContext,
        tools_enabled: set[str] | None,
    ) -> Pipeline | None:
        """Forward to the agentic pipeline when tools are enabled."""
        if self.agentic_pipeline is None or not (
            tools_enabled
            or self.resolved_triggers_agent(inputs)
        ):
            return None
        return cast(
            Pipeline,
            await self.agentic_pipeline.run(
                context,
                question=ctx.question,
                user=ctx.user,
                session_id=ctx.session_id,
                tools_enabled=tools_enabled,
                top_k=ctx.top_k,
                history=ctx.history,
            ),
        )

    @staticmethod
    def resolved_triggers_agent(inputs: dict[str, Any]) -> bool:
        """Return whether ``resolved_config`` activates the agent loop.

        Args:
            inputs: The resolved ``inputs`` mapping for the request.
                Looks for ``resolved_config`` and inspects it for
                ``agent_enabled`` / ``tools_enabled`` keys.

        Returns:
            ``True`` when either flag is set in ``resolved_config``;
            ``False`` otherwise (including when ``resolved_config``
            is absent or not a dict).

        """
        record_overrides = inputs.get("resolved_config")
        if not isinstance(record_overrides, dict):
            return False
        return bool(
            record_overrides.get("agent_enabled") or record_overrides.get("tools_enabled")
        )

    async def run_query_leg(
        self,
        context: PipelineCtx,
        ctx: QueryContext,
    ) -> Pipeline:
        """Embed → retrieve → rerank → generate → (optional) structured."""
        with self.telemetry.span("query", question=ctx.question[:128], top_k=ctx.top_k) as span:
            QueryPipeline.annotate_query_span(span, ctx.user, ctx.session_id)

            hits, transforms_applied = await self.retrieve_hits(
                ctx.question, ctx.history, ctx.top_k, ctx.rbac_filter, ctx.user_filter
            )

            answer, citations = await self.generate_answer(ctx.question, ctx.history, hits)
            structured_output = await self.maybe_structured(
                ctx.question, hits, ctx.response_model
            )
            self.record_turn(ctx.record, ctx.session_id, ctx.question, answer)

        result = Pipeline(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            outputs={
                "answer": answer,
                "citations": citations,
                "hits": hits,
                "structured": structured_output,
                "history": ctx.history,
                "transforms_applied": transforms_applied,
                "resolved_config": context.metadata.get("resolved_config"),
            },
        )
        self.maybe_cache_store(result, ctx)
        return result

    @staticmethod
    def annotate_query_span(
        span: Any,
        user: Any | None,
        session_id: str | None,
    ) -> None:
        """Stamp user / session attributes on the active query span."""
        if user is not None:
            email = getattr(user, "email", None)
            if email:
                span.set_attribute("user_id", email)
        if session_id:
            span.set_attribute("session_id", session_id)

    async def retrieve_hits(
        self,
        question: str,
        history: list[Turn],
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter: dict[str, Any] | str,
    ) -> tuple[list[Hit], list[str]]:
        """Embed the query and retrieve (and optionally rerank) hits."""
        with self.telemetry.span("query.embed_query"):
            vector = self.embedder.embed_text(question)

        transformed = await self.maybe_transform(question, history, top_k)
        if transformed is not None:
            return transformed

        hits = self.vector_search(vector, top_k, rbac_filter)
        hits = self.filter_user_hits(hits, user_filter)
        if self.reranker is not None:
            with self.telemetry.span("query.rerank"):
                hits = self.reranker.rerank(question=question, hits=hits)
        if self.long_context_pass is not None and hits:
            with self.telemetry.span("query.long_context_pass"):
                hits = await self.long_context_pass.rerank(question=question, hits=hits)
        return hits, []

    async def maybe_transform(
        self,
        question: str,
        history: list[Turn],
        top_k: int,
    ) -> tuple[list[Hit], list[str]] | None:
        """Apply query transforms when configured; ``None`` falls back to plain search."""
        if self.transformer is None or self.retrieval_pipeline is None:
            return None
        variants = await self.transformer.transform(question=question, history=history)
        multi = [v for v in variants if v.text and v.text.strip()]
        if not (len(multi) > 1 or (len(multi) == 1 and multi[0].kind != "original")):
            return None
        with self.telemetry.span(
            "query.search_variants",
            count=len(multi),
            kinds=",".join(v.kind for v in multi),
        ):
            hits = self.retrieval_pipeline.retrieve_variants(
                user=None, variants=multi, top_k=top_k
            )
        return cast(list[Hit], hits), [v.kind for v in multi]

    def vector_search(
        self,
        vector: list[float],
        top_k: int,
        rbac_filter: dict[str, Any] | str,
    ) -> list[Hit]:
        """Run a plain vector-store search and convert raw records to ``Hit``."""
        with self.telemetry.span("query.search", top_k=top_k):
            raw = self.vector_store.search(
                vector=vector,
                top_k=top_k,
                metadata_filter=rbac_filter,
            )
        return [
            Hit(score=float(h["score"]), chunk=h["chunk"])
            for h in raw
        ]

    @staticmethod
    def filter_user_hits(
        hits: list[Hit],
        user_filter: dict[str, Any] | str,
    ) -> list[Hit]:
        """Drop hits that fail the per-user metadata filter."""
        if not (isinstance(user_filter, dict) and user_filter):
            return hits
        return [
            h
            for h in hits
            if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
        ]

    async def generate_answer(
        self,
        question: str,
        history: list[Turn],
        hits: list[Hit],
    ) -> tuple[Any, list[Citation]]:
        """Generate the answer and record token usage on the telemetry span."""
        citations = self.build_citations(hits)
        with self.telemetry.span("query.generate"):
            result = await awaitable(
                self.generator.generate(
                    question=question,
                    context=hits,
                    conversation=history,
                )
            )
            if isinstance(result, tuple):
                answer, citations = result
            else:
                answer = result
            await self.maybe_record_generate_tokens()
        return answer, citations

    @staticmethod
    def build_citations(hits: list[Hit]) -> list[Citation]:
        """Convert ``Hit`` objects into the facade's ``Citation`` shape."""
        return [
            Citation(
                chunk=h.chunk,
                document_id=h.chunk.document_id,
                version=h.chunk.version,
                page=h.chunk.page,
                section=h.chunk.section,
                quote=h.chunk.text,
                score=h.score,
                source_uri=h.chunk.source_location or h.chunk.document_id,
            )
            for h in hits
        ]

    async def maybe_record_generate_tokens(self) -> None:
        """Forward LLM token usage to the telemetry provider when available."""
        record_tokens = getattr(self.generator, "record_tokens", None)
        if not callable(record_tokens):
            return
        tokens = record_tokens()
        if inspect.isawaitable(tokens):
            tokens = await tokens
        if not isinstance(tokens, dict) or not tokens:
            return
        self.telemetry.record_tokens(
            "query.generate",
            prompt_tokens=int(tokens.get("prompt", 0)),
            completion_tokens=int(tokens.get("completion", 0)),
            model=str(tokens.get("model", "")),
        )

    async def maybe_structured(
        self,
        question: str,
        hits: list[Hit],
        response_model: Any,
    ) -> Any:
        """Run the structured-output provider when configured."""
        if self.structured is None or response_model is None:
            return None
        with self.telemetry.span("query.structured"):
            return await self.structured.generate(
                response_model=response_model,
                question=question,
                context=hits,
            )

    def record_turn(
        self,
        record: bool,
        session_id: str | None,
        question: str,
        answer: Any,
    ) -> None:
        """Append a turn to the conversation store when conditions allow."""
        if not (record and session_id and answer):
            return
        self.conversation_store.append(
            session_id,
            Turn(question=question, answer=str(answer)),
        )

    def maybe_cache_store(self, result: Pipeline, ctx: QueryContext) -> None:
        """Persist the pipeline result in the cache when configured."""
        if self.cache is None:
            return
        self.cache.set(
            ctx.question,
            ctx.user_id,
            ctx.user_filter,
            result,
            top_k=ctx.top_k,
            response_model=ctx.response_model,
            session_id=ctx.session_id,
            history=ctx.history,
            scope=ctx.scope,
        )

    async def stream(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict[str, Any] | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        rbac_filter = self.metadata_filter_for_user(user)

        with self.telemetry.span("query.stream", question=question[:128], top_k=top_k) as span:
            self.annotate_stream_span(span, user, session_id)
            hits = await self.stream_retrieve_hits(
                question, top_k, rbac_filter, user_filter
            )
            history: list[Turn] = []
            if session_id:
                history = self.conversation_store.load(session_id, limit=20)
            collected: list[str] = []
            async for piece in self.stream_answer(question, hits, history):
                if piece:
                    collected.append(piece)
                    yield piece
            self.stream_record_tokens()
            self.stream_record_turn(session_id, question, collected)

    @staticmethod
    def annotate_stream_span(
        span: Any,
        user: Any | None,
        session_id: str | None,
    ) -> None:
        """Stamp user / session attributes on the active stream span."""
        if user is not None and getattr(user, "email", None):
            span.set_attribute("user_id", user.email)
        if session_id:
            span.set_attribute("session_id", session_id)

    async def stream_retrieve_hits(
        self,
        question: str,
        top_k: int,
        rbac_filter: dict[str, Any] | str,
        user_filter: dict[str, Any] | str,
    ) -> list[Hit]:
        """Embed, search, optionally rerank, and return the streaming hits."""
        with self.telemetry.span("query.embed_query"):
            vector = self.embedder.embed_text(question)
        with self.telemetry.span("query.search"):
            raw = self.vector_store.search(
                vector=vector,
                top_k=top_k,
                metadata_filter=rbac_filter,
            )
        hits = [
            Hit(score=float(h["score"]), chunk=h["chunk"])
            for h in raw
        ]
        hits = QueryPipeline.filter_user_hits(hits, user_filter)
        if self.reranker is not None:
            with self.telemetry.span("query.rerank"):
                hits = self.reranker.rerank(question=question, hits=hits)
        if self.long_context_pass is not None and hits:
            with self.telemetry.span("query.long_context_pass"):
                hits = await self.long_context_pass.rerank(question=question, hits=hits)
        return hits

    async def stream_answer(
        self,
        question: str,
        hits: list[Hit],
        history: list[Turn],
    ) -> AsyncIterator[str]:
        """Yield tokens from the generator's ``astream`` method, if present."""
        astream = getattr(self.generator, "astream", None)
        if astream is None:
            return
        async for piece in astream(
            question=question,
            context=hits,
            conversation=history,
        ):
            yield piece

    def stream_record_tokens(self) -> None:
        """Forward streaming token usage to the telemetry provider."""
        record_tokens = getattr(self.generator, "record_tokens", None)
        if not callable(record_tokens):
            return
        tokens = record_tokens()
        if inspect.isawaitable(tokens):
            return
        if not isinstance(tokens, dict) or not tokens:
            return
        with self.telemetry.span("query.tokens") as tok_span:
            tok_span.set_attribute("prompt_tokens", int(tokens.get("prompt", 0)))
            tok_span.set_attribute(
                "completion_tokens", int(tokens.get("completion", 0))
            )
        self.telemetry.record_tokens(
            "query.stream",
            prompt_tokens=int(tokens.get("prompt", 0)),
            completion_tokens=int(tokens.get("completion", 0)),
            model=str(tokens.get("model", "")),
        )

    def stream_record_turn(
        self,
        session_id: str | None,
        question: str,
        collected: list[str],
    ) -> None:
        """Append the streamed answer to the conversation store."""
        if not (session_id and collected):
            return
        self.conversation_store.append(
            session_id,
            Turn(
                question=question,
                answer="".join(collected),
            ),
        )


# ---------------------------------------------------------------------------
# AgentPipeline
# ---------------------------------------------------------------------------


class AgentPipeline(PipelineRunner):
    """Query pipeline powered by the ReAct agent."""

    name = "query_agent"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        agent: Agent,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: GeneratorProtocol,
        **components: Any,
    ) -> None:
        """Initialise the agentic pipeline.

        Args:
            agent: The ReAct agent.
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Answer generator.
            **components: Optional collaborators — ``llm=``,
                ``telemetry=``, ``long_context_pass=``.

        """
        if agent is None:
            raise ValueError("AgentPipeline requires an Agent")
        self.agent = agent
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.llm = components.get("llm") or getattr(agent, "llm", None)
        self.telemetry = components.get("telemetry") or NoOpTelemetry()
        self.long_context_pass = components.get("long_context_pass")

    async def run(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Pipeline:
        """Run the agentic pipeline."""
        with DurationTimer(context):
            question: str = inputs["question"]
            user: User | None = inputs.get("user")
            session_id: str | None = inputs.get("session_id")
            tools_enabled: set[str] | None = inputs.get("tools_enabled")
            history: Sequence[Turn] = list(inputs.get("history") or [])
            top_k: int = int(inputs.get("top_k", 5))

            with self.telemetry.span("query_agent", question=question[:128]) as sp:
                if user is not None and getattr(user, "email", None):
                    sp.set_attribute("user_id", user.email)
                if session_id:
                    sp.set_attribute("session_id", session_id)

                trace = await self.agent.run(
                    AgentRequest(
                        question=question,
                        history=history,
                        tools_enabled=tools_enabled,
                        user=user,
                        session_id=session_id,
                    )
                )

                citations = citations_from_trace(trace)
                hits = hits_from_trace(trace, top_k)

                if (
                    self.long_context_pass is not None
                    and hits
                    and self.long_context_pass.is_eligible()
                ):
                    with self.telemetry.span("query_agent.long_context_pass"):
                        hits = await self.long_context_pass.rerank(question=question, hits=hits)

                agent_answer = trace.final_answer
                generator_result = await awaitable(
                    self.generator.generate(
                        question=question,
                        context=hits,
                        conversation=history,
                    )
                )
                generator_citations = (
                    generator_result[1] if isinstance(generator_result, tuple) else citations
                )
                if not generator_citations:
                    generator_citations = cast(list[Citation], citations)
                answer = agent_answer

            return Pipeline(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                outputs={
                    "answer": answer or trace.final_answer,
                    "citations": generator_citations,
                    "hits": hits,
                    "structured": None,
                    "history": list(history),
                    "transforms_applied": [],
                    "resolved_config": context.metadata.get("resolved_config"),
                    "planner_trace": [event.model_dump(mode="json") for event in trace.events],
                    "tools_invoked": list(trace.tools_invoked),
                    "agent_trace": trace.to_dict(),
                },
            )

    async def astream(
        self,
        context: PipelineCtx,
        **inputs: Any,
    ) -> Any:
        """Async-iterate :class:`raghub.agent.PlannerEvent`."""
        question: str = inputs["question"]
        user: User | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        tools_enabled: set[str] | None = inputs.get("tools_enabled")
        history: Sequence[Turn] = list(inputs.get("history") or [])
        async for event in self.agent.astream(
            AgentRequest(
                question=question,
                history=history,
                tools_enabled=tools_enabled,
                user=user,
                session_id=session_id,
            )
        ):
            yield event


def citations_from_trace(trace: AgentTrace) -> list[dict[str, Any]]:
    """Build citation dicts from the agent's tool observations."""
    citations: list[dict[str, Any]] = []
    for observation in trace.observations:
        for hit in observation.get("data", {}).get("hits", []) or []:
            citations.append(
                {
                    "document_id": hit.get("document_id"),
                    "chunk_id": hit.get("chunk_id"),
                    "score": hit.get("score"),
                    "source": observation.get("name"),
                }
            )
    return citations


def hits_from_trace(trace: AgentTrace, top_k: int) -> list[Any]:
    """Reconstruct :class:`Hit` instances from observations."""
    hits: list[Hit] = []
    for observation in trace.observations:
        name = observation.get("name", "")
        if name not in {
            "vector_search",
            "keyword_search",
            "hybrid_search",
            "summary_search",
            "graph_search",
        }:
            continue
        for hit in observation.get("data", {}).get("hits", []) or []:
            text = hit.get("text", "")
            record = Chunk(
                id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id") or "graphrag://summary",
                version=1,
                page=1,
                source_location=name,
                section="",
                company="",
                owner="",
                department="",
                text=text,
                checksum=sha256(text.encode("utf-8")).hexdigest(),
                metadata={"source_tool": name, **hit.get("metadata", {})},
            )
            hits.append(
                Hit(
                    score=float(hit.get("score", 0.0) or 0.0),
                    chunk=record,
                )
            )
    deduped: dict[str, Hit] = {}
    for hit in hits:
        prior = deduped.get(hit.chunk_id)
        if prior is None or hit.score > prior.score:
            deduped[hit.chunk_id] = hit
    ordered = sorted(deduped.values(), key=lambda h: h.score, reverse=True)
    return ordered[: int(top_k)]
