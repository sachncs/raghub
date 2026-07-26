"""Orchestration pipelines — ingest, query, agentic, cache, timing.

This is the **wiring** file. It composes every other module
(retrieval, embeddings, vector store, generation, knowledge,
observability, agent, conversation) into the public pipeline
surface. The file is large because the wiring itself is the
framework's value-add; co-locating every pipeline keeps the
single-responsibility split legible.

Section map:

* :class:`DurationTimer` — context manager that records pipeline
  wall-clock duration onto the ``PipelineContext.metadata``.
* :class:`QueryCache` — TTL-based in-memory query cache.
* :class:`ConversationRouter` — facade over a pluggable conversation
  store.
* :class:`PipelineResultBuilder` — fluent builder for
  :class:`PipelineResult` records.
* :class:`IngestPipeline` — convert → chunk → embed → index.
* :class:`QueryPipeline` — embed → retrieve → rerank → generate.
* :class:`AgenticQueryPipeline` — agent-driven query pipeline.
* :func:`chunks_from_knowledge_bundle` / :func:`primary_company` /
  :func:`sha256_checksum` — small ingest helpers.
* :func:`citations_from_trace` / :func:`hits_from_trace` —
  agent-trace → citation/hit coercion.
* :func:`canonical_filters` — flatten filters into hashable tuples.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractContextManager
from hashlib import sha256
from types import TracebackType
from typing import Any

from tqdm import tqdm

from raghub.agent import Agent, AgentTrace
from raghub.conversation import InMemoryConversationStore
from raghub.documents import PlainTextConverter
from raghub.embeddings import BaseEmbeddingProvider
from raghub.exceptions import PipelineError
from raghub.interfaces.chunker import Chunker
from raghub.interfaces.converter import DocumentConverter
from raghub.interfaces.embeddings import EmbeddingProvider
from raghub.interfaces.generator import Generator
from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.interfaces.observability import TelemetryProvider
from raghub.interfaces.pipeline import Pipeline
from raghub.interfaces.retrieval import Reranker
from raghub.interfaces.structured import StructuredOutputProvider
from raghub.interfaces.vectorstore import VectorStore
from raghub.knowledge import InMemoryKnowledgeRepository
from raghub.llm import BaseLLMProvider
from raghub.models import (
    Chunk,
    ChunkRecord,
    Classification,
    ConversationTurn,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    RetrievalHit,
    UserPrincipal,
    deterministic_id,
)
from raghub.observability import NoOpTelemetry

# ---------------------------------------------------------------------------
# DurationTimer (formerly raghub.pipelines._timing)
# ---------------------------------------------------------------------------


class DurationTimer(AbstractContextManager["DurationTimer"]):
    """Set ``context.metadata["duration_ms"]`` on exit."""

    def __init__(self, context: Any) -> None:
        """Store the context; the start time is captured on entry."""
        self.context = context
        self.start: float = 0.0

    def __enter__(self) -> "DurationTimer":
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
# QueryCache (formerly raghub.pipelines.cache)
# ---------------------------------------------------------------------------


def canonical_filters(filters: dict | str | None) -> tuple:
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


class QueryCache:
    """Simple TTL-based in-memory query cache."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self.store: dict[tuple, tuple[float, PipelineResult]] = {}

    def make_key(
        self,
        question: str,
        user_id: str | None,
        filters: dict | str | None,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> tuple:
        """Build the cache key for the given query context."""
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
        filters: dict | str | None = None,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> PipelineResult | None:
        """Return a cached :class:`PipelineResult` or ``None``."""
        key = self.make_key(
            question,
            user_id,
            filters,
            top_k=top_k,
            response_model=response_model,
            session_id=session_id,
            history=history,
            scope=scope,
        )
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
        filters: dict | str | None,
        result: PipelineResult,
        *,
        top_k: int = 5,
        response_model: Any | None = None,
        session_id: str | None = None,
        history: Sequence[Any] = (),
        scope: Any = None,
    ) -> None:
        """Store a :class:`PipelineResult` in the cache."""
        key = self.make_key(
            question,
            user_id,
            filters,
            top_k=top_k,
            response_model=response_model,
            session_id=session_id,
            history=history,
            scope=scope,
        )
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
# ConversationRouter (formerly raghub.pipelines.rag.conversation)
# ---------------------------------------------------------------------------


class ConversationRouter:
    """Thin facade over a pluggable conversation store."""

    def __init__(self, store: Any) -> None:
        """Store the backing conversation store reference."""
        self.store = store

    def load_history(self, session_id: str | None, limit: int = 20) -> list[Any]:
        """Return the recent turns for ``session_id``."""
        if not session_id:
            return []
        return self.store.load(session_id, limit=limit)

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
# PipelineResultBuilder (formerly raghub.pipelines.rag.result)
# ---------------------------------------------------------------------------


class PipelineResultBuilder:
    """Fluent builder for :class:`PipelineResult` records."""

    def __init__(self, context: PipelineContext, pipeline_name: str) -> None:
        """Store the context and pipeline name for subsequent builds."""
        self.context = context
        self.pipeline_name = pipeline_name

    def success(self, outputs: dict[str, Any]) -> PipelineResult:
        """Build a successful :class:`PipelineResult` with ``outputs``."""
        return PipelineResult(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            success=True,
            outputs=outputs,
        )

    def failure(self, error: str, outputs: dict[str, Any] | None = None) -> PipelineResult:
        """Build a failed :class:`PipelineResult` with ``error``."""
        return PipelineResult(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            success=False,
            error=error,
            outputs=outputs,
        )


# ---------------------------------------------------------------------------
# Ingest helpers (formerly raghub.pipelines.rag.ingest)
# ---------------------------------------------------------------------------


def chunks_from_knowledge_bundle(
    bundle: KnowledgeBundle, document_id: str, company: str = ""
) -> list[Chunk]:
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
                    chunk_id=chunk_id,
                    document_id=document_id,
                    version=1,
                    page=(section.page_numbers[0] if section.page_numbers else section.index),
                    source_location=section.source_location or bundle.source_uri,
                    section=section.heading,
                    company=tenant_company,
                    owner=bundle.metadata.get("owner", ""),
                    department=bundle.metadata.get("department", ""),
                    text=text,
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
    """Return the primary company for a :class:`UserPrincipal`."""
    if user is None:
        return ""
    companies = getattr(user, "allowed_companies", None) or []
    if getattr(user, "is_admin", False):
        return ""
    if not companies:
        return ""
    return str(companies[0])


class IngestPipeline(Pipeline):
    """Convert → chunk → embed → index pipeline."""

    name: str = "ingest"

    def __init__(
        self,
        *,
        converter: DocumentConverter | None = None,
        chunker: Chunker | None = None,
        embedder: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        knowledge_repo: KnowledgeRepository | None = None,
        telemetry: TelemetryProvider | None = None,
        raptor: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        """Initialise the ingest pipeline."""
        from raghub.ingestion import WordWindowChunker

        if embedder is None or vector_store is None:
            raise PipelineError("IngestPipeline requires embedder and vector_store")
        self.converter = converter or PlainTextConverter()
        self.chunker = chunker or WordWindowChunker()
        self.embedder = embedder
        self.vector_store = vector_store
        self.knowledge_repo = knowledge_repo or InMemoryKnowledgeRepository()
        self.telemetry = telemetry or NoOpTelemetry()
        self.raptor = raptor
        self.graph = graph
        self.show_progress = True

    def vectors_already_indexed(self, chunks: list[Chunk]) -> bool:
        """Return ``True`` when every chunk already lives in the vector store."""
        if not chunks:
            return True
        has_chunk = getattr(self.vector_store, "has_chunk", None)
        if not callable(has_chunk):
            return True
        return all(bool(has_chunk(chunk.chunk_id)) for chunk in chunks)

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the ingest pipeline."""
        with DurationTimer(context):
            file_bytes: bytes = inputs["file_bytes"]
            source_uri: str = inputs["source_uri"]
            mime_type: str = inputs.get("mime_type", "")
            language: str = inputs.get("language", "")
            metadata: dict[str, Any] = dict(inputs.get("metadata") or {})
            force: bool = bool(inputs.get("force", False))
            user: Any | None = inputs.get("user")
            tenant_company: str = str(
                inputs.get("company") or primary_company(user) or metadata.get("company", "")
            )
            checksum = sha256_checksum(file_bytes)
            bundle_id = deterministic_id("bundle", source_uri, checksum)
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
            normalized_metadata = {
                **metadata,
                "company": tenant_company,
                "owner": owner,
                "classification": classification.value,
                "document_id": document_id,
                "version": version,
            }
            with self.telemetry.span("ingest", source_uri=source_uri, bundle_id=bundle_id) as sp:
                sp.set_attribute("checksum", checksum)

                existing = self.knowledge_repo.get(bundle_id) if not force else None
                if existing is not None and existing.checksum == checksum:
                    prior_chunks = chunks_from_knowledge_bundle(
                        existing, document_id, company=tenant_company
                    )
                    if self.vectors_already_indexed(prior_chunks):
                        return PipelineResult(
                            pipeline_id=context.pipeline_id,
                            pipeline_name=self.name,
                            success=True,
                            outputs={
                                "bundle": existing,
                                "chunks": prior_chunks,
                                "chunk_count": len(prior_chunks),
                                "embeddings": [],
                                "incremental": True,
                            },
                        )

                with self.telemetry.span("ingest.convert"):
                    bundle: KnowledgeBundle = self.converter.convert(
                        source_uri=source_uri,
                        file_bytes=file_bytes,
                        mime_type=mime_type,
                        language=language,
                        metadata=normalized_metadata,
                    )
                bundle.bundle_id = bundle_id
                bundle.checksum = checksum
                bundle.metadata = {**bundle.metadata, **normalized_metadata}

                with self.telemetry.span("ingest.chunk"):
                    raw_chunks = self.chunker.chunk(bundle)
                    chunks: list = []
                    for chunk in tqdm(
                        raw_chunks,
                        desc="Chunking",
                        disable=not getattr(self, "show_progress", True),
                        unit="chunk",
                    ):
                        chunk.document_id = document_id
                        chunk.version = version
                        chunk.company = tenant_company
                        chunk.owner = owner
                        chunk.classification = classification
                        chunks.append(chunk)

                texts = [chunk.text for chunk in chunks]
                with self.telemetry.span("ingest.embed", count=len(texts)):
                    vectors = self.embedder.embed_texts(texts) if texts else []

                with self.telemetry.span("ingest.upsert", count=len(chunks)):
                    if chunks:
                        self.vector_store.upsert(chunks, vectors)
                        if self.raptor is not None:
                            with self.telemetry.span("ingest.raptor"):
                                self.raptor.add_chunks(chunks, vectors)
                        if self.graph is not None:
                            with self.telemetry.span("ingest.graph"):
                                self.graph.add_chunks(chunks, vectors)

                self.knowledge_repo.save(bundle)

                return PipelineResult(
                    pipeline_id=context.pipeline_id,
                    pipeline_name=self.name,
                    success=True,
                    outputs={
                        "bundle": bundle,
                        "chunks": chunks,
                        "chunk_count": len(chunks),
                        "embeddings": vectors,
                        "incremental": False,
                    },
                )


# ---------------------------------------------------------------------------
# QueryPipeline (formerly raghub.pipelines.rag.query)
# ---------------------------------------------------------------------------


class QueryPipeline(Pipeline):
    """Embed → retrieve → rerank → generate pipeline."""

    name: str = "query"

    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        reranker: Reranker | None = None,
        structured: StructuredOutputProvider | None = None,
        telemetry: TelemetryProvider | None = None,
        conversation_store: Any | None = None,
        cache: Any | None = None,
        transformer: Any | None = None,
        retrieval_pipeline: Any | None = None,
        long_context_pass: Any | None = None,
        agentic_pipeline: Any | None = None,
    ) -> None:
        """Initialise the query pipeline."""
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker
        self.structured = structured
        self.telemetry = telemetry or NoOpTelemetry()
        if conversation_store is None:
            conversation_store = InMemoryConversationStore()
        self.conversation_store = conversation_store
        self.cache = cache
        self.transformer = transformer
        self.retrieval_pipeline = retrieval_pipeline
        self.long_context_pass = long_context_pass
        self.agentic_pipeline = agentic_pipeline

    def metadata_filter_for_user(self, user: Any) -> dict | str:
        """Derive a metadata filter for the vector store from a user."""
        if user is None:
            return ""
        if getattr(user, "is_admin", False):
            return ""
        companies = list(getattr(user, "allowed_companies", []) or [])
        return {"company": companies}

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the query pipeline."""
        with DurationTimer(context):
            return await self.run_inner(context, inputs)

    async def run_inner(
        self,
        context: PipelineContext,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        """Body of :meth:`run` separated so the timing ``finally`` is obvious."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        response_model = inputs.get("response_model")
        record: bool = bool(inputs.get("record", True))
        tools_enabled: set[str] | None = inputs.get("tools_enabled")

        history: list = []
        if session_id:
            history = self.conversation_store.load(session_id, limit=20)

        rbac_filter = self.metadata_filter_for_user(user)
        user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
        scope = (
            bool(getattr(user, "is_admin", False)),
            tuple(sorted(str(value) for value in getattr(user, "allowed_companies", []) or [])),
            tuple(sorted(str(value) for value in getattr(user, "allowed_groups", []) or [])),
        )
        if self.cache is not None:
            cached = self.cache.get(
                question,
                user_id,
                user_filter,
                top_k=top_k,
                response_model=response_model,
                session_id=session_id,
                history=history,
                scope=scope,
            )
            if isinstance(cached, PipelineResult):
                return cached

        if self.agentic_pipeline is not None and (
            tools_enabled
            or ((record_overrides := inputs.get("resolved_config")) is not None
            and (
                record_overrides.get("agent_enabled")
                or record_overrides.get("tools_enabled")
            ))
        ):
            return await self.agentic_pipeline.run(
                context,
                question=question,
                user=user,
                session_id=session_id,
                tools_enabled=tools_enabled,
                top_k=top_k,
                history=history,
            )

        with self.telemetry.span("query", question=question[:128], top_k=top_k) as span:
            if user is not None:
                email = getattr(user, "email", None)
                if email:
                    span.set_attribute("user_id", email)
            if session_id:
                span.set_attribute("session_id", session_id)

            with self.telemetry.span("query.embed_query"):
                vector = self.embedder.embed_text(question)

            transforms_applied: list[str] = []
            if self.transformer is not None and self.retrieval_pipeline is not None:
                variants = await self.transformer.transform(
                    question=question, history=history
                )
                multi = [v for v in variants if v.text and v.text.strip()]
                if len(multi) > 1 or (
                    len(multi) == 1 and multi[0].kind != "original"
                ):
                    transforms_applied = [v.kind for v in multi]
                    with self.telemetry.span(
                        "query.search_variants",
                        count=len(multi),
                        kinds=",".join(transforms_applied),
                    ):
                        hits = self.retrieval_pipeline.retrieve_variants(
                            user=user, variants=multi, top_k=top_k
                        )
                else:
                    hits = None
            else:
                hits = None

            if hits is None:
                with self.telemetry.span("query.search", top_k=top_k):
                    raw = self.vector_store.search(
                        vector=vector,
                        top_k=top_k,
                        metadata_filter=rbac_filter,
                    )
                hits = [
                    RetrievalHit(
                        chunk_id=h["chunk_id"],
                        score=float(h["score"]),
                        chunk=h["chunk"],
                    )
                    for h in raw
                ]
                if isinstance(user_filter, dict) and user_filter:
                    hits = [
                        h
                        for h in hits
                        if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
                    ]
                if self.reranker is not None:
                    with self.telemetry.span("query.rerank"):
                        hits = self.reranker.rerank(question=question, hits=hits)
            if self.long_context_pass is not None and hits:
                with self.telemetry.span("query.long_context_pass"):
                    hits = await self.long_context_pass.rerank(
                        question=question, hits=hits
                    )

            answer: Any
            citations: list = []
            with self.telemetry.span("query.generate"):
                answer, citations = await self.generator.generate(
                    question=question,
                    context=hits,
                    conversation=history,
                )
                record_tokens = getattr(self.generator, "record_tokens", None)
                if callable(record_tokens):
                    tokens = record_tokens()
                    if inspect.isawaitable(tokens):
                        tokens = await tokens
                    if isinstance(tokens, dict) and tokens:
                        self.telemetry.record_tokens(
                            "query.generate",
                            prompt_tokens=int(tokens.get("prompt", 0)),
                            completion_tokens=int(tokens.get("completion", 0)),
                            model=str(tokens.get("model", "")),
                        )

            structured_output: Any = None
            if self.structured is not None and response_model is not None:
                with self.telemetry.span("query.structured"):
                    structured_output = await self.structured.generate(
                        response_model=response_model,
                        question=question,
                        context=hits,
                    )

            if record and session_id and answer:
                self.conversation_store.append(
                    session_id,
                    ConversationTurn(
                        question=question,
                        answer=str(answer),
                    ),
                )

        result = PipelineResult(
            pipeline_id=context.pipeline_id,
            pipeline_name=self.name,
            success=True,
            outputs={
                "answer": answer,
                "citations": citations,
                "hits": hits,
                "structured": structured_output,
                "history": history,
                "transforms_applied": transforms_applied,
                "resolved_config": context.metadata.get("resolved_config"),
            },
        )
        if self.cache is not None:
            self.cache.set(
                question,
                user_id,
                user_filter,
                result,
                top_k=top_k,
                response_model=response_model,
                session_id=session_id,
                history=history,
                scope=scope,
            )
        return result

    async def stream(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token."""
        question: str = inputs["question"]
        top_k: int = int(inputs.get("top_k", 5))
        user_filter: dict | str = inputs.get("metadata_filter") or {}
        user: Any | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        rbac_filter = self.metadata_filter_for_user(user)

        with self.telemetry.span("query.stream", question=question[:128], top_k=top_k) as span:
            if user is not None and getattr(user, "email", None):
                span.set_attribute("user_id", user.email)
            if session_id:
                span.set_attribute("session_id", session_id)
            with self.telemetry.span("query.embed_query"):
                vector = self.embedder.embed_text(question)
            with self.telemetry.span("query.search"):
                raw = self.vector_store.search(
                    vector=vector,
                    top_k=top_k,
                    metadata_filter=rbac_filter,
                )
            hits = [
                RetrievalHit(
                    chunk_id=h["chunk_id"],
                    score=float(h["score"]),
                    chunk=h["chunk"],
                )
                for h in raw
            ]
            if isinstance(user_filter, dict) and user_filter:
                hits = [
                    h
                    for h in hits
                    if all(getattr(h.chunk, k, None) == v for k, v in user_filter.items())
                ]
            if self.reranker is not None:
                with self.telemetry.span("query.rerank"):
                    hits = self.reranker.rerank(question=question, hits=hits)
            if self.long_context_pass is not None and hits:
                with self.telemetry.span("query.long_context_pass"):
                    hits = await self.long_context_pass.rerank(
                        question=question, hits=hits
                    )
            history: list = []
            if session_id:
                history = self.conversation_store.load(session_id, limit=20)
            astream = getattr(self.generator, "astream", None)
            if astream is not None:
                collected: list[str] = []
                async for piece in astream(question=question, context=hits, conversation=history):
                    if piece:
                        collected.append(piece)
                        yield piece
                record_tokens = getattr(self.generator, "record_tokens", None)
                if callable(record_tokens):
                    tokens = record_tokens()
                    if inspect.isawaitable(tokens):
                        tokens = await tokens
                    if isinstance(tokens, dict) and tokens:
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
                if session_id and collected:
                    self.conversation_store.append(
                        session_id,
                        ConversationTurn(
                            question=question,
                            answer="".join(collected),
                        ),
                    )
                return
            answer, _ = await self.generator.generate(
                question=question, context=hits, conversation=history
            )
            if session_id and answer:
                self.conversation_store.append(
                    session_id, ConversationTurn(question=question, answer=str(answer))
                )
            for word in answer.split():
                yield word + " "


# ---------------------------------------------------------------------------
# AgenticQueryPipeline (formerly raghub.pipelines.agentic)
# ---------------------------------------------------------------------------


class AgenticQueryPipeline:
    """Query pipeline powered by the ReAct agent."""

    name = "query_agent"

    def __init__(
        self,
        *,
        agent: Agent,
        embedder: BaseEmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        llm: BaseLLMProvider | None = None,
        telemetry: TelemetryProvider | None = None,
        long_context_pass: Any | None = None,
    ) -> None:
        """Initialise the agentic pipeline."""
        if agent is None:
            raise ValueError("AgenticQueryPipeline requires an Agent")
        self.agent = agent
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.llm = llm or getattr(agent, "llm", None)
        self.telemetry = telemetry or NoOpTelemetry()
        self.long_context_pass = long_context_pass

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the agentic pipeline."""
        with DurationTimer(context):
            question: str = inputs["question"]
            user: UserPrincipal | None = inputs.get("user")
            session_id: str | None = inputs.get("session_id")
            tools_enabled: set[str] | None = inputs.get("tools_enabled")
            history: Sequence[ConversationTurn] = list(inputs.get("history") or [])
            top_k: int = int(inputs.get("top_k", 5))

            with self.telemetry.span("query_agent", question=question[:128]) as sp:
                if user is not None and getattr(user, "email", None):
                    sp.set_attribute("user_id", user.email)
                if session_id:
                    sp.set_attribute("session_id", session_id)

                trace = await self.agent.run(
                    question=question,
                    history=history,
                    tools_enabled=tools_enabled,
                    user=user,
                    session_id=session_id,
                )

                citations = citations_from_trace(trace)
                hits = hits_from_trace(trace, top_k)

                if (
                    self.long_context_pass is not None
                    and hits
                    and self.long_context_pass.is_eligible()
                ):
                    with self.telemetry.span("query_agent.long_context_pass"):
                        hits = await self.long_context_pass.rerank(
                            question=question, hits=hits
                        )

                agent_answer = trace.final_answer
                _generator_text, generator_citations = await self.generator.generate(
                    question=question,
                    context=hits,
                    conversation=history,
                )
                if not generator_citations:
                    generator_citations = citations
                answer = agent_answer

            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=True,
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
        context: PipelineContext,
        **inputs: Any,
    ) -> Any:
        """Async-iterate :class:`raghub.agent.PlannerEvent`."""
        question: str = inputs["question"]
        user: UserPrincipal | None = inputs.get("user")
        session_id: str | None = inputs.get("session_id")
        tools_enabled: set[str] | None = inputs.get("tools_enabled")
        history: Sequence[ConversationTurn] = list(inputs.get("history") or [])
        async for event in self.agent.astream(
            question=question,
            history=history,
            tools_enabled=tools_enabled,
            user=user,
            session_id=session_id,
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
    """Reconstruct :class:`RetrievalHit` instances from observations."""
    hits: list[RetrievalHit] = []
    for observation in trace.observations:
        name = observation.get("name", "")
        if name not in {"vector_search", "keyword_search", "hybrid_search", "summary_search", "graph_search"}:
            continue
        for hit in observation.get("data", {}).get("hits", []) or []:
            record = ChunkRecord(
                chunk_id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id") or "graphrag://summary",
                version=1,
                page=1,
                source_location=name,
                section="",
                company="",
                owner="",
                department="",
                text=hit.get("text", ""),
                metadata={"source_tool": name, **hit.get("metadata", {})},
            )
            hits.append(
                RetrievalHit(
                    chunk_id=record.chunk_id,
                    score=float(hit.get("score", 0.0) or 0.0),
                    chunk=record,
                )
            )
    deduped: dict[str, RetrievalHit] = {}
    for hit in hits:
        prior = deduped.get(hit.chunk_id)
        if prior is None or hit.score > prior.score:
            deduped[hit.chunk_id] = hit
    ordered = sorted(deduped.values(), key=lambda h: h.score, reverse=True)
    return ordered[: int(top_k)]