"""Default RAG pipeline — ingest + query orchestrations.

Both pipelines accept an optional :class:`TelemetryProvider` for
observability. When supplied, the pipeline wraps each stage in a
span and records latency, embedding/model token usage where
available, and per-stage error events.

Incremental indexing: the :class:`IngestPipeline` computes a
SHA-256 checksum of the file bytes and short-circuits the work
when an existing knowledge bundle for the same checksum is
already in the :class:`KnowledgeRepository`. Callers can force
re-indexing by passing ``force=True`` in the inputs.

Multi-user support: the :class:`QueryPipeline` accepts an optional
``user: UserPrincipal`` input. The pipeline derives a metadata
filter from the user's ``allowed_companies`` (admins see
everything) and forwards the filter to the vector store. The LLM
only ever sees authorised context.

Conversation support: the :class:`QueryPipeline` accepts an
optional ``session_id`` input. The pipeline loads the most recent
turns from the global :class:`ConversationManager` and prepends
them to the prompt so the LLM can answer follow-up questions
without the caller managing history.
"""

from __future__ import annotations

import time
from hashlib import sha256
from typing import Any, AsyncIterator

from raghub.converters.plaintext import PlainTextConverter
from raghub.exceptions import PipelineError
from raghub.ingestion.chunkers.word_window import WordWindowChunker
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
from raghub.knowledge.repository import InMemoryKnowledgeRepository
from raghub.models import (
    Chunk,
    KnowledgeBundle,
    PipelineContext,
    PipelineResult,
    deterministic_id,
)
from raghub.observability.noop import NoOpTelemetry


def chunks_from_knowledge_bundle(
    bundle: KnowledgeBundle, document_id: str, company: str = ""
) -> list[Chunk]:
    """Materialise the :class:`Chunk` list for a bundle's sections.

    Args:
        bundle: The source knowledge bundle.
        document_id: Document id to install on every chunk.
        company: Tenant (company) tag; falls back to ``bundle.metadata``.

    Returns:
        The list of :class:`Chunk` records.
    """
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
    """Return the primary company for a :class:`UserPrincipal`.

    Args:
        user: The :class:`UserPrincipal` (or any duck-typed object
            with ``allowed_companies``). Admin users and users
            without an allow-list are returned as the empty string
            (no per-document tenant restriction).

    Returns:
        The first ``allowed_companies`` entry, or ``""``.
    """
    if user is None:
        return ""
    companies = getattr(user, "allowed_companies", None) or []
    if getattr(user, "is_admin", False):
        return ""
    if not companies:
        return ""
    return str(companies[0])


class IngestPipeline(Pipeline):
    """Convert → chunk → embed → index pipeline.

    Supports incremental indexing: when the same SHA-256 has been
    ingested before, the pipeline returns the prior chunk ids
    without re-embedding.
    """

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
    ) -> None:
        """Initialise the ingest pipeline.

        Args:
            converter: Document converter. Falls back to plaintext.
            chunker: Chunker. Defaults to :class:`WordWindowChunker`.
            embedder: Embedding provider. **Required.**
            vector_store: Vector store. **Required.**
            knowledge_repo: Optional knowledge repository.
            telemetry: Optional telemetry provider.
        """
        if embedder is None or vector_store is None:
            raise PipelineError("IngestPipeline requires embedder and vector_store")
        self.converter = converter or PlainTextConverter()
        self.chunker = chunker or WordWindowChunker()
        self.embedder = embedder
        self.vector_store = vector_store
        self.knowledge_repo = knowledge_repo or InMemoryKnowledgeRepository()
        self.telemetry = telemetry or NoOpTelemetry()

    async def run(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> PipelineResult:
        """Run the ingest pipeline.

        Required inputs: ``file_bytes``, ``source_uri``, ``mime_type``.
        Optional: ``language``, ``metadata``, ``force``, ``company``,
        ``user``.

        When ``user`` is provided, the user's email is recorded as
        the chunk owner and the user's primary company (the first
        entry in ``allowed_companies``) is used as the document
        tenant.

        Returns:
            A :class:`PipelineResult` with ``bundle``, ``chunks``,
            ``chunk_count``, ``embeddings``, ``incremental`` keys.
        """
        started = time.perf_counter()
        try:
            file_bytes: bytes = inputs["file_bytes"]
            source_uri: str = inputs["source_uri"]
            mime_type: str = inputs.get("mime_type", "")
            language: str = inputs.get("language", "")
            metadata: dict | None = inputs.get("metadata")
            force: bool = bool(inputs.get("force", False))
            user: Any | None = inputs.get("user")
            tenant_company: str = inputs.get("company", "") or primary_company(user)

            checksum = sha256_checksum(file_bytes)
            bundle_id = deterministic_id("bundle", source_uri, checksum)
            with self.telemetry.span(
                "ingest", source_uri=source_uri, bundle_id=bundle_id
            ) as sp:
                sp.set_attribute("checksum", checksum)

                # Incremental short-circuit: an unchanged file is
                # recognised by its SHA-256.
                existing = (
                    self.knowledge_repo.get(bundle_id)
                    if not force
                    else None
                )
                if existing is not None and existing.checksum == checksum:
                    return PipelineResult(
                        pipeline_id=context.pipeline_id,
                        pipeline_name=self.name,
                        success=True,
                        outputs={
                            "bundle": existing,
                            "chunks": chunks_from_knowledge_bundle(
                                existing, bundle_id, company=tenant_company
                            ),
                            "chunk_count": sum(
                                sum(1 for b in s.blocks if b.kind.value == "text")
                                for s in existing.sections
                            ),
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
                        metadata={**(metadata or {}), "company": tenant_company},
                    )
                bundle.bundle_id = bundle_id
                bundle.checksum = checksum

                self.knowledge_repo.save(bundle)

                with self.telemetry.span("ingest.chunk"):
                    chunks = chunks_from_knowledge_bundle(
                        bundle, bundle_id, company=tenant_company
                    )
                    if user is not None:
                        for c in chunks:
                            c.owner = getattr(user, "email", c.owner) or c.owner

                texts = [chunk.text for chunk in chunks]
                with self.telemetry.span("ingest.embed", count=len(texts)):
                    vectors = self.embedder.embed_texts(texts) if texts else []

                with self.telemetry.span("ingest.upsert", count=len(chunks)):
                    if chunks and vectors:
                        self.vector_store.upsert(chunks, vectors)

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
        except Exception as exc:
            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=False,
                error=str(exc),
            )
        finally:
            context.metadata["duration_ms"] = (time.perf_counter() - started) * 1000.0


class QueryPipeline(Pipeline):
    """Embed → retrieve → rerank → generate pipeline.

    The pipeline enforces RBAC at the retrieval step. When ``user``
    is provided, the pipeline derives a metadata filter from the
    user's ``allowed_companies`` and forwards it to the vector
    store. Admins see every company. The LLM only ever sees the
    filtered hits — no unauthorised context can leak through.

    The pipeline also loads recent conversation turns from
    ``conversation_store`` (when ``session_id`` is provided) and
    prepends them to the prompt so the LLM can answer follow-up
    questions in context.
    """

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
    ) -> None:
        """Initialise the query pipeline.

        Args:
            embedder: Embedding provider.
            vector_store: Vector store.
            generator: Generator.
            reranker: Reranker. Defaults to identity.
            structured: Optional structured-output provider.
            telemetry: Optional telemetry provider.
            conversation_store: Optional pluggable conversation
                store. Defaults to an in-memory store so
                :class:`QueryPipeline` always has a working backend.
            cache: Optional :class:`QueryCache` instance. When set,
                the pipeline checks the cache before running and
                stores results after a successful run.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker
        self.structured = structured
        self.telemetry = telemetry or NoOpTelemetry()
        if conversation_store is None:
            from raghub.conversation.memory import InMemoryConversationStore

            conversation_store = InMemoryConversationStore()
        self.conversation_store = conversation_store
        self.cache = cache

    def metadata_filter_for_user(self, user: Any) -> dict | str:
        """Derive a metadata filter for the vector store from a user.

        Args:
            user: The :class:`UserPrincipal` (or any duck-typed
                object with ``is_admin`` and ``allowed_companies``).

        Returns:
            A dict for stores that accept dict filters (e.g.
                :class:`InMemoryVectorStore`) or a string for stores
                that accept SQL-like filters (e.g. legacy zvec). An
                admin returns ``""`` (no filter). A user with no
                allow-list returns a filter that matches nothing
                (``{"company": []}``) so the LLM never sees
                unauthorised content.
        """
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
        """Run the query pipeline.

        Required inputs: ``question``. Optional: ``top_k`` (default 5),
        ``metadata_filter`` (dict), ``user`` (UserPrincipal for
        RBAC), ``session_id`` (loads conversation history),
        ``response_model`` (a Pydantic class) to request a typed
        response, ``record`` (when ``True``, append the turn to the
        conversation store).
        """
        started = time.perf_counter()
        try:
            question: str = inputs["question"]
            top_k: int = int(inputs.get("top_k", 5))
            user_filter: dict | str = inputs.get("metadata_filter") or {}
            user: Any | None = inputs.get("user")
            session_id: str | None = inputs.get("session_id")
            response_model = inputs.get("response_model")
            record: bool = bool(inputs.get("record", True))
            from raghub.models import RetrievalHit

            # Query cache check.
            if self.cache is not None:
                user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
                cached = self.cache.get(question, user_id, dict(user_filter) if isinstance(user_filter, dict) else None)
                if cached is not None:
                    return cached

            # RBAC: derive the metadata filter from the user.
            rbac_filter = self.metadata_filter_for_user(user)

            with self.telemetry.span("query", question=question[:128], top_k=top_k) as span:
                if user is not None:
                    email = getattr(user, "email", None)
                    if email:
                        span.set_attribute("user_id", email)
                if session_id:
                    span.set_attribute("session_id", session_id)

                with self.telemetry.span("query.embed_query"):
                    vector = self.embedder.embed_text(question)

                with self.telemetry.span("query.search", top_k=top_k):
                    raw = self.vector_store.search(
                        vector=vector,
                        top_k=top_k,
                        metadata_filter=rbac_filter,
                    )
                hits: list[RetrievalHit] = [
                    RetrievalHit(
                        chunk_id=h["chunk_id"],
                        score=float(h["score"]),
                        chunk=h["chunk"],
                    )
                    for h in raw
                ]
                # Apply additional user-supplied filter post-hoc
                # (the in-memory store accepts both dict and str).
                if isinstance(user_filter, dict) and user_filter:
                    hits = [
                        h
                        for h in hits
                        if all(
                            getattr(h.chunk, k, None) == v
                            for k, v in user_filter.items()
                        )
                    ]
                if self.reranker is not None:
                    with self.telemetry.span("query.rerank"):
                        hits = self.reranker.rerank(question=question, hits=hits)

                # Load conversation history for follow-up questions.
                history: list = []
                if session_id:
                    try:
                        history = self.conversation_store.load(session_id, limit=20)
                    except Exception:
                        history = []

                answer: Any
                citations: list = []
                with self.telemetry.span("query.generate"):
                    answer, citations = await self.generator.generate(
                        question=question,
                        context=hits,
                        conversation=history,
                    )
                    if hasattr(self.generator, "record_tokens"):
                        tokens = getattr(self.generator, "record_tokens", lambda: None)()
                        if tokens:
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

                # Record the turn in the conversation store.
                if record and session_id and answer:
                    from raghub.models import ConversationTurn

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
                },
            )
            if self.cache is not None:
                user_id = getattr(user, "email", None) or getattr(user, "user_id", None)
                self.cache.set(question, user_id, dict(user_filter) if isinstance(user_filter, dict) else None, result)
            return result
        except Exception as exc:
            return PipelineResult(
                pipeline_id=context.pipeline_id,
                pipeline_name=self.name,
                success=False,
                error=str(exc),
            )
        finally:
            context.metadata["duration_ms"] = (time.perf_counter() - started) * 1000.0

    async def stream(
        self,
        context: PipelineContext,
        **inputs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token.

        Args:
            context: Per-invocation state.
            **inputs: Same as :meth:`run` (``question``, ``top_k``,
                ``metadata_filter``, ``user``, ``session_id``).

        Yields:
            String chunks of the answer.
        """
        from raghub.models import RetrievalHit

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
                    if all(
                        getattr(h.chunk, k, None) == v
                        for k, v in user_filter.items()
                    )
                ]
            if self.reranker is not None:
                with self.telemetry.span("query.rerank"):
                    hits = self.reranker.rerank(question=question, hits=hits)
            history: list = []
            if session_id:
                try:
                    history = self.conversation_store.load(session_id, limit=20)
                except Exception:
                    history = []
            astream = getattr(self.generator, "astream", None)
            if astream is not None:
                collected: list[str] = []
                async for piece in astream(
                    question=question, context=hits, conversation=history
                ):
                    if piece:
                        collected.append(piece)
                        yield piece
                # Propagate token usage from the underlying LLM.
                if hasattr(self.generator, "record_tokens"):
                    tokens = self.generator.record_tokens()
                    if tokens:
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
                    from raghub.models import ConversationTurn

                    self.conversation_store.append(
                        session_id,
                        ConversationTurn(
                            question=question,
                            answer="".join(collected),
                        ),
                    )
                return
            # Fallback: non-streaming generator.
            answer, _ = await self.generator.generate(
                question=question, context=hits, conversation=history
            )
            if session_id and answer:
                from raghub.models import ConversationTurn

                self.conversation_store.append(
                    session_id, ConversationTurn(question=question, answer=str(answer))
                )
            for word in answer.split():
                yield word + " "


__all__ = ["IngestPipeline", "QueryPipeline"]
