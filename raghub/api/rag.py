"""Public RAGHub facade.

A single recommended entry point that wires every spec-mandated
component — Marker → OKF → Chonkie → LiteLLM → Qdrant/ZVec →
Langfuse → Instructor — behind a ``RAG(...)`` builder and a
``RAG.from_config("raghub.yaml")`` helper.

Quick start (fewer than 10 lines of Python)::

    from raghub import RAG

    rag = RAG()
    rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
    print(rag.query("revenue").answer)

The facade supports sync (``ingest``, ``query``, ``evaluate``),
async (``aingest``, ``aquery``, ``astream``), and streaming
(``astream``) entry points. All public methods return or accept
typed Pydantic models from :mod:`raghub.models`; raw dictionaries
are never exchanged across the public boundary.

Multi-user isolation
--------------------

Conversation history is keyed by both ``session_id`` **and** the
caller's ``UserPrincipal``. The facade namespaces keys internally
so that two callers who happen to share or guess a ``session_id``
cannot read each other's history. :meth:`conversation_history`
and :meth:`clear_conversation` both accept a ``user`` argument;
the public surface mirrors the rest of the RBAC contract.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Sequence, TypeVar, cast

from pydantic import BaseModel

from raghub.api.async_runner import maybe_await
from raghub.interfaces.generator import Generator
from raghub.api.defaults import (
    default_chunker,
    default_converter,
    default_embedder,
    default_llm,
    default_structured,
    default_telemetry,
    default_vector_store,
)
from raghub.api.response import build_response
from raghub.config.settings import AppSettings, load_settings
from raghub.conversation.memory import InMemoryConversationStore
from raghub.evaluation.financebench import FinanceBenchEvaluator
from raghub.exceptions import ConfigurationError, RagHubError
from raghub.generation.generator import DefaultGenerator
from raghub.ingestion.resumable import ResumableBackgroundIngestionService
from raghub.knowledge.manifest import SourceManifest
from raghub.knowledge.repository import InMemoryKnowledgeRepository
from raghub.models import (
    CanonicalResponse as Response,
    EvaluationResult,
    PipelineContext,
    PipelineResult,
    deterministic_id,
)
from raghub.observability.redact import RedactingTelemetry
from raghub.pipelines.rag import IngestPipeline, QueryPipeline
from raghub.plugins.registry import PluginRegistry
from raghub.retrieval.reranker import IdentityReranker

T = TypeVar("T", bound=BaseModel)


class RAG:
    """High-level RAGHub facade.

    Construct via :meth:`RAG.from_config` for the standard
    configuration-driven path, or pass components directly for
    advanced customisation. Every collaborator is replaceable
    through the constructor.

    Attributes:
        settings: The configuration snapshot.
        registry: The plugin registry.
        converter: Document converter (Marker by default).
        chunker: Chunker (Chonkie by default).
        embedder: Embedding provider (LiteLLM by default).
        llm: LLM provider (LiteLLM by default).
        vector_store: Vector store (Qdrant by default).
        generator: Answer generator (wraps ``llm``).
        knowledge_repo: Knowledge repository.
        structured: Structured-output provider (Instructor by default).
        telemetry: Telemetry provider (Langfuse by default; redacted).
        reranker: Reranker (IdentityReranker by default).
        manifest: Source manifest for incremental indexing.
        background_ingestion: Background ingestion service.
    """

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        converter: Any = None,
        chunker: Any = None,
        embedder: Any = None,
        llm: Any = None,
        vector_store: Any = None,
        generator: Any = None,
        reranker: Any = None,
        knowledge_repo: Any = None,
        structured: Any = None,
        telemetry: Any = None,
        registry: Any = None,
        background_service: Any = None,
        manifest: Any = None,
    ) -> None:
        """Initialise the facade.

        Args:
            settings: Configuration; default uses
                :func:`load_settings`.
            converter: Document converter. Defaults to
                :class:`MarkerConverter` (with
                :class:`PlainTextConverter` fallback).
            chunker: Chunker. Defaults to Chonkie (with
                :class:`WordWindowChunker` fallback).
            embedder: Embedding provider. Defaults to
                :class:`LiteLLMEmbeddingProvider` (with
                :class:`HashingEmbeddingProvider` fallback).
            llm: LLM provider. Defaults to
                :class:`LiteLLMProvider` (with
                :class:`HeuristicLLMProvider` fallback).
            vector_store: Vector store. Defaults to
                :class:`QdrantVectorStore` (with
                :class:`InMemoryVectorStore` fallback).
            generator: Answer generator. Defaults to
                :class:`DefaultGenerator` wrapping ``llm``.
            reranker: Reranker. Defaults to
                :class:`IdentityReranker`.
            knowledge_repo: Knowledge repository. Defaults to
                :class:`InMemoryKnowledgeRepository`.
            structured: Structured-output provider. Defaults to
                :class:`InstructorStructuredOutputProvider`; falls
                back to ``None`` when Instructor is not installed.
            telemetry: Telemetry provider. Defaults to Langfuse
                (when credentials are present); falls back to
                :class:`NoOpTelemetry`. The default is wrapped in
                :class:`RedactingTelemetry` to scrub secrets.
            registry: Optional plugin registry.
            background_service: Optional background ingestion
                service. A
                :class:`ResumableBackgroundIngestionService` is
                instantiated on demand when callers invoke
                :meth:`ingest_async`.
            manifest: Optional source manifest. Defaults to a
                ``manifest.json`` next to the data directory.
        """
        self.settings = settings or load_settings()
        self.registry = registry or PluginRegistry()

        self.knowledge_repo = knowledge_repo or InMemoryKnowledgeRepository()
        self.vector_store = vector_store or default_vector_store(
            self.settings.embedding_dim
        )
        self.embedder = embedder or default_embedder(
            self.settings.embedding_model, self.settings.embedding_dim
        )
        self.llm = llm or default_llm(self.settings.llm_model)
        self.converter = converter or default_converter()
        self.chunker = chunker or default_chunker(
            self.settings.chunk_size_words, self.settings.chunk_overlap_words
        )
        self.reranker = reranker or IdentityReranker()
        self.generator = cast(Generator, generator or DefaultGenerator(llm=self.llm))
        self.structured = (
            structured if structured is not None else default_structured()
        )

        if telemetry is None:
            inner = default_telemetry()
            self.telemetry: Any = RedactingTelemetry(inner)
        else:
            self.telemetry = telemetry

        self.ingest_pipeline = IngestPipeline(
            converter=self.converter,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            knowledge_repo=self.knowledge_repo,
            telemetry=self.telemetry,
        )
        self.conversation_store: Any = InMemoryConversationStore()
        from raghub.pipelines.cache import QueryCache

        self.query_cache: QueryCache | None = (
            QueryCache(ttl_seconds=self.settings.query_cache_ttl_seconds)
            if self.settings.enable_query_cache
            else None
        )
        self.query_pipeline = QueryPipeline(
            embedder=self.embedder,
            vector_store=self.vector_store,
            generator=self.generator,
            reranker=self.reranker,
            structured=self.structured,
            telemetry=self.telemetry,
            conversation_store=self.conversation_store,
            cache=self.query_cache,
        )

        self.manifest: SourceManifest = manifest or SourceManifest(
            self.settings.data_dir / "manifest.json"
        )
        self.background_ingestion = background_service

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path) -> "RAG":
        """Build a facade from a YAML or TOML configuration file.

        Args:
            path: Path to a YAML or TOML file compatible with
                :class:`AppSettings`.

        Returns:
            A configured :class:`RAG` instance.
        """
        from raghub.config.settings import AppSettings

        p = Path(path)
        if p.suffix.lower() == ".toml":
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
            payload = tomllib.loads(p.read_text(encoding="utf-8")) or {}
        else:
            import yaml

            payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        settings = AppSettings(
            **{k: v for k, v in payload.items() if k in AppSettings.model_fields}
        )
        settings.ensure_dirs()
        return cls(settings=settings)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise held resources (vector store, knowledge repo)."""
        if hasattr(self.vector_store, "create_collection"):
            self.vector_store.create_collection()
        if hasattr(self.knowledge_repo, "initialize"):
            self.knowledge_repo.initialize()

    def shutdown(self) -> None:
        """Release all held resources; safe to call multiple times.

        Closes the telemetry provider, the vector store, the
        knowledge repository, and the background ingestion
        service. Errors from any single collaborator are swallowed
        so the rest of the shutdown still completes. The LLM,
        embedder, and generator are also closed when they expose
        a ``close()`` method.
        """
        if hasattr(self.telemetry, "end_trace"):
            try:
                self.telemetry.end_trace()
            except Exception:
                pass
        for collaborator in (
            self.vector_store,
            self.knowledge_repo,
            getattr(self, "background_ingestion", None),
            getattr(self, "embedder", None),
            getattr(self, "llm", None),
            getattr(self, "generator", None),
        ):
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(result)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | Path | bytes,
        *,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
        user: Any | None = None,
    ) -> PipelineResult:
        """Ingest a file, directory, or raw bytes synchronously.

        Args:
            source: Path to a file/directory or raw bytes.
            source_uri: Override the source URI (when ``source`` is
                raw bytes).
            mime_type: MIME hint for raw bytes.
            metadata: Optional extra metadata.
            force: When ``True``, bypass incremental-indexing dedup
                and always re-embed.
            user: Optional :class:`UserPrincipal`. When set, the
                user's email is recorded as the chunk owner and the
                user's primary company is used as the document
                tenant.

        Returns:
            A :class:`PipelineResult` for a single source, or a
            composite result for a directory.

        Raises:
            RagHubError: When the source bytes are empty.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                return self.ingest_directory_sync(p, metadata, user)
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = source_uri or "bytes://memory"
        if not file_bytes:
            raise RagHubError(
                f"ingest({source!r}) received empty bytes; nothing to index."
            )
        return maybe_await(
            self.ingest_one_async(file_bytes, uri, mime_type, metadata, force, user)
        )

    def ingest_directory_sync(
        self, directory: Path, metadata: dict[str, Any] | None, user: Any | None
    ) -> PipelineResult:
        """Recursively ingest a directory synchronously."""
        from raghub.models import PipelineResult

        results: list[PipelineResult] = []
        for child in sorted(directory.rglob("*")):
            if not child.is_file():
                continue
            results.append(self.ingest(child, metadata=metadata, user=user))
        return PipelineResult(
            pipeline_id="batch",
            pipeline_name="ingest",
            success=all(r.success for r in results),
            outputs={"batch": results},
        )

    async def aingest(
        self,
        source: str | Path | bytes,
        *,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        force: bool = False,
        user: Any | None = None,
    ) -> PipelineResult:
        """Async version of :meth:`ingest`.

        Raises:
            RagHubError: When the source bytes are empty.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                return await self.ingest_directory_async(p, metadata, user)
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = source_uri or "bytes://memory"
        if not file_bytes:
            raise RagHubError(
                f"aingest({source!r}) received empty bytes; nothing to index."
            )
        return await self.ingest_one_async(
            file_bytes, uri, mime_type, metadata, force, user
        )

    async def ingest_directory_async(
        self, directory: Path, metadata: dict[str, Any] | None, user: Any | None
    ) -> PipelineResult:
        """Recursively ingest a directory asynchronously."""
        from raghub.models import PipelineResult

        results: list[PipelineResult] = []
        for child in sorted(directory.rglob("*")):
            if not child.is_file():
                continue
            results.append(await self.aingest(child, metadata=metadata, user=user))
        return PipelineResult(
            pipeline_id="batch",
            pipeline_name="ingest",
            success=all(r.success for r in results),
            outputs={"batch": results},
        )

    async def ingest_one_async(
        self,
        file_bytes: bytes,
        source_uri: str,
        mime_type: str,
        metadata: dict[str, Any] | None,
        force: bool = False,
        user: Any | None = None,
    ) -> PipelineResult:
        """Run a single ingest pipeline asynchronously."""
        context = PipelineContext(
            pipeline_name="ingest",
            metadata={"user_id": getattr(user, "email", None)} if user is not None else {},
        )
        return await self.ingest_pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=source_uri,
            mime_type=mime_type,
            metadata=metadata or {},
            force=force,
            user=user,
        )

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, document_id: str) -> None:
        """Delete a document and all of its chunks.

        Accepts either a bundle id (the deterministic
        ``document_id`` recorded on each chunk) or a source URI
        (the ``source_uri`` argument supplied to :meth:`ingest`).
        """
        target_ids: set[str] = {document_id}
        if hasattr(self.knowledge_repo, "list_by_source"):
            for bundle in self.knowledge_repo.list_by_source(document_id):
                target_ids.add(bundle.bundle_id)
        for tid in target_ids:
            if hasattr(self.vector_store, "delete_document"):
                self.vector_store.delete_document(tid)
            if hasattr(self.knowledge_repo, "delete"):
                self.knowledge_repo.delete(tid)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        *,
        user: Any | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        response_model: type | None = None,
    ) -> Response:
        """Ask a question and return a typed :class:`Response`."""
        return maybe_await(
            self.aquery(
                question,
                user=user,
                session_id=session_id,
                top_k=top_k,
                metadata_filter=metadata_filter,
                response_model=response_model,
            )
        )

    @staticmethod
    def scoped_session_id(user: Any, session_id: str | None) -> str | None:
        """Combine ``user`` and ``session_id`` into a single opaque key.

        The conversation store is keyed by this combined value so two
        callers who happen to share or guess a ``session_id`` cannot
        read each other's history. When ``user`` is ``None`` the
        method returns the raw ``session_id`` (back-compat behaviour
        for tests that exercise the in-process store anonymously).

        Args:
            user: The :class:`UserPrincipal` (or any duck-typed
                object with ``user_id`` / ``email`` attributes).
            session_id: The caller-supplied session id.

        Returns:
            The namespaced key, or ``None`` when no session id is set.
        """
        if session_id is None:
            return None
        if user is None:
            return session_id
        uid = (
            getattr(user, "user_id", None)
            or getattr(user, "email", None)
            or "anonymous"
        )
        return f"{uid}::{session_id}"

    async def aquery(
        self,
        question: str,
        *,
        user: Any | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        response_model: type | None = None,
    ) -> Response:
        """Async version of :meth:`query`."""
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineContext(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )
        result = await self.query_pipeline.run(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            response_model=response_model,
            user=user,
            session_id=scoped,
        )
        if not result.success:
            raise RagHubError(result.error or "query failed")
        return build_response(result)

    async def astream(
        self,
        question: str,
        *,
        user: Any | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token via the LLM's ``astream``."""
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineContext(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )
        async for piece in self.query_pipeline.stream(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            user=user,
            session_id=scoped,
        ):
            yield piece

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        benchmark: str = "financebench",
        *,
        response_factory: Callable[[dict], Any] | None = None,
        examples: Sequence[dict] | None = None,
    ) -> list[EvaluationResult]:
        """Run a benchmark evaluation."""
        if benchmark != "financebench":
            raise ConfigurationError(f"Unknown benchmark: {benchmark!r}")

        evaluator = FinanceBenchEvaluator()
        factory = response_factory

        async def coerce_answer(example: dict) -> Any:
            """Coerce the result of ``response_factory`` to a coroutine.

            Args:
                example: The benchmark example dict.

            Returns:
                The factory's response, or the live :meth:`aquery`
                answer when no factory is provided.
            """
            if factory is None:
                return await self.aquery(example.get("question", ""))
            result = factory(example)
            if inspect.isawaitable(result):
                return await result
            return result

        return maybe_await(evaluator.evaluate(examples, response_factory=coerce_answer))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return a health summary for the facade."""
        return {
            "status": "ok",
            "vector_store": type(self.vector_store).__name__,
            "embedder": type(self.embedder).__name__,
            "llm": type(self.llm).__name__,
            "chunker": type(self.chunker).__name__,
            "converter": type(self.converter).__name__,
            "telemetry": type(self.telemetry).__name__,
            "structured": type(self.structured).__name__ if self.structured else None,
            "reranker": type(self.reranker).__name__,
        }

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def sync_index(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
    ) -> dict[str, list[str]]:
        """Reconcile ``directory`` against the manifest."""
        from raghub.knowledge.manifest import sha256_bytes

        directory = Path(directory)
        if not directory.is_dir():
            raise RagHubError(f"{directory} is not a directory")

        seen: set[str] = set()
        summary: dict[str, list[str]] = {
            "added": [],
            "modified": [],
            "unchanged": [],
            "removed": [],
        }

        for child in sorted(directory.rglob("*")):
            if not child.is_file():
                continue
            uri = str(child.resolve())
            seen.add(uri)
            data = child.read_bytes()
            checksum = sha256_bytes(data)
            prior = self.manifest[uri] if uri in self.manifest else None
            if prior is None:
                self.ingest(child, metadata=metadata, user=user)
                self.manifest.record(
                    uri,
                    bundle_id=deterministic_id("bundle", uri, checksum),
                    checksum=checksum,
                )
                summary["added"].append(uri)
            elif prior.get("checksum") != checksum:
                self.ingest(child, metadata=metadata, force=True, user=user)
                self.manifest.record(
                    uri,
                    bundle_id=deterministic_id("bundle", uri, checksum),
                    checksum=checksum,
                )
                summary["modified"].append(uri)
            else:
                summary["unchanged"].append(uri)

        for prior_uri in self.manifest.sources():
            if prior_uri in seen:
                continue
            if not prior_uri.startswith(str(directory.resolve())):
                continue
            bundle_id = self.manifest[prior_uri].get("bundle_id", "")
            self.delete(bundle_id)
            self.manifest.remove(prior_uri)
            summary["removed"].append(prior_uri)

        self.manifest.save()
        return summary

    def ingest_async(
        self,
        source: str | Path | bytes,
        *,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
    ) -> str:
        """Submit an ingest job to the background service."""
        if self.background_ingestion is None:
            self.background_ingestion = ResumableBackgroundIngestionService(
                db_path=self.settings.data_dir / "ingestion_jobs.db"
            )

        if isinstance(source, (str, Path)):
            p = Path(source)
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = source_uri or "bytes://memory"

        return self.background_ingestion.submit(
            self.ingest,
            source=file_bytes,
            source_uri=uri,
            mime_type=mime_type,
            metadata=metadata,
            user=user,
        )

    def job_status(self, job_id: str) -> str | None:
        """Return the status of a background ingestion job."""
        if self.background_ingestion is None:
            return None
        return self.background_ingestion.get_status(job_id)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def conversation_history(
        self,
        session_id: str,
        *,
        user: Any | None = None,
        limit: int = 50,
    ) -> list:
        """Return the most recent conversation turns for a session.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`UserPrincipal` whose
                ``user_id`` / ``email`` scopes the lookup. When
                omitted, the lookup uses the raw ``session_id`` and
                will only return history created with ``user=None``
                — preventing accidental cross-user reads.
            limit: Maximum number of turns to return.

        Returns:
            The list of :class:`ConversationTurn` records, oldest
            first.
        """
        scoped = self.scoped_session_id(user, session_id) or session_id
        return self.conversation_store.load(scoped, limit=limit)

    def clear_conversation(
        self,
        session_id: str,
        *,
        user: Any | None = None,
    ) -> None:
        """Clear a session's conversation history.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`UserPrincipal` whose
                ``user_id`` / ``email`` scopes the delete. When
                omitted, the raw ``session_id`` is used.
        """
        scoped = self.scoped_session_id(user, session_id) or session_id
        self.conversation_store.clear(scoped)


__all__ = ["RAG"]
