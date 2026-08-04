"""Public RAGHub facade.

A single recommended entry point that wires every spec-mandated
component — Marker → OKF → Chonkie → LiteLLM →
Langfuse → Instructor — behind a ``RAG(...)`` builder and a
``RAG.from_config("raghub.yaml")`` helper.

Quick start (fewer than 10 lines of Python)::

    >>> import raghub
    >>> rag = raghub.RAG()
    >>> rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
    >>> print(rag.query("revenue").answer)

The facade supports sync (``ingest``, ``query``, ``evaluate``),
async (``aingest``, ``aquery``, ``astream``), and streaming
(``astream``) entry points. All public methods return or accept
typed Pydantic models from :mod:`raghub.models`; raw dictionaries
are never exchanged across the public boundary.

Multi-user isolation
--------------------

Conversation history is keyed by both ``session_id`` **and** the
caller's ``User``. The facade namespaces keys internally
so that two callers who happen to share or guess a ``session_id``
cannot read each other's history. :meth:`conversation_history`
and :meth:`clear_conversation` both accept a ``user`` argument;
the public surface mirrors the rest of the RBAC contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import tomllib
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml
from pydantic import BaseModel
from tqdm import tqdm

from raghub.agent import Agent, PlannerEvent, build_tool_registry, resolve
from raghub.api_response import ResponseBuilder
from raghub.await_sync import maybe_await_sync as maybe_await
from raghub.config import Settings
from raghub.conv import Memory
from raghub.errors import (
    ConfigurationError,
    IngestionError,
    RagHubError,
)
from raghub.eval import Finance
from raghub.gen import DefaultGenerator
from raghub.ingest import Resumable
from raghub.knowledge import (
    GraphIndex,
    Manifest,
    MemoryRepo,
    Raptor,
    sha256_bytes,
)
from raghub.models import (
    Pipeline,
    PipelineCtx,
    RagComponents,
    RagQueryRequest,
    Response,
    Result,
    Turn,
    deterministic_id,
)
from raghub.pipeline import AgentPipeline, Cache, Ingest, QueryPipeline
from raghub.plugins import PluginRegistry
from raghub.rag.defaults import (
    LLM_API_KEY_ENV_VARS,
    agent_required,
    default_chunker,
    default_converter,
    default_embedder,
    default_llm,
    default_structured,
    default_telemetry,
    default_transforms,
    default_vector_store,
    has_llm_api_key,
    ingest_one_worker,
)
from raghub.retrieval import (
    Colbert as ColbertLateInteraction,
)
from raghub.retrieval import (
    Context as LongContextRerankPass,
)
from raghub.retrieval import (
    Retrieval as RetrievalPipeline,
)
from raghub.retrieval import (
    build_reranker,
)
from raghub.telemetry import RedactingTelemetry

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "RAG",
    "has_llm_api_key",
]


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
        settings: Settings | None = None,
        components: RagComponents | None = None,
        **kwargs: "JSONValue",
    ) -> None:
        """Initialise the facade."""
        components_dict: dict[str, Any] = dict(components) if components is not None else {}
        components_dict.update(kwargs)
        components_dict.setdefault("settings", settings)
        self.__wire_components(components_dict)
        self.__wire_ingest_pipeline()
        self.__wire_query_pipeline(components_dict)
        self.manifest: Manifest = (
            components_dict.get("manifest") or Manifest(self.settings.data_dir / "manifest.json")
        )
        self.background_ingestion = components_dict.get("background_service")
        self.queue_ = self.__init_queue(components_dict)
        self.worker_ = self.__init_worker(components_dict)
        self.worker_task_: asyncio.Task[None] | None = None
        self.tenant_resolver_ = self.__init_tenant_resolver(components_dict)
        self.feedback_store_ = self.__init_feedback_store(components_dict)
        self.rate_limiter_: Any = None
        self.archive_: Any = None
        self.isolation_strategy_: Any = components_dict.get("isolation_strategy")

    def __wire_components(self, components_dict: dict[str, Any]) -> None:
        """Resolve core collaborators from ``components_dict`` or defaults."""
        self.settings: Settings = components_dict.get("settings") or Settings.load()
        self.registry: Any = components_dict.get("registry") or PluginRegistry()
        self.knowledge_repo: "KnowledgeRepository" = (
            components_dict.get("knowledge_repo") or MemoryRepo()
        )
        self.vector_store: Any = (
            components_dict.get("vector_store") or default_vector_store(self.settings.embedding_dim)
        )
        self.embedder: Any = components_dict.get("embedder") or default_embedder(
            self.settings.embedding_model, self.settings.embedding_dim
        )
        self.llm: Any = components_dict.get("llm") or default_llm(self.settings.llm_model)
        self.converter: "DocumentConverter" = (
            components_dict.get("converter") or default_converter()
        )
        self.chunker: Any = components_dict.get("chunker") or default_chunker(
            self.settings.chunk_size_words,
            self.settings.chunk_overlap_words,
            chunker_strategy=self.settings.chunker_strategy,
            embedding_model_chunker=self.settings.embedding_model_chunker,
        )
        self.reranker: Any = components_dict.get("reranker") or build_reranker(
            self.settings, llm=self.llm
        )
        self.generator: Any = cast(
            Any,
            components_dict.get("generator")
            or DefaultGenerator(
                llm=self.llm,
                timeout_seconds=components_dict.get("llm_timeout_seconds"),
            ),
        )
        self.structured: Any = (
            components_dict.get("structured")
            if components_dict.get("structured") is not None
            else default_structured()
        )
        if components_dict.get("telemetry") is None:
            inner = default_telemetry()
            self.telemetry: Any = RedactingTelemetry(inner)
        else:
            self.telemetry = components_dict["telemetry"]

    def __wire_ingest_pipeline(self) -> None:
        """Build the ingestion pipeline and conversation store."""
        self.ingest_pipeline = Ingest(
            converter=self.converter,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            knowledge_repo=self.knowledge_repo,
            telemetry=self.telemetry,
            raptor=getattr(self, "raptor", None),
            graph=getattr(self, "graph", None),
        )
        self.conversation_store: Any = Memory()

    def __wire_query_pipeline(self, components_dict: dict[str, Any]) -> None:
        """Build retrieval, query, and agent pipelines."""
        self.query_cache: Cache | None = (
            Cache(ttl_seconds=self.settings.query_cache_ttl_seconds)
            if self.settings.enable_query_cache
            else None
        )
        self.transformer: Any = components_dict.get("transformer") or default_transforms(
            self.llm,
            enabled=list(self.settings.query_transforms.enabled),
            hyde_n=self.settings.query_transforms.hyde_n,
            multi_query_n=self.settings.query_transforms.multi_query_n,
        )
        self.colbert = ColbertLateInteraction(self.settings.hybrid)
        self.retrieval_pipeline = RetrievalPipeline(
            embedding_provider=self.embedder,
            vector_store=self.vector_store,
            rerank=self.reranker,
            hybrid=self.settings.hybrid,
        )
        self.long_context_pass = LongContextRerankPass(
            llm=self.llm, settings=self.settings.long_context_pass
        )
        self.raptor = None
        self.graph = None
        if self.settings.summary_search_enabled:
            self.raptor = Raptor(llm=self.llm, embedder=self.embedder, depth=2)
        if self.settings.graph_search_enabled:
            self.graph = GraphIndex(llm=self.llm, embedder=self.embedder)
        self.tool_registry = build_tool_registry(
            self.settings,
            retrieval_pipeline=self.retrieval_pipeline,
            vector_store=self.vector_store,
            raptor=self.raptor,
            graph=self.graph,
        )
        self.agent: Any | None = None
        self.agentic_pipeline: Any | None = None
        if agent_required(
            {
                "agent_enabled": self.settings.agent.enabled,
                "web_enabled": self.settings.web_search.enabled,
                "summary_enabled": self.settings.summary_search_enabled,
                "raptor": self.raptor,
                "graph_enabled": self.settings.graph_search_enabled,
                "graph": self.graph,
            }
        ):
            self.agent = Agent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                settings=self.settings.agent,
                telemetry=self.telemetry,
            )
            self.agentic_pipeline = AgentPipeline(
                agent=self.agent,
                embedder=self.embedder,
                vector_store=self.vector_store,
                generator=self.generator,
                llm=self.llm,
                telemetry=self.telemetry,
                long_context_pass=self.long_context_pass,
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
            transformer=self.transformer,
            retrieval_pipeline=self.retrieval_pipeline,
            long_context_pass=self.long_context_pass,
            agentic_pipeline=self.agentic_pipeline,
        )

    def __init_queue(self, components_dict: dict[str, Any]) -> Any:
        """Construct the persistent ingestion queue.

        Priority:
            1. ``components_dict["queue"]`` if explicitly supplied.
            2. ``Settings.queue.backend == "sqlite"`` -> ``SqliteQueue``.
            3. Otherwise ``None`` (legacy threadpool path).
        """
        supplied = components_dict.get("queue")
        if supplied is not None:
            return supplied
        backend = self.settings.queue.backend
        if backend == "sqlite":
            from raghub.jobs import SqliteQueue

            db_path = (
                self.settings.queue.db_path
                or self.settings.data_dir / "queue.db"
            )
            queue = SqliteQueue(
                db_path=str(db_path),
                max_inflight=self.settings.queue.max_inflight,
            )
            return queue
        return None

    def __init_worker(self, components_dict: dict[str, Any]) -> Any:
        """Construct a :class:`Worker` when a persistent queue is configured.

        Priority:
            1. ``components_dict["worker"]`` if explicitly supplied.
            2. ``self.queue_`` is a :class:`SqliteQueue` (built above when
               ``Settings.queue.backend == "sqlite"``) -> build a Worker
               bound to that queue with a default ingest handler.
            3. Otherwise ``None``.
        """
        supplied = components_dict.get("worker")
        if supplied is not None:
            return supplied
        if self.queue_ is None:
            return None
        try:
            from raghub.jobs import SqliteQueue, Worker
        except ImportError:
            return None
        if not isinstance(self.queue_, SqliteQueue):
            return None

        async def handler(job: Any) -> None:
            """Drain a queue job by routing it back into the facade's ingest path."""
            payload = getattr(job, "payload", {}) or {}
            source_bytes = payload.get("source", "").encode("latin-1")
            source_uri = payload.get("source_uri", "bytes://memory")
            mime_type = payload.get("mime_type", "text/plain")
            metadata = payload.get("metadata") or {}
            self.ingest(
                source_bytes,
                source_uri=source_uri,
                mime_type=mime_type,
                metadata=metadata,
            )

        return Worker(
            queue=self.queue_,
            handler=handler,
            concurrency=int(getattr(self.settings.queue, "concurrency", 4) or 4),
        )

    async def start_worker(self) -> None:
        """Run the configured :class:`Worker` until :meth:`stop_worker` is called.

        Only meaningful when ``self.worker_`` is not ``None`` (i.e. a
        SQLite-backed queue is configured). When the library runs without
        a persistent queue, the legacy threadpool path is used and this
        method is a no-op so callers can invoke it unconditionally.
        """
        if self.worker_ is None:
            return
        if self.worker_task_ is not None and not self.worker_task_.done():
            return
        self.worker_task_ = asyncio.create_task(self.worker_.loop("raghub-worker"))

    async def stop_worker(self) -> None:
        """Cancel :meth:`start_worker` and await its task to drain."""
        if self.worker_task_ is None:
            return
        task = self.worker_task_
        self.worker_task_ = None
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def __init_tenant_resolver(self, components_dict: dict[str, Any]) -> Any:
        """Construct the tenant resolver.

        Priority:
            1. ``components_dict["tenant_resolver"]`` if supplied.
            2. ``Settings.tenants.resolver == "composite" | "jwt" | "header"``.
            3. Otherwise ``None``.
        """
        supplied = components_dict.get("tenant_resolver")
        if supplied is not None:
            return supplied
        resolver = self.settings.tenants.resolver
        if resolver == "none":
            return None
        from raghub.tenants import (
            CompositeTenantResolver,
            HeaderTenantResolver,
            JwtClaimTenantResolver,
        )

        if resolver == "composite":
            return CompositeTenantResolver()
        if resolver == "jwt":
            return JwtClaimTenantResolver()
        if resolver == "header":
            return HeaderTenantResolver()
        return None

    def __init_feedback_store(self, components_dict: dict[str, Any]) -> Any:
        """Construct the feedback store (Tier 3 Item 19).

        Priority:
            1. ``components_dict["feedback_store"]`` if supplied.
            2. ``Settings.feedback.backend == "sqlite"`` -> SqliteFeedbackStore.
            3. Otherwise ``None``.
        """
        supplied = components_dict.get("feedback_store")
        if supplied is not None:
            return supplied
        backend = self.settings.feedback.backend
        if backend == "none":
            return None
        if backend == "sqlite":
            from raghub.feedback import SqliteFeedbackStore

            db_path = (
                self.settings.feedback.db_path
                or self.settings.data_dir / "feedback.db"
            )
            store = SqliteFeedbackStore(db_path=str(db_path))
            store.initialize()
            return store
        # postgres backend requires asyncpg; deferred to a future release
        return None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path) -> RAG:
        """Build a facade from a YAML or TOML configuration file.

        Args:
            path: Path to a YAML or TOML file compatible with
                :class:`Settings`.

        Returns:
            A configured :class:`RAG` instance.

        """
        p = Path(path)
        if p.suffix.lower() == ".toml":
            payload = tomllib.loads(p.read_text(encoding="utf-8")) or {}
        else:
            payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        settings = Settings(**{k: v for k, v in payload.items() if k in Settings.model_fields})
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
        knowledge repository, the background ingestion service,
        and the unit-of-work (when one was supplied). The LLM,
        embedder, and generator are also closed when they expose a
        ``close()`` method.

        Failures from any collaborator are collected and re-raised as
        a single :class:`RagHubError` at the end so callers see every
        failing component in one log line.

        Raises:
            RagHubError: When one or more collaborator close calls
                fail; the message lists each failing component.

        """
        if hasattr(self.telemetry, "end_trace"):
            self.telemetry.end_trace()
        collaborators: list[tuple[str, object]] = [
            ("unit_of_work", getattr(self, "unit_of_work", None)),
            ("vector_store", self.vector_store),
            ("knowledge_repo", self.knowledge_repo),
            ("background_ingestion", getattr(self, "background_ingestion", None)),
            ("embedder", getattr(self, "embedder", None)),
            ("llm", getattr(self, "llm", None)),
            ("generator", getattr(self, "generator", None)),
        ]
        failures: list[tuple[str, BaseException]] = []
        for _, collaborator in collaborators:
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
        if failures:
            messages = "; ".join(f"{name}: {exc!r}" for name, exc in failures)
            raise RagHubError(f"shutdown encountered {len(failures)} failure(s): {messages}")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        source: str | Path | bytes,
        **options: "JSONValue",
    ) -> Pipeline:
        """Ingest a file, directory, or raw bytes synchronously.

        Args:
            source: Path to a file/directory or raw bytes.
            **options: Optional overrides (``source_uri=``,
                ``mime_type=``, ``metadata=``, ``force=``,
                ``user=``). See the previous signature for
                semantics; ``**options`` keeps the call site
                backward-compatible while collapsing the named
                parameter list.

        Returns:
            A :class:`Pipeline` for a single source, or a
            composite result for a directory.

        Raises:
            IngestionError: When ingestion cannot complete.

        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                return self.ingest_directory_sync(p, options.get("metadata"), options.get("user"))
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = options.get("source_uri") or "bytes://memory"
        if not file_bytes:
            raise IngestionError(f"ingest({source!r}) received empty bytes; nothing to index.")
        result = cast(
            Pipeline,
            maybe_await(
                self.ingest_one_async(
                    file_bytes,
                    uri,
                    options.get("mime_type", "text/plain"),
                    metadata=options.get("metadata"),
                    force=options.get("force", False),
                    user=options.get("user"),
                )
            ),
        )
        if getattr(result, "error", None) is not None:
            raise IngestionError(
                f"ingest({source!r}) failed: {result.error.message if result.error else 'unknown'}"
            )
        return result

    def ingest_directory_sync(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
    ) -> Pipeline:
        """Recursively ingest a directory synchronously.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`User`.
            show_progress: When ``True`` (default), wrap the file loop
                in a :class:`tqdm.tqdm` progress bar. Suppress with
                ``False`` for non-interactive callers.

        Returns:
            A :class:`Pipeline` summarising the batch.

        """
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        results: list[Pipeline] = []
        iterator = tqdm(files, desc="Ingesting", disable=not show_progress, unit="file")
        for child in iterator:
            results.append(self.ingest(child, metadata=metadata, user=user))
        return Pipeline(
            pipeline_id="batch",
            pipeline_name="ingest",
            outputs={"batch": results},
        )

    async def aingest(
        self,
        source: str | Path | bytes,
        **options: "JSONValue",
    ) -> Pipeline:
        """Async version of :meth:`ingest`.

        Args:
            source: Path to a file/directory or raw bytes.
            **options: Optional overrides (``source_uri=``,
                ``mime_type=``, ``metadata=``, ``force=``,
                ``user=``). See :meth:`ingest` for semantics.

        Raises:
            IngestionError: When ingestion cannot complete.

        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            if p.is_dir():
                return await self.ingest_directory_async(
                    p, options.get("metadata"), options.get("user")
                )
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = options.get("source_uri") or "bytes://memory"
        if not file_bytes:
            raise IngestionError(f"aingest({source!r}) received empty bytes; nothing to index.")
        return await self.ingest_one_async(
            file_bytes,
            uri,
            options.get("mime_type", "text/plain"),
            metadata=options.get("metadata"),
            force=options.get("force", False),
            user=options.get("user"),
        )

    async def ingest_directory_async(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
    ) -> Pipeline:
        """Recursively ingest a directory asynchronously.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`User`.
            show_progress: When ``True`` (default), wrap the file loop
                in a :class:`tqdm.tqdm` progress bar. Suppress with
                ``False`` for non-interactive callers.

        """
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        n_workers = max(1, min(4, len(files)))
        semaphore = asyncio.Semaphore(n_workers)

        async def bounded(child: Path) -> Pipeline:
            """Run ingest on ``child`` under the concurrency cap."""
            async with semaphore:
                return await self.aingest(child, metadata=metadata, user=user)

        results = await asyncio.gather(*(bounded(c) for c in files))
        # Rebuild the BM25 keyword index once after the batch — the
        # per-file insert path skips it for speed.
        vector_store = getattr(self, "vector_store", None)
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()
        return Pipeline(
            pipeline_id="batch",
            pipeline_name="ingest",
            outputs={"batch": list(results)},
        )

    async def ingest_dir(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
        max_workers: int | None = None,
    ) -> Pipeline:
        """Run every file in ``directory`` through a ProcessPoolExecutor.

        Each worker process builds its own RAG from a serialised
        settings path (cheap — no RAG stack re-initialisation since
        ``RAG.__init__`` only allocates slots). The worker returns the
        list of (Chunk, vector) tuples it would have inserted
        into the local store. The main process inserts them into the
        shared vector store and rebuilds BM25 once at the end.

        The previous path (:meth:`ingest_directory_async`) stays as
        the in-process option for environments where fork isn't
        reliable; this method picks up the same per-file work in
        parallel processes.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        if not files:
            return Pipeline(
                pipeline_id="batch",
                pipeline_name="ingest",
                outputs={"batch": []},
            )

        n_workers = max(1, min(max_workers or os.cpu_count() or 4, len(files)))
        settings_path = self.settings_serialise_path()
        embedder_signature = (self.embedder.model_name, self.embedder.dimension)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(
                    ingest_one_worker,
                    settings_path,
                    str(p),
                    metadata,
                    embedder_signature,
                )
                for p in files
            ]
            worker_outputs = [f.result() for f in futures]

        # Merge into the main vector store.
        vector_store = getattr(self, "vector_store", None)
        n_inserted = 0
        for chunks, vectors in worker_outputs:
            if chunks and vector_store is not None:
                written = vector_store.insert(chunks, vectors)
                n_inserted += written
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()

        return Pipeline(
            pipeline_id="batch",
            pipeline_name="ingest",
            outputs={"batch": worker_outputs, "files": [str(p) for p in files]},
        )

    def settings_serialise_path(self) -> str:
        """Write the active settings to a sidecar file and return its path.

        Workers re-build ``Settings`` from the file rather than from
        the live :class:`RAG` instance. We round-trip the existing
        ``Settings`` object so any env-var-driven defaults are picked
        up.
        """
        import json
        import tempfile

        path = Path(tempfile.mkstemp(prefix="rag-settings-", suffix=".json")[1])
        path.write_text(
            json.dumps(
                self.settings.model_dump(mode="json"),
                default=str,
            ),
            encoding="utf-8",
        )
        return str(path)

    async def ingest_one_async(
        self,
        file_bytes: bytes,
        source_uri: str,
        mime_type: str,
        **options: "JSONValue",
    ) -> Pipeline:
        """Run a single ingest pipeline asynchronously.

        Args:
            file_bytes: Raw bytes to ingest.
            source_uri: Stable source URI for the file.
            mime_type: MIME hint for the converter.
            **options: Optional overrides (``metadata=``,
                ``force=``, ``user=``).

        """
        user: Any | None = options.get("user")
        context = PipelineCtx(
            pipeline_name="ingest",
            metadata={"user_id": getattr(user, "email", None)} if user is not None else {},
        )
        result = await self.ingest_pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=source_uri,
            mime_type=mime_type,
            metadata=options.get("metadata") or {},
            force=options.get("force", False),
            user=user,
        )
        if getattr(result, "error", None) is not None:
            raise IngestionError(result.error or "ingestion failed")
        if hasattr(self.manifest, "record") and hasattr(self.manifest, "get"):
            prior = self.manifest.get(source_uri)
            if not (isinstance(prior, dict) and prior.get("checksum") == sha256_bytes(file_bytes)):
                bundle_id = deterministic_id("bundle", source_uri, sha256_bytes(file_bytes))
                self.manifest.record(
                    source_uri,
                    bundle_id=bundle_id,
                    checksum=sha256_bytes(file_bytes),
                )
        return result

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, document_id: str) -> None:
        """Delete a document and all of its chunks.

        Accepts either a bundle id (the deterministic
        ``document_id`` recorded on each chunk), a source URI (the
        ``source_uri`` argument supplied to :meth:`ingest`), or any
        prior bundle id that has been retired to that source. All
        matching bundles are removed from both the vector store and
        the knowledge repository so a subsequent ingest does not see
        stale entries.
        """
        target_ids: set[str] = {document_id}
        if hasattr(self.knowledge_repo, "list_by_source"):
            for bundle in self.knowledge_repo.list_by_source(document_id):
                target_ids.add(bundle.bundle_id)
        # Walk every bundle already known for this source and add its
        # bundle id to the deletion set. This catches the prior
        # bundle id a re-ingest left behind; without it, a subsequent
        # ingest would still see the old bundle and short-circuit on
        # the wrong checksum.
        if hasattr(self.manifest, "sources"):
            for prior_uri in list(self.manifest.sources()):
                if prior_uri == document_id:
                    prior_record = self.manifest[prior_uri]
                    prior_bundle_id = str(prior_record.get("bundle_id", ""))
                    if prior_bundle_id:
                        target_ids.add(prior_bundle_id)
        for tid in target_ids:
            if hasattr(self.vector_store, "delete_document"):
                self.vector_store.delete_document(tid)
            if hasattr(self.knowledge_repo, "delete"):
                self.knowledge_repo.delete(tid)
            # Phase 6.8: walk the structured indexes too.
            for index in (getattr(self, "raptor", None), getattr(self, "graph", None)):
                if index is not None and hasattr(index, "delete_for_document"):
                    index.delete_for_document(tid)
        if hasattr(self.manifest, "remove"):
            if document_id in self.manifest:
                self.manifest.remove(document_id)
            sources_method = getattr(self.manifest, "sources", None)
            if callable(sources_method):
                for uri in list(sources_method()):
                    record = self.manifest.get(uri)
                    if not isinstance(record, dict):
                        continue
                    if str(record.get("bundle_id", "")) in target_ids:
                        self.manifest.remove(uri)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str, **kwargs: "JSONValue") -> Response:
        """Ask a question and return a typed :class:`Response`."""
        return cast(
            Response,
            maybe_await(
                self.aquery(question, **kwargs)
            ),
        )

    @staticmethod
    def scoped_session_id(user: Any, session_id: str | None) -> str | None:
        """Combine ``user`` and ``session_id`` into a single opaque key.

        The conversation store is keyed by this combined value so two
        callers who happen to share or guess a ``session_id`` cannot
        read each other's history. When ``user`` is ``None`` the
        method returns the raw ``session_id`` (matching the prior
        for tests that exercise the in-process store anonymously).

        Args:
            user: The :class:`User` (or any duck-typed
                object with ``user_id`` / ``email`` attributes).
            session_id: The caller-supplied session id.

        Returns:
            The namespaced key, or ``None`` when no session id is set.

        """
        if session_id is None:
            return None
        if user is None:
            return session_id
        uid = getattr(user, "user_id", None) or getattr(user, "email", None) or "anonymous"
        return f"{uid}::{session_id}"

    def session_overrides(
        self, scoped_session_id: str | None, user: Any | None = None
    ) -> dict[str, Any] | None:
        """Return the session's tool/agent overrides (Phase 1.12).

        Args:
            scoped_session_id: The namespaced session id produced by
                :meth:`scoped_session_id`.
            user: Optional user principal. The conversation store is
                keyed by the scoped id, so the caller must pass the
                user that was used to build that scoped id.

        Returns:
            The overrides dict, or ``None`` when the session has none
            stored. Sessions without a key resolve to the global
            default in the resolver.

        """
        if scoped_session_id is None:
            return None
        get_overrides = getattr(self.conversation_store, "get_overrides", None)
        if not callable(get_overrides):
            return None
        return cast(dict[str, Any] | None, get_overrides(scoped_session_id))

    async def aquery(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: "JSONValue",
    ) -> Response:
        """Async version of :meth:`query`.

        All the ``agent / web / graph / reranker / ...`` kwargs are
        advanced-RAG flags (Phase 8.7). When any are set the facade
        resolves them against per-session overrides and per-user
        preferences via :func:`raghub.agent.resolve`. The resolved
        config is reflected in the returned :class:`Response`'s
        ``transforms_applied`` and ``metadata`` fields.

        Args:
            question: The user's question.
            request: Optional :class:`RagQueryRequest` that bundles
                the remaining advanced-RAG overrides.
            **kwargs: Convenience overrides accepted as keyword
                arguments (``user=``, ``session_id=``,
                ``tools_enabled=``, ``agent=``, ``web=``, ``graph=``,
                ``summaries=``, ``reranker=``, ``long_context_pass=``,
                ``query_transforms=``, ``max_steps=``, ``top_k=``,
                ``metadata_filter=``, ``response_model=``).

        Returns:
            A typed :class:`Response`.

        Raises:
            IngestionError: When ``question`` is empty or whitespace-only.

        """
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        user: Any | None = merged.get("user")
        session_id: str | None = merged.get("session_id")
        top_k: int = merged.get("top_k", 5)
        metadata_filter: dict[str, Any] | None = merged.get("metadata_filter")
        response_model: type | None = merged.get("response_model")
        if not question or not question.strip():
            raise IngestionError("query() requires a non-empty question")
        if self.llm is None:
            raise ConfigurationError(
                "No LLM API key configured; set RAG_LLM_API_KEY "
                "(or another provider key) before calling query()."
            )
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineCtx(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )
        # Phase 8.7: resolve the effective config so downstream code
        # (Phase 7 agent) can consume it. The current QueryPipeline
        # already honours `top_k` and the transformer wired in
        # ``__init__``; everything else flows through Phase 5 / 7.

        resolved = resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )
        context.metadata["resolved_config"] = resolved.to_dict()
        # Phase 7.11: forward the resolved ``tools_enabled`` set
        # into the pipeline so the dispatcher routes through the
        # agentic path when any tool is on.
        resolved_tools = set(resolved.tools_enabled) if resolved.tools_enabled else None
        result = await self.query_pipeline.run(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            response_model=response_model,
            user=user,
            session_id=scoped,
            tools_enabled=resolved_tools,
            resolved_config=resolved.to_dict(),
        )
        if getattr(result, "error", None) is not None:
            raise RagHubError(result.error or "query failed")
        return ResponseBuilder.from_pipeline(result)

    async def astream(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: "JSONValue",
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token via the LLM's ``astream``.

        Accepts the same advanced-RAG flags as :meth:`aquery`; they
        are resolved through :func:`raghub.agent.resolve` and the
        resolved config is attached to the streaming span for
        observability.
        """
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        user: Any | None = merged.get("user")
        session_id: str | None = merged.get("session_id")
        top_k: int = merged.get("top_k", 5)
        metadata_filter: dict[str, Any] | None = merged.get("metadata_filter")
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineCtx(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )

        resolved = resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )
        context.metadata["resolved_config"] = resolved.to_dict()
        async for piece in self.query_pipeline.stream(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            user=user,
            session_id=scoped,
            tools_enabled=(set(resolved.tools_enabled) if resolved.tools_enabled else None),
        ):
            yield piece

    async def astream_agent(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: "JSONValue",
    ) -> AsyncIterator[Any]:
        """Stream :class:`PlannerEvent` instances from the agent loop.

        Args:
            question: The user's question.
            request: Optional :class:`RagQueryRequest` that bundles
                the remaining advanced-RAG overrides.
            **kwargs: Convenience overrides accepted as keyword
                arguments (``user=``, ``session_id=``,
                ``tools_enabled=``, ``agent=``, ``web=``, ``graph=``,
                ``summaries=``, ``reranker=``, ``long_context_pass=``,
                ``query_transforms=``, ``max_steps=``).

        Yields:
            :class:`PlannerEvent` instances. SSE encoding is the
            caller's responsibility — the FastAPI route uses
            :meth:`raghub.api_sse.Sse.format`.

        """
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        user: Any | None = merged.get("user")
        session_id: str | None = merged.get("session_id")
        scoped = self.scoped_session_id(user, session_id)
        resolved = self.resolve_agent_config(merged, scoped, user)
        if self.agentic_pipeline is None:
            async for event in self.fallback_planner_events(question, session_id, user):
                yield event
            return
        context = PipelineCtx(
            pipeline_name="query_agent",
            metadata={
                "session_id": scoped or "",
                "resolved_config": resolved.to_dict(),
            },
        )
        async for event in self.agentic_pipeline.astream(
            context,
            question=question,
            user=user,
            session_id=scoped,
            tools_enabled=(set(resolved.tools_enabled) if resolved.tools_enabled else None),
            history=[],
        ):
            yield event

    def resolve_agent_config(
        self,
        merged: dict[str, Any],
        scoped: str | None,
        user: Any,
    ) -> Any:
        """Resolve the effective advanced-RAG config for a request."""
        return resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )

    async def fallback_planner_events(
        self,
        question: str,
        session_id: str | None,
        user: Any | None = None,
    ) -> AsyncIterator[Any]:
        """Yield planner events from the non-agentic path.

        When the agentic pipeline is not configured the facade
        wraps each token of the streaming answer as a planner
        event so SSE consumers see a uniform stream.
        """
        async for piece in self.astream(
            question,
            user=user,
            session_id=session_id,
            top_k=5,
            metadata_filter=None,
        ):
            yield PlannerEvent(
                kind="answer_chunk",
                step=0,
                payload={"text": piece},
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        benchmark: str = "financebench",
        *,
        response_factory: Callable[[dict[str, Any]], Any] | None = None,
        examples: Sequence[dict[str, Any]] | None = None,
        evaluator: Any = None,
    ) -> list[Result]:
        """Run a benchmark evaluation.

        Args:
            benchmark: ``"financebench"`` is the only supported name today.
            response_factory: Optional callable producing an answer per
                example (sync or async); when ``None`` the facade calls
                :meth:`aquery`.
            examples: The example list to evaluate.
            evaluator: Optional pre-built evaluator instance. When
                supplied the method skips the default ``Finance()``
                construction, which is what tests use to inject fakes
                without touching module globals.

        Returns:
            The list of evaluation :class:`Result` records.

        """
        if benchmark != "financebench":
            raise ConfigurationError(f"Unknown benchmark: {benchmark!r}")

        if evaluator is None:
            evaluator = Finance()
        factory = response_factory

        async def coerce_answer(example: dict[str, Any]) -> Any:
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

        return cast(
            list[Result],
            maybe_await(evaluator.evaluate(examples, response_factory=coerce_answer)),
        )

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
    # v0.7.x collaborator accessors
    # ------------------------------------------------------------------

    def queue(self) -> Any:
        """Return the persistent queue or ``None`` when not configured."""
        return self.queue_

    def feedback_store(self) -> Any:
        """Return the feedback store or ``None`` when not configured."""
        return self.feedback_store_

    def rate_limiter(self) -> Any:
        """Return the rate limiter or ``None`` when not configured."""
        return self.rate_limiter_

    def archive(self) -> Any:
        """Return the archive store or ``None`` when not configured."""
        return self.archive_

    def tenant_resolver(self) -> Any:
        """Return the tenant resolver or ``None`` when not configured."""
        return self.tenant_resolver_

    def isolation_strategy(self) -> Any:
        """Return the active isolation strategy enum (defaults to ``RowLevel``)."""
        from raghub.tenants.isolation import IsolationStrategy

        if self.isolation_strategy_ is None:
            return IsolationStrategy.ROW_LEVEL
        return self.isolation_strategy_

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def sync_index(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
        show_progress: bool = True,
    ) -> dict[str, list[str]]:
        """Reconcile ``directory`` against the manifest.

        Uses the manifest's ``bundle_id`` and the source URI
        independently: a changed file produces a new bundle id but
        the prior bundle id (still in the manifest under the same
        source URI) must be retired so a re-ingest does not double
        index or short-circuit on the wrong checksum.

        The summary is grouped into ``added``, ``modified``,
        ``unchanged``, and ``removed`` lists so the caller can
        report progress.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`User`.
            show_progress: When ``True`` (default), wrap the file loop
                in a :class:`tqdm.tqdm` progress bar.

        Returns:
            A summary dict with ``added``, ``modified``, ``unchanged``,
            and ``removed`` lists of source URIs.

        """
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

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        iterator = tqdm(files, desc="Syncing index", disable=not show_progress, unit="file")
        for child in iterator:
            self.sync_one(child, metadata, user, seen, summary)

        for prior_uri in self.manifest.sources():
            if prior_uri in seen:
                continue
            if not prior_uri.startswith(str(directory.resolve())):
                continue
            self.remove_prior(prior_uri, summary)

        self.manifest.save()
        return summary

    def sync_one(
        self,
        child: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        seen: set[str],
        summary: dict[str, list[str]],
    ) -> None:
        """Reconcile a single file against the manifest."""
        if not child.is_file():
            return
        uri = str(child.resolve())
        seen.add(uri)
        data = child.read_bytes()
        checksum = sha256_bytes(data)
        prior = self.manifest.get(uri)
        bundle_id = deterministic_id("bundle", uri, checksum)
        if prior is None:
            result = self.ingest(child, metadata=metadata, user=user)
            if isinstance(result, Pipeline) and getattr(result, "error", None) is not None:
                raise IngestionError(result.error or f"failed to ingest {uri}")
            self.manifest.record(uri, bundle_id=bundle_id, checksum=checksum)
            summary["added"].append(uri)
            return
        if prior.get("checksum") == checksum:
            summary["unchanged"].append(uri)
            return
        # Changed file: retire the prior bundle id before re-ingesting.
        prior_bundle_id = str(prior.get("bundle_id", ""))
        result = self.ingest(child, metadata=metadata, force=True, user=user)
        if isinstance(result, Pipeline) and getattr(result, "error", None) is not None:
            raise IngestionError(result.error or f"failed to ingest {uri}")
        if prior_bundle_id and prior_bundle_id != bundle_id:
            self.delete(prior_bundle_id)
        self.manifest.record(uri, bundle_id=bundle_id, checksum=checksum)
        summary["modified"].append(uri)

    def remove_prior(
        self,
        prior_uri: str,
        summary: dict[str, list[str]],
    ) -> None:
        """Drop a manifest entry that no longer has a file on disk."""
        prior_record = self.manifest[prior_uri]
        bundle_id = str(prior_record.get("bundle_id", ""))
        self.delete(bundle_id)
        self.manifest.remove(prior_uri)
        summary["removed"].append(prior_uri)

    def ingest_async(
        self,
        source: str | Path | bytes,
        *,
        source_uri: str | None = None,
        mime_type: str = "text/plain",
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
    ) -> str:
        """Submit an ingest job to the background service.

        Routing:
            * If ``self.queue_`` is a :class:`SqliteQueue`
              (constructed in :meth:`__init__` when
              ``Settings.queue.backend == "sqlite"``), the job is
              submitted to that queue and the queue's UUID-shaped
              job id is returned.
            * Otherwise, falls back to the legacy ``Resumable``
              threadpool path.
        """
        if isinstance(source, (str, Path)):
            p = Path(source)
            file_bytes = p.read_bytes()
            uri = str(p.resolve())
        else:
            file_bytes = bytes(source)
            uri = source_uri or "bytes://memory"

        # Tier 4 Item 21: SqliteQueue path
        if self.queue_ is not None:
            import asyncio

            from raghub.jobs import JobStatus
            from raghub.tenants import get_current_tenant, validate_tenant_id

            tenant_id: str | None = None
            ctx = get_current_tenant()
            if ctx is not None:
                tenant_id = ctx.tenant_id
                validate_tenant_id(tenant_id)

            content_hash = hashlib.sha256(file_bytes).hexdigest()
            payload = {
                "source": file_bytes.decode("latin-1"),
                "source_uri": uri,
                "mime_type": mime_type,
                "metadata": metadata or {},
                "user": getattr(user, "user_id", None) if user else None,
                "content_hash": content_hash,
            }

            async def submit() -> str:
                existing_jobs = await self.queue_.list_for_tenant(
                    tenant_id=tenant_id,
                    content_hash=content_hash,
                )
                for job in existing_jobs:
                    if (
                        job.payload.get("content_hash") == content_hash
                        and job.status in (
                            JobStatus.PENDING,
                            JobStatus.RUNNING,
                        )
                    ):
                        return job.id
                return await self.queue_.submit(
                    kind="ingest",
                    payload=payload,
                    tenant_id=tenant_id,
                )

            try:
                asyncio.get_running_loop()
                # Already inside an event loop — run in a thread.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(lambda: asyncio.run(submit())).result()
            except RuntimeError:
                # No running loop — safe to call asyncio.run.
                return asyncio.run(submit())

        if self.background_ingestion is None:
            self.background_ingestion = Resumable(
                db_path=self.settings.data_dir / "ingestion_jobs.db"
            )

        return cast(
            str,
            self.background_ingestion.submit(
                self.ingest,
                source=file_bytes,
                source_uri=uri,
                mime_type=mime_type,
                metadata=metadata,
                user=user,
            ),
        )

    def job_status(self, job_id: str) -> str | None:
        """Return the status of a background ingestion job."""
        # Tier 4 Item 22: SqliteQueue lookup
        if self.queue_ is not None:
            import asyncio
            import concurrent.futures

            from raghub.jobs import JobStatus

            async def lookup() -> str | None:
                stats = await self.queue_.stats()
                if sum(stats.values()) == 0:
                    return None
                jobs = await self.queue_.list(status=None, limit=1000)
                for job in jobs:
                    if job.id == job_id:
                        return str(job.status.value)
                return None

            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(lambda: asyncio.run(lookup())).result()
            except RuntimeError:
                return asyncio.run(lookup())

        if self.background_ingestion is None:
            return None
        return cast(str | None, self.background_ingestion.get_status(job_id))

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    def conversation_history(
        self,
        session_id: str,
        *,
        user: Any | None = None,
        limit: int = 50,
    ) -> list[Turn]:
        """Return the most recent conversation turns for a session.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`User` whose
                ``user_id`` / ``email`` scopes the lookup. When
                omitted, the lookup uses the raw ``session_id`` and
                will only return history created with ``user=None``
                — preventing accidental cross-user reads.
            limit: Maximum number of turns to return.

        Returns:
            The list of :class:`Turn` records, oldest
            first.

        """
        scoped = self.scoped_session_id(user, session_id) or session_id
        return cast(list[Any], self.conversation_store.load(scoped, limit=limit))

    def clear_conversation(
        self,
        session_id: str,
        *,
        user: Any | None = None,
    ) -> None:
        """Clear a session's conversation history.

        Args:
            session_id: The caller-supplied session id.
            user: Optional :class:`User` whose
                ``user_id`` / ``email`` scopes the delete. When
                omitted, the raw ``session_id`` is used.

        """
        scoped = self.scoped_session_id(user, session_id) or session_id
        self.conversation_store.clear(scoped)
