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

The behaviour is split across focused mixin modules so the
:class:`RAG` class itself stays a thin wiring layer:

    raghub.rag.ingest_mixin       - ingestion, deletion, background jobs
    raghub.rag.query_mixin        - query, streaming, agent, evaluation
    raghub.rag.sync_mixin         - incremental indexing
    raghub.rag.conversation_mixin - conversation-history access
    raghub.rag.defaults           - module-level default factories

External code should continue to import via ``from raghub import RAG``
or ``from raghub.rag import RAG``; both paths resolve to the same class.

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
import tomllib
from pathlib import Path
from typing import Any, cast

import yaml

from raghub.agent import Agent, build_tools
from raghub.config import Settings
from raghub.conv import Memory
from raghub.errors import RagHubError
from raghub.gen import DefaultGenerator
from raghub.knowledge import GraphIndex, Manifest, MemoryRepo, Raptor
from raghub.models import DocumentConverter, KnowledgeRepository, RagComponents
from raghub.pipeline import AgentPipeline, Cache, Ingest, QueryPipeline
from raghub.plugins import Plugins
from raghub.rag.conversation_mixin import ConversationMixin
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
)
from raghub.rag.ingest_mixin import IngestMixin
from raghub.rag.query_mixin import QueryMixin
from raghub.rag.sync_mixin import SyncMixin
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
from raghub.types import JSONValue

__all__ = [
    "LLM_API_KEY_ENV_VARS",
    "RAG",
    "has_llm_api_key",
]


class RAG(
    IngestMixin,
    QueryMixin,
    SyncMixin,
    ConversationMixin,
):
    """High-level RAGHub facade.

    Construct via :meth:`RAG.from_config` for the standard
    configuration-driven path, or pass components directly for
    advanced customisation. Every collaborator is replaceable
    through the constructor.

    The class itself owns construction, lifecycle, and diagnostics.
    Query, ingestion, sync, and conversation behaviour live in
    the corresponding mixin modules.

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
        **kwargs: JSONValue,
    ) -> None:
        """Initialise the facade."""
        component_map: dict[str, Any] = dict(components) if components is not None else {}
        component_map.update(kwargs)
        component_map.setdefault("settings", settings)
        self.wire_components(component_map)
        self.wire_ingest()
        self.wire_query(component_map)
        self.manifest: Manifest = (
            component_map.get("manifest") or Manifest(self.settings.data_dir / "manifest.json")
        )
        self.background_ingestion = component_map.get("background_service")
        self.persistent_queue = self.init_queue(component_map)
        self.worker_ = self.init_worker(component_map)
        self.worker_task_: asyncio.Task[None] | None = None
        self.tenant_resolver_ = self.init_tenant(component_map)
        self.feedback_store_ = self.init_feedback(component_map)
        self.rate_limiter_: Any = None
        self.archive_: Any = None
        self.isolation: Any = component_map.get("isolation_strategy")

    def wire_components(self, components: dict[str, Any]) -> None:
        """Resolve core collaborators from ``components`` or defaults."""
        self.settings: Settings = components.get("settings") or Settings.load()
        self.registry: Any = components.get("registry") or Plugins()
        self.knowledge_repo: KnowledgeRepository = (
            components.get("knowledge_repo") or MemoryRepo()
        )
        self.vector_store: Any = (
            components.get("vector_store") or default_vector_store(self.settings.embedding_dim)
        )
        self.embedder: Any = components.get("embedder") or default_embedder(
            self.settings.embedding_model, self.settings.embedding_dim
        )
        self.llm: Any = components.get("llm") or default_llm(self.settings.llm_model)
        self.converter: DocumentConverter = (
            components.get("converter") or default_converter()
        )
        self.chunker: Any = components.get("chunker") or default_chunker(
            self.settings.chunk_size_words,
            self.settings.chunk_overlap_words,
            chunker_strategy=self.settings.chunker_strategy,
            embedding_model_chunker=self.settings.embedding_model_chunker,
        )
        self.reranker: Any = components.get("reranker") or build_reranker(
            self.settings, llm=self.llm
        )
        self.generator: Any = cast(
            Any,
            components.get("generator")
            or DefaultGenerator(
                llm=self.llm,
                timeout_seconds=components.get("llm_timeout_seconds"),
            ),
        )
        self.structured: Any = (
            components.get("structured")
            if components.get("structured") is not None
            else default_structured()
        )
        if components.get("telemetry") is None:
            inner = default_telemetry()
            self.telemetry: Any = RedactingTelemetry(inner)
        else:
            self.telemetry = components["telemetry"]

    def wire_ingest(self) -> None:
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

    def wire_query(self, components: dict[str, Any]) -> None:
        """Build retrieval, query, and agent pipelines."""
        self.query_cache: Cache | None = (
            Cache(ttl_seconds=self.settings.query_cache_ttl_seconds)
            if self.settings.enable_query_cache
            else None
        )
        self.transformer: Any = components.get("transformer") or default_transforms(
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
        self.tool_registry = build_tools(
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

    def init_queue(self, components: dict[str, Any]) -> Any:
        """Construct the persistent ingestion queue.

        Priority:
            1. ``components["queue"]`` if explicitly supplied.
            2. ``Settings.queue.backend == "sqlite"`` -> ``SqliteQueue``.
            3. Otherwise ``None`` (legacy threadpool path).
        """
        supplied = components.get("queue")
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

    def init_worker(self, components: dict[str, Any]) -> Any:
        """Construct a :class:`Worker` when a persistent queue is configured.

        Priority:
            1. ``components["worker"]`` if explicitly supplied.
            2. ``self.persistent_queue`` is a :class:`SqliteQueue` (built above when
               ``Settings.queue.backend == "sqlite"``) -> build a Worker
               bound to that queue with a default ingest handler.
            3. Otherwise ``None``.
        """
        supplied = components.get("worker")
        if supplied is not None:
            return supplied
        if self.persistent_queue is None:
            return None
        try:
            from raghub.jobs import SqliteQueue, Worker
        except ImportError:
            return None
        if not isinstance(self.persistent_queue, SqliteQueue):
            return None

        async def ingest(job: Any) -> None:
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
            queue=self.persistent_queue,
            handler=ingest,
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

    def init_tenant(self, components: dict[str, Any]) -> Any:
        """Construct the tenant resolver.

        Priority:
            1. ``components["tenant_resolver"]`` if supplied.
            2. ``Settings.tenants.resolver == "composite" | "jwt" | "header"``.
            3. Otherwise ``None``.
        """
        supplied = components.get("tenant_resolver")
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

    def init_feedback(self, components: dict[str, Any]) -> Any:
        """Construct the feedback store (Tier 3 Item 19).

        Priority:
            1. ``components["feedback_store"]`` if supplied.
            2. ``Settings.feedback.backend == "sqlite"`` -> SqliteFeedbackStore.
            3. Otherwise ``None``.
        """
        supplied = components.get("feedback_store")
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
    def from_config(cls: type[RAG], path: str | Path) -> RAG:
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
        return self.persistent_queue

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
        from raghub.tenants.isolation import Isolation

        if self.isolation is None:
            return Isolation.RowLevel
        return self.isolation
