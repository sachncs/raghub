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
caller's ``UserPrincipal``. The facade namespaces keys internally
so that two callers who happen to share or guess a ``session_id``
cannot read each other's history. :meth:`conversation_history`
and :meth:`clear_conversation` both accept a ``user`` argument;
the public surface mirrors the rest of the RBAC contract.
"""

from __future__ import annotations

import asyncio
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
from raghub.config import Settings
from raghub.conversation import InMemoryConversationStore
from raghub.embeddings import BaseEmbeddingProvider, HashingEmbeddingProvider
from raghub.exceptions import ConfigurationError, IngestionError, RagHubError
from raghub.generation import DefaultGenerator
from raghub.helper.evaluation import FinanceBench
from raghub.helper.response import ResponseBuilder
from raghub.helper.retrieval import (
    Colbert as ColbertLateInteraction,
)
from raghub.helper.retrieval import (
    Compose as ComposeTransformer,
)
from raghub.helper.retrieval import (
    Context as LongContextRerankPass,
)
from raghub.helper.retrieval import (
    Decompose as DecomposeTransformer,
)
from raghub.helper.retrieval import (
    Hyde as HydeTransformer,
)
from raghub.helper.retrieval import (
    MultiQuery as MultiQueryTransformer,
)
from raghub.helper.retrieval import (
    Retrieval as RetrievalPipeline,
)
from raghub.helper.retrieval import (
    StepBack as StepBackTransformer,
)
from raghub.helper.retrieval import (
    Transformer as QueryTransformer,
)
from raghub.helper.retrieval import (
    build_reranker,
)
from raghub.ingestion import ResumableBackgroundIngestionService, build_chonkie_chunker
from raghub.knowledge import (
    GraphRagIndex,
    InMemoryKnowledgeRepository,
    RaptorIndex,
    SourceManifest,
    sha256_bytes,
)
from raghub.models import (
    Chunker,
    ConversationTurn,
    DocumentConverter,
    EvaluationResult,
    PipelineContext,
    PipelineResult,
    Response,
    deterministic_id,
)
from raghub.observability import DEFAULT_METRICS_REGISTRY, PrometheusMetrics, RedactingTelemetry
from raghub.pipeline import AgenticQueryPipeline, IngestPipeline, QueryCache, QueryPipeline
from raghub.plugins import PluginRegistry
from raghub.utils import maybe_await_sync as maybe_await
from raghub.vectorstore import InMemoryVectorStore

T = TypeVar("T", bound=BaseModel)

"""Default factories for the RAG facade's optional dependencies.

Each ``default_*`` method is a thin wrapper that picks the best
available implementation based on what's installed and which
environment variables are set. The public :class:`raghub.RAG`
delegates to these so the class body itself stays small.

All optional dependencies (``Marker``, ``LiteLLM``, ``Qdrant``,
``Instructor``, ``Langfuse``, ``Chonkie``) are imported at module
top. The factories rely on the SDK constructors to raise
:class:`ConfigurationError` when their respective backends are
unusable.
"""

LLM_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NVIDIA_API_KEY",
    "GROQ_API_KEY",
    "LITELLM_API_KEY",
)


def has_llm_api_key() -> bool:
    """Return ``True`` when any provider API key env var is set."""
    return any(os.getenv(k) for k in LLM_API_KEY_ENV_VARS)


def default_converter() -> DocumentConverter:
    """Return the default document converter.

    Since ``marker-pdf`` is now a required runtime dependency, this
    always returns :class:`MarkerConverter`. Tests patch the
    `raghub.documents.MarkerConverter` symbol via this re-import.
    """
    from raghub.documents import MarkerConverter as _MarkerConverter

    return _MarkerConverter()


def default_chunker(
    chunk_size: int,
    chunk_overlap: int,
    *,
    chunker_strategy: str = "recursive",
    embedding_model_chunker: str = "minishlab/potion-base-8M",
) -> Chunker:
    """Return the default chunker.

    Args:
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of overlapping words.
        chunker_strategy: Chunking strategy name.
        embedding_model_chunker: Embedding model for semantic/late chunkers.

    Returns:
        :class:`ChonkieChunker` when Chonkie is available;
        :class:`WordWindowChunker` otherwise.
    """
    return build_chonkie_chunker(
        chunker_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model_chunker,
    )


def default_embedder(embedding_model: str, embedding_dim: int) -> BaseEmbeddingProvider:
    """Return the default embedding provider.

    Args:
        embedding_model: The model name (e.g. ``"text-embedding-3-small"``).
        embedding_dim: Output vector dimensionality.

    Returns:
        :class:`LiteLLMEmbeddingProvider` when LiteLLM is
        installed and an API key is configured; otherwise
        :class:`HashingEmbeddingProvider` for offline operation.
    """
    if not has_llm_api_key():
        return HashingEmbeddingProvider(dimension=embedding_dim, model_name=embedding_model)
    from raghub.embeddings import LiteLLMEmbeddingProvider as _LiteLLMEmbeddingProvider

    return _LiteLLMEmbeddingProvider(model=embedding_model)


def default_llm(llm_model: str) -> Any:
    """Return the default LLM provider.

    Args:
        llm_model: The configured LLM model name.

    Returns:
        :class:`LiteLLMProvider` for the configured model when an API
        key is available; :class:`HeuristicProvider` (offline fallback)
        otherwise.
    """
    if not has_llm_api_key():
        from raghub.llm import HeuristicProvider

        return HeuristicProvider()
    from raghub.llm import LiteLLMProvider as _LiteLLMProvider

    return _LiteLLMProvider(model=llm_model)


def default_vector_store(embedding_dim: int) -> Any:
    """Construct the default vector store.

    Args:
        embedding_dim: Expected output dimensionality of the embedder.

    Returns:
        :class:`InMemoryVectorStore` for the in-process test/dev path.
        The full pipeline factory :func:`raghub.vectorstore.build_vector_store`
        is used by the rest of the framework and points at a SQLite-backed
        store (sqlite-vector when installed, NumPy fallback otherwise).
    """
    return InMemoryVectorStore(embedding_dim=embedding_dim)


def default_structured() -> Any:
    """Return the default structured-output provider.

    Returns:
        :class:`InstructorStructuredOutputProvider` when Instructor
        is installed; ``None`` otherwise.
    """
    if not has_llm_api_key():
        return None
    from raghub.generation import InstructorStructuredOutputProvider as _Instructor

    return _Instructor()


def default_telemetry() -> Any:
    """Return the default telemetry provider.

    Returns:
        :class:`LangfuseTelemetryProvider` when Langfuse is
        configured (env vars set); otherwise :class:`NoOpTelemetry`.
    """
    from raghub.observability import (
        LangfuseTelemetryProvider as _LangfuseTelemetryProvider,
    )
    from raghub.observability import (
        NoOpTelemetry as _NoOpTelemetry,
    )

    if not _LangfuseTelemetryProvider.is_configured():
        return _NoOpTelemetry()
    return _LangfuseTelemetryProvider()


def default_transforms(
    llm: Any,
    *,
    enabled: list[str] | None = None,
    hyde_n: int = 1,
    multi_query_n: int = 4,
) -> Any:
    """Build the configured :class:`ComposeTransformer`.

    Args:
        llm: Any object with ``async_generate`` — typically the same
            LLM the facade already holds.
        enabled: Ordered list of transform names. Empty / ``None``
            returns an empty :class:`ComposeTransformer` (zero-cost
            fast path).
        hyde_n: Number of hypothetical passages for ``hyde``.
        multi_query_n: Number of rephrasings for ``multi_query``.

    Returns:
        A :class:`raghub.helper.retrieval.Compose`.
        Unknown names are dropped silently.
    """
    enabled = enabled or []
    transformers: list[QueryTransformer] = []
    for name in enabled:
        if name == "hyde":
            transformers.append(HydeTransformer(llm, n=hyde_n))
        elif name == "multi_query":
            transformers.append(MultiQueryTransformer(llm, n=multi_query_n))
        elif name == "step_back":
            transformers.append(StepBackTransformer(llm))
        elif name == "decompose":
            transformers.append(DecomposeTransformer(llm))
    return ComposeTransformer(transformers)


def ingest_one_worker(
    settings_path: str,
    pdf_path: str,
    metadata: dict[str, Any] | None,
    embedder_signature: tuple[str, int],
) -> tuple[list[Any], list[list[float]]]:
    """Worker entry-point for :meth:`RAG.ingest_directory_concurrent`.

    Each subprocess reconstructs a minimal :class:`RAG` from the
    settings serialised at ``settings_path`` and re-ingests a single
    PDF. It returns the chunks and vectors it produced so the parent
    process can insert them into the shared vector store and skip the
    duplicated embed / insert work.

    Returns:
        ``(chunks, vectors)`` lists pulled from the worker's local
        vector store after ``ingest`` completes. The vectors match the
        ``embedding_dim`` of the embedder that produced them.
    """
    import json as _json
    from pathlib import Path as _P

    from raghub.config import Settings as _S

    settings_dict = _json.loads(_P(settings_path).read_text(encoding="utf-8"))
    settings = _S.model_validate(settings_dict)
    # Each worker re-uses the process-pool's environment for the LLM
    # creds. The vector store is local to the worker (an in-memory
    # list) — the parent process owns the merged, durable index.
    rag = RAG(settings=settings)
    rag.ingest(_P(pdf_path), metadata=metadata, user=None)
    # Pull the chunks + vectors back out of the worker's store.
    vector_store = rag.vector_store
    chunks: list[Any] = []
    vectors: list[list[float]] = []
    for attr in ("records",):
        records = getattr(vector_store, attr, None)
        if records is None:
            continue
        if isinstance(records, dict):
            records = records.values()
        for record in records:
            chunk = getattr(record, "chunk", None)
            vec = getattr(record, "vector", None)
            if chunk is None or vec is None:
                continue
            chunks.append(chunk)
            vectors.append(vec)
        break
    return chunks, vectors


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
        converter: Any = None,
        chunker: Any = None,
        embedder: Any = None,
        llm: Any = None,
        llm_timeout_seconds: float | None = None,
        vector_store: Any = None,
        generator: Any = None,
        reranker: Any = None,
        knowledge_repo: Any = None,
        structured: Any | None = None,
        telemetry: Any = None,
        registry: Any = None,
        background_service: Any = None,
        manifest: Any = None,
        transformer: Any = None,
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
                :class:`LiteLLMProvider` when an API key is available;
                :class:`HeuristicProvider` (offline fallback) otherwise.
            llm_timeout_seconds: Maximum completion time for the default generator.
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
            transformer: Optional pre-built query-transform composer.
                Defaults to :func:`raghub.api.defaults.default_transforms`
                built from ``settings.query_transforms.enabled`` and
                ``self.llm``. Pass an empty :class:`ComposeTransformer`
                to disable transforms explicitly.
        """
        self.settings = settings or Settings.load()
        self.registry = registry or PluginRegistry()

        self.knowledge_repo = knowledge_repo or InMemoryKnowledgeRepository()
        self.vector_store = vector_store or default_vector_store(self.settings.embedding_dim)
        self.embedder = embedder or default_embedder(
            self.settings.embedding_model, self.settings.embedding_dim
        )
        self.llm = llm or default_llm(self.settings.llm_model)
        self.converter = converter or default_converter()
        self.chunker = chunker or default_chunker(
            self.settings.chunk_size_words,
            self.settings.chunk_overlap_words,
            chunker_strategy=self.settings.chunker_strategy,
            embedding_model_chunker=self.settings.embedding_model_chunker,
        )
        self.reranker = reranker or build_reranker(self.settings, llm=self.llm)
        self.generator = cast(
            Any,
            generator or DefaultGenerator(llm=self.llm, timeout_seconds=llm_timeout_seconds),
        )
        self.structured = structured if structured is not None else default_structured()

        if telemetry is None:
            inner = default_telemetry()
            self.telemetry: Any = RedactingTelemetry(inner)
        else:
            self.telemetry = telemetry
        # Phase 4.8: register the Prometheus metrics instance so
        # rerankers (and future hot-path components) can record
        # observations without coupling to the telemetry provider.
        metrics = PrometheusMetrics()
        self.metrics = metrics
        DEFAULT_METRICS_REGISTRY.set(metrics)

        self.ingest_pipeline = IngestPipeline(
            converter=self.converter,
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            knowledge_repo=self.knowledge_repo,
            telemetry=self.telemetry,
            raptor=getattr(self, "raptor", None),
            graph=getattr(self, "graph", None),
        )
        self.conversation_store: Any = InMemoryConversationStore()

        self.query_cache: QueryCache | None = (
            QueryCache(ttl_seconds=self.settings.query_cache_ttl_seconds)
            if self.settings.enable_query_cache
            else None
        )
        self.transformer = transformer if transformer is not None else default_transforms(
            self.llm,
            enabled=list(self.settings.query_transforms.enabled),
            hyde_n=self.settings.query_transforms.hyde_n,
            multi_query_n=self.settings.query_transforms.multi_query_n,
        )
        # Phase 2.8: build a RetrievalPipeline so multi-variant
        # retrieval can delegate to ``retrieve_variants``. Identity
        # reranker is fine — the transformer adds variants but
        # doesn't replace reranking.

        self.colbert = ColbertLateInteraction(self.settings.hybrid)
        self.retrieval_pipeline = RetrievalPipeline(
            embedding_provider=self.embedder,
            vector_store=self.vector_store,
            rerank=self.reranker,
            hybrid=self.settings.hybrid,
        )
        # Phase 5.3: build the long-context pass when the config
        # says so. The pass is a no-op when the configured LLM is
        # not in the allowlist, so building it eagerly is cheap.

        self.long_context_pass = LongContextRerankPass(
            llm=self.llm, settings=self.settings.long_context_pass
        )
        # Phase 6.6: build the structured knowledge indexes when
        # the operator opted in. Both default to None so the
        # fast path stays byte-equivalent (Phase 10.6).
        self.raptor = None
        self.graph = None
        if self.settings.summary_search_enabled:

            self.raptor = RaptorIndex(
                llm=self.llm,
                embedder=self.embedder,
                depth=2,
            )
        if self.settings.graph_search_enabled:

            self.graph = GraphRagIndex(llm=self.llm, embedder=self.embedder)

        # Phase 7.8 + 7.11: build the agent + tool registry + the
        # agentic pipeline. The agent is wired only when the
        # settings say so; the early-exit path is preserved when
        # ``settings.agent.enabled`` is ``False`` AND no tool is
        # explicitly requested.

        self.tool_registry = build_tool_registry(
            self.settings,
            retrieval_pipeline=self.retrieval_pipeline,
            vector_store=self.vector_store,
            raptor=self.raptor,
            graph=self.graph,
        )
        self.agent: Any | None = None
        self.agentic_pipeline: Any | None = None
        if self.settings.agent.enabled or self.settings.web_search.enabled or (
            self.settings.summary_search_enabled and self.raptor is not None
        ) or (self.settings.graph_search_enabled and self.graph is not None):

            self.agent = Agent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                settings=self.settings.agent,
                telemetry=self.telemetry,
            )
            self.agentic_pipeline = AgenticQueryPipeline(
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

        self.manifest: SourceManifest = manifest or SourceManifest(
            self.settings.data_dir / "manifest.json"
        )
        self.background_ingestion = background_service

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

        settings = Settings(
            **{k: v for k, v in payload.items() if k in Settings.model_fields}
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
            IngestionError: When ingestion cannot complete.
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
            raise IngestionError(f"ingest({source!r}) received empty bytes; nothing to index.")
        return cast(
            PipelineResult,
            maybe_await(self.ingest_one_async(file_bytes, uri, mime_type, metadata, force, user)),
        )

    def ingest_directory_sync(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
    ) -> PipelineResult:
        """Recursively ingest a directory synchronously.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`UserPrincipal`.
            show_progress: When ``True`` (default), wrap the file loop
                in a :class:`tqdm.tqdm` progress bar. Suppress with
                ``False`` for non-interactive callers.

        Returns:
            A :class:`PipelineResult` summarising the batch.
        """
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        results: list[PipelineResult] = []
        iterator = tqdm(files, desc="Ingesting", disable=not show_progress, unit="file")
        for child in iterator:
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
            IngestionError: When ingestion cannot complete.
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
            raise IngestionError(f"aingest({source!r}) received empty bytes; nothing to index.")
        return await self.ingest_one_async(file_bytes, uri, mime_type, metadata, force, user)

    async def ingest_directory_async(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
    ) -> PipelineResult:
        """Recursively ingest a directory asynchronously.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`UserPrincipal`.
            show_progress: When ``True`` (default), wrap the file loop
                in a :class:`tqdm.tqdm` progress bar. Suppress with
                ``False`` for non-interactive callers.
        """

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        n_workers = max(1, min(4, len(files)))
        semaphore = asyncio.Semaphore(n_workers)

        async def bounded(child: Path) -> PipelineResult:
            async with semaphore:
                return await self.aingest(child, metadata=metadata, user=user)

        results = await asyncio.gather(*(bounded(c) for c in files))
        # Rebuild the BM25 keyword index once after the batch — the
        # per-file insert path skips it for speed.
        vector_store = getattr(self, "vector_store", None)
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()
        return PipelineResult(
            pipeline_id="batch",
            pipeline_name="ingest",
            success=all(r.success for r in results),
            outputs={"batch": list(results)},
        )

    async def ingest_directory_concurrent(
        self,
        directory: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        *,
        show_progress: bool = True,
        max_workers: int | None = None,
    ) -> PipelineResult:
        """Run every file in ``directory`` through a ProcessPoolExecutor.

        Each worker process builds its own RAG from a serialised
        settings path (cheap — no RAG stack re-initialisation since
        ``RAG.__init__`` only allocates slots). The worker returns the
        list of (ChunkRecord, vector) tuples it would have inserted
        into the local store. The main process inserts them into the
        shared vector store and rebuilds BM25 once at the end.

        The previous path (:meth:`ingest_directory_async`) stays as
        the in-process option for environments where fork isn't
        reliable; this method picks up the same per-file work in
        parallel processes.
        """
        import contextlib
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        if not files:
            return PipelineResult(
                pipeline_id="batch",
                pipeline_name="ingest",
                success=True,
                outputs={"batch": []},
            )

        n_workers = max(
            1, min(max_workers or os.cpu_count() or 4, len(files))
        )
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
            if chunks and getattr(vector_store, "insert", None):
                written = vector_store.insert(chunks, vectors)
                n_inserted += written
        rebuild = getattr(vector_store, "rebuild_index", None)
        if callable(rebuild):
            rebuild()

        return PipelineResult(
            pipeline_id="batch",
            pipeline_name="ingest",
            success=True,
            outputs={"batch": worker_outputs, "files": [str(p) for p in files]},
        )

    def settings_serialise_path(self) -> str:
        """Write the active settings to a sidecar file and return its path.

        Workers re-build ``Settings`` from the file rather than from
        the live :class:`RAG` instance. We round-trip the existing
        ``Settings`` object so any env-var-driven defaults are picked
        up.
        """
        import json as _json
        import tempfile

        path = Path(tempfile.mkstemp(prefix="rag-settings-", suffix=".json")[1])
        path.write_text(
            _json.dumps(
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
        metadata: dict[str, Any] | None,
        force: bool = False,
        user: Any | None = None,
    ) -> PipelineResult:
        """Run a single ingest pipeline asynchronously."""
        context = PipelineContext(
            pipeline_name="ingest",
            metadata={"user_id": getattr(user, "email", None)} if user is not None else {},
        )
        result = await self.ingest_pipeline.run(
            context,
            file_bytes=file_bytes,
            source_uri=source_uri,
            mime_type=mime_type,
            metadata=metadata or {},
            force=force,
            user=user,
        )
        if not result.success:
            raise IngestionError(result.error or "ingestion failed")
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
        return cast(
            Response,
            maybe_await(
                self.aquery(
                    question,
                    user=user,
                    session_id=session_id,
                    top_k=top_k,
                    metadata_filter=metadata_filter,
                    response_model=response_model,
                )
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
        user: Any | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        response_model: type | None = None,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
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
            user: Optional :class:`UserPrincipal` for RBAC and per-user
                tool defaults.
            session_id: Optional session id; conversation history is
                loaded from the conversation store.
            top_k: Override of the default retrieval depth.
            metadata_filter: Optional metadata filter applied on top
                of the RBAC filter.
            response_model: Optional Pydantic model for structured
                output.
            tools_enabled: Tool allow-list override. ``None`` defers
                to session/user/global defaults.
            agent: When ``True``, force the agent loop on (Phase 7).
            web: Shortcut for ``"web_search" in tools_enabled``.
            graph: Shortcut for ``"graph_search" in tools_enabled``.
            summaries: Shortcut for ``"summary_search" in tools_enabled``.
            reranker: Per-request reranker override.
            long_context_pass: Per-request toggle for the second-pass
                long-context rerank.
            query_transforms: Per-request list of transform names.
            max_steps: Per-request cap on planner steps.

        Returns:
            A typed :class:`Response`.
        """
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineContext(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )
        # Phase 8.7: resolve the effective config so downstream code
        # (Phase 7 agent) can consume it. The current QueryPipeline
        # already honours `top_k` and the transformer wired in
        # ``__init__``; everything else flows through Phase 5 / 7.

        resolved = resolve(
            request_overrides={
                "tools_enabled": tools_enabled,
                "agent": agent,
                "web": web,
                "graph": graph,
                "summaries": summaries,
                "reranker": reranker,
                "long_context_pass": long_context_pass,
                "query_transforms": query_transforms,
                "max_steps": max_steps,
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )
        context.metadata["resolved_config"] = resolved.to_dict()
        # Phase 7.11: forward the resolved ``tools_enabled`` set
        # into the pipeline so the dispatcher routes through the
        # agentic path when any tool is on.
        resolved_tools = (
            set(resolved.tools_enabled) if resolved.tools_enabled else None
        )
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
        if not result.success:
            raise RagHubError(result.error or "query failed")
        return ResponseBuilder.from_pipeline(result)

    async def astream(
        self,
        question: str,
        *,
        user: Any | None = None,
        session_id: str | None = None,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token via the LLM's ``astream``.

        Accepts the same advanced-RAG flags as :meth:`aquery`; they
        are resolved through :func:`raghub.agent.resolve` and the
        resolved config is attached to the streaming span for
        observability.
        """
        scoped = self.scoped_session_id(user, session_id)
        context = PipelineContext(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )

        resolved = resolve(
            request_overrides={
                "tools_enabled": tools_enabled,
                "agent": agent,
                "web": web,
                "graph": graph,
                "summaries": summaries,
                "reranker": reranker,
                "long_context_pass": long_context_pass,
                "query_transforms": query_transforms,
                "max_steps": max_steps,
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
            tools_enabled=(
                set(resolved.tools_enabled) if resolved.tools_enabled else None
            ),
        ):
            yield piece

    async def astream_agent(
        self,
        question: str,
        *,
        user: Any | None = None,
        session_id: str | None = None,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
    ) -> AsyncIterator[Any]:
        """Stream :class:`PlannerEvent` instances from the agent loop.

        Args:
            question: The user's question.
            user: Optional :class:`UserPrincipal` for RBAC.
            session_id: Optional session id.
            tools_enabled, agent, web, graph, summaries, reranker,
            long_context_pass, query_transforms, max_steps: Same
                semantics as :meth:`aquery`.

        Yields:
            :class:`PlannerEvent` instances. SSE encoding is the
            caller's responsibility — the FastAPI route uses
            :meth:`raghub.helper.sse.Sse.format`.
        """

        scoped = self.scoped_session_id(user, session_id)
        resolved = resolve(
            request_overrides={
                "tools_enabled": tools_enabled,
                "agent": agent,
                "web": web,
                "graph": graph,
                "summaries": summaries,
                "reranker": reranker,
                "long_context_pass": long_context_pass,
                "query_transforms": query_transforms,
                "max_steps": max_steps,
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )
        if self.agentic_pipeline is None:
            # No agent configured — wrap planner tokens as events.
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
            return
        context = PipelineContext(
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
            tools_enabled=(
                set(resolved.tools_enabled) if resolved.tools_enabled else None
            ),
            history=[],
        ):
            yield event

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        benchmark: str = "financebench",
        *,
        response_factory: Callable[[dict[str, Any]], Any] | None = None,
        examples: Sequence[dict[str, Any]] | None = None,
    ) -> list[EvaluationResult]:
        """Run a benchmark evaluation."""
        if benchmark != "financebench":
            raise ConfigurationError(f"Unknown benchmark: {benchmark!r}")

        evaluator = FinanceBench()
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
            list[EvaluationResult],
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
            user: Optional :class:`UserPrincipal`.
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
            if not child.is_file():
                continue
            uri = str(child.resolve())
            seen.add(uri)
            data = child.read_bytes()
            checksum = sha256_bytes(data)
            prior = None
            if uri in self.manifest:
                prior = self.manifest[uri]
            bundle_id = deterministic_id("bundle", uri, checksum)
            if prior is None:
                result = self.ingest(child, metadata=metadata, user=user)
                if isinstance(result, PipelineResult) and not result.success:
                    raise IngestionError(result.error or f"failed to ingest {uri}")
                self.manifest.record(
                    uri,
                    bundle_id=bundle_id,
                    checksum=checksum,
                )
                summary["added"].append(uri)
            elif prior.get("checksum") != checksum:
                # Retire the prior bundle id from the manifest before
                # re-ingesting so the manifest lookup cannot return a
                # stale record on the next incremental ingest.
                prior_bundle_id = str(prior.get("bundle_id", ""))
                result = self.ingest(child, metadata=metadata, force=True, user=user)
                if isinstance(result, PipelineResult) and not result.success:
                    raise IngestionError(result.error or f"failed to ingest {uri}")
                if prior_bundle_id and prior_bundle_id != bundle_id:
                    # ``delete`` uses both ``bundle_id`` from the
                    # manifest and ``source_uri`` so the old vector
                    # chunks are removed even if the bundle_id is
                    # later reused.
                    self.delete(prior_bundle_id)
                self.manifest.record(
                    uri,
                    bundle_id=bundle_id,
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
            prior_record = self.manifest[prior_uri]
            bundle_id = str(prior_record.get("bundle_id", ""))
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
    ) -> list[ConversationTurn]:
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
            user: Optional :class:`UserPrincipal` whose
                ``user_id`` / ``email`` scopes the delete. When
                omitted, the raw ``session_id`` is used.
        """
        scoped = self.scoped_session_id(user, session_id) or session_id
        self.conversation_store.clear(scoped)


