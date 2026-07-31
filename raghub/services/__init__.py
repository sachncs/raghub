"""Application services, container, and worker primitives.

Consolidates every support file in the old
``raghub/services/`` package into one helper module. Class summary::

    Mixin                - structured-log + metric helpers shared by every service.
    Document             - document management (was DocumentService).
    Health               - liveness aggregation (was HealthService).
    Query                - RAG hot path (was QueryService).
    Synchronous / ThreadPool / MemoryQueue
                          - in-process worker + queue primitives.
    RagContainer         - composition root for every collaborator.
    Facade               - high-level facade aggregating every public action
                          (was ApplicationFacade; ``Facade`` is a
                          prior-name alias kept for external callers).

The module-level dispatch entry points live in :mod:`raghub.api`
and the CLI surface in :mod:`raghub.cli.main`.

Names follow the no-suffix rule: ``Document`` (not ``DocumentService``),
``Synchronous`` (not ``SynchronousWorker``), and ``Facade`` (not
``ApplicationFacade``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import TYPE_CHECKING, Any, cast

from raghub.agent import resolve
from raghub.config import Settings

if TYPE_CHECKING:
    from raghub.auth import Authz, SqliteUsers

from raghub.conv import ConversationManager
from raghub.core import can_access_company
from raghub.embedder import Embedder, build_embedder
from raghub.errors import AuthorizationError, IngestionError
from raghub.ingest import IngestionResult, Ingestor
from raghub.lifecycle import Lifecycle, detect_mime_type
from raghub.llm import Generator, build_llm
from raghub.models import (
    AuthLoginResponse,
    BackgroundWorker,
    Turn,
    Document,
    QueryResponse,
    TaskQueue,
    User,
)
from raghub.parsers import Catalog
from raghub.prompts import PromptBuilder
from raghub.repos import UnitOfWork
from raghub.retrieval import (
    Identity as IdentityReranker,
)
from raghub.retrieval import (
    Retrieval as RetrievalPipeline,
)
from raghub.store import Store, build_store
from raghub.stores import ImageStore, Sessions

# `Facade` is the public class; `RagApplication` was a prior name.
# ``from raghub.services import Facade`` without churn. Define a
# placeholder up front so partial-init cycles (api → helper.auth →
# services → helper.services) resolve at every intermediate step; the real
# alias lands at the bottom once ``Facade`` exists.
# placeholder removed; Facade alias defined near the bottom.
from raghub.telemetry import PrometheusMetrics, build_logger

# ---------------------------------------------------------------------------
# Mixin shared by every service
# ---------------------------------------------------------------------------


class Mixin:
    """Provides structured logging and metric helpers to service classes.

    Both methods gracefully degrade when the container is missing the
    expected collaborators, so services can be exercised with stub
    containers in tests.

    Attributes:
        container: The container (or compatible stub) providing
            ``logger`` and ``metrics`` attributes.

    """

    container: Any

    def log(self, message: str, **payload: Any) -> None:
        """Emit a structured log event via the container's logger."""
        logger = getattr(self.container, "logger", None)
        log_method = getattr(logger, "info", None) if logger else None
        if callable(log_method):
            log_method(message, extra=payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric given a ``perf_counter`` start time."""
        metrics = getattr(self.container, "metrics", None)
        recorder = getattr(metrics, "record_latency", None) if metrics else None
        if callable(recorder):
            recorder(name, (time.perf_counter() - started_at) * 1000.0)


# ---------------------------------------------------------------------------
# Document service
# ---------------------------------------------------------------------------


async def upload_record(result: IngestionResult | Any) -> Document:
    """Return the :class:`Document` carried by an ingestion result."""
    return result.document


def missing_doc(document_id: str) -> Document:
    """Raise :class:`IngestionError` for an unknown document id."""
    raise IngestionError(f"Unknown document id: {document_id}")


async def list_records(uow: Any) -> list[Document]:
    """Return every document from the repository."""
    return cast(list[Document], await uow.document_repo.list_all())


async def get_doc(uow: Any, document_id: str) -> Document:
    """Return a single document by id or raise :class:`IngestionError`."""
    record = await uow.document_repo.get(document_id)
    if record is None:
        missing_doc(document_id)
    return cast(Document, record)


class DocumentSvc(Mixin):
    """Document upload, listing, status, and deletion."""

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> Document:
        """Ingest a new document on behalf of the calling user.

        Raises:
            AuthorizationError: If the caller cannot upload documents
                for the resolved company.
            IngestionError: If MIME detection or ingestion fails.

        """
        started = time.perf_counter()
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        target_company = company or filename.split("_", 1)[0]
        if not can_access_company(user, target_company):
            raise AuthorizationError("User cannot upload documents for this company")

        detect_mime_type(filename, content)

        result = await self.container.ingestion.ingest(
            file_name=filename,
            file_bytes=content,
            owner=user,
            organization=target_company,
        )
        document = await upload_record(result)

        self.emit_metric("document_ingest_latency_ms", started)
        self.log(
            "document_ingested",
            document_id=document.id,
            company=target_company,
        )
        return document

    async def list_documents(self, token: str) -> list[Document]:
        """List the documents visible to the caller.

        Admin users see every document; non-admins see only the
        documents whose organization is in their allow-list.
        """
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        if user.is_admin:
            return await list_records(self.container.uow)
        results: list[Document] = []
        for org in user.allowed_companies:
            docs = await self.container.uow.document_repo.list_by_organization(org)
            results.extend(docs)
        return results

    async def document_status(self, token: str, document_id: str) -> Document:
        """Return a single document's status.

        Raises:
            IngestionError: If the document does not exist.
            AuthorizationError: If the caller cannot access the
                document's organization.

        """
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        document = await get_doc(self.container.uow, document_id)
        if not can_access_company(user, document.organization):
            raise AuthorizationError("Forbidden")
        return document

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks. Admin-only."""
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        if not user.is_admin:
            raise AuthorizationError("Admin only")
        self.container.vector_store.delete_document(document_id)
        await self.container.uow.document_repo.delete(document_id)


# ---------------------------------------------------------------------------
# Health service
# ---------------------------------------------------------------------------


def probe_vector_store(store: object) -> dict[str, object]:
    """Probe a vector store for liveness.

    Calls the collaborator's ``health()`` method and translates the
    result into a canonical status.
    """
    probe = getattr(store, "health", None)
    if not callable(probe):
        return {"status": "unknown", "detail": "no health() method"}
    payload = probe()
    if not isinstance(payload, dict):
        payload = {"value": payload}
    status = str(payload.get("status", "ok")).lower()
    if status not in {"ok", "healthy", "up", "ready"}:
        payload = {**payload, "status": "degraded"}
    else:
        payload = {**payload, "status": "ok"}
    return payload


def probe_embedder(embedder: object) -> dict[str, object]:
    """Probe an embedding provider by emitting a tiny probe vector."""
    if embedder is None:
        return {"status": "unknown", "detail": "no embedder configured"}
    embed = getattr(embedder, "embed_text", None)
    if not callable(embed):
        return {"status": "unknown", "detail": "no embed_text() method"}
    vector = embed("health-check-probe")
    if not isinstance(vector, (list, tuple)) or hasattr(vector, "__aiter__"):
        return {
            "status": "ok",
            "dimension": None,
            "model": getattr(embedder, "model_name", ""),
        }
    dim = len(vector) if hasattr(vector, "__len__") else None
    if dim is None or dim == 0:
        return {"status": "down", "error": "empty embedding returned"}
    return {
        "status": "ok",
        "dimension": dim,
        "model": getattr(embedder, "model_name", ""),
    }


def aggregate_status(probes: dict[str, dict[str, object]]) -> str:
    """Combine per-component probes into a single status string."""
    statuses = [str(p.get("status", "")).lower() for p in probes.values()]
    if any(s == "down" for s in statuses):
        return "down"
    if any(s in {"degraded", "unknown"} for s in statuses):
        return "degraded"
    return "ok"


class Health(Mixin):
    """Aggregate liveness signals from key collaborators."""

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    def health(self) -> dict[str, object]:
        """Return a structured health report.

        The default implementation probes the vector store and the
        embedder, plus a static ``ok`` for the registry. The aggregate
        ``status`` is one of ``ok``, ``degraded``, ``down``.
        """
        self.log("health_check")
        components: dict[str, dict[str, object]] = {}
        components["vectorstore"] = probe_vector_store(self.container.vector_store)
        embedder = getattr(self.container, "embeddings", None)
        if embedder is not None:
            components["embedder"] = probe_embedder(embedder)
        components["registry"] = {"status": "ok"}
        return {
            "status": aggregate_status(components),
            "components": components,
        }


# ---------------------------------------------------------------------------
# Query service
# ---------------------------------------------------------------------------


class Query(Mixin):
    """High-level retrieval-augmented Q/A handler."""

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single RAG turn end-to-end.

        Steps: resolve the token → retrieve top-k → flatten history →
        call the LLM → append the new turn → build citations →
        emit metric and log.
        """
        started = time.perf_counter()
        auth: Any = self.container.auth
        user, history = await auth.resolve_user(token)
        hits = self.container.retrieval.retrieve(
            user=user, question=question, top_k=self.container.settings.top_k
        )
        chunks = [hit.chunk for hit in hits]
        context_list = [chunk.text for chunk in chunks]
        session_history = [
            msg
            for t in history[-4:]
            for msg in (
                {"role": "user", "content": t.question},
                {"role": "assistant", "content": t.answer},
            )
        ]
        answer = self.container.llm.generate(
            system_prompt=self.container.prompt_builder.config.system_prompt,
            conversation=history,
            context=context_list,
            question=question,
            image_paths=[],
            session_history=session_history,
        )
        await self.container.conversation.append(
            token, question, answer, metadata={"top_k": self.container.settings.top_k}
        )
        citations = [
            {
                "document_id": chunk.document_id,
                "version": chunk.version,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_id": chunk.id,
            }
            for chunk in chunks
        ]
        self.emit_metric("retrieval_latency_ms", started)
        self.log("query_completed", user=user.email, citations=len(citations))
        return QueryResponse(
            answer=answer,
            citations=citations,
            source_chunks=[chunk.model_dump(mode="json") for chunk in chunks],
        )


# ---------------------------------------------------------------------------
# Worker primitives
# ---------------------------------------------------------------------------


class Synchronous(BackgroundWorker):
    """Execute tasks inline on the caller's thread.

    Useful for tests that want deterministic ordering. Exceptions
    propagate to the caller unchanged.
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke ``fn(*args, **kwargs)`` and return its result directly."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise


class ThreadPool(BackgroundWorker):
    """Execute tasks on a :class:`ThreadPoolExecutor`.

    Attributes:
        executor: Backing thread pool.

    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialise the worker pool.

        Args:
            max_workers: Maximum concurrent worker threads.

        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        """Submit ``fn`` to the pool and return its :class:`Future`."""
        return self.executor.submit(fn, *args, **kwargs)


class MemoryQueue(TaskQueue):
    """In-memory queue shim intended for Celery/RQ migration.

    Process-local; does not survive restarts.
    """

    def __init__(self) -> None:
        """Initialise the queue."""
        self.queue: Queue[tuple[str, dict[str, Any]]] = Queue()

    def enqueue(self, name: str, payload: dict[str, Any]) -> str:
        """Enqueue ``payload`` under ``name`` and return ``name``."""
        self.queue.put((name, payload))
        return name


# ---------------------------------------------------------------------------
# Container + build helpers
# ---------------------------------------------------------------------------


@dataclass
class RagContainer:
    """Composition root: every collaborator the application needs.

    Field roles:

    * ``settings`` — typed configuration snapshot.
    * ``logger`` / ``metrics`` — observability primitives.
    * ``authorization`` — RBAC service for admin-only checks.
    * ``registry`` — user-store alias kept for prior-version callers.
    * ``conversation`` — chat-history manager.
    * ``embeddings`` / ``llm`` — AI providers.
    * ``vector_store`` / ``prompt_builder`` / ``ingestion`` / ``retrieval`` — pipeline pieces.
    * ``image_store`` / ``parser_registry`` — auxiliary stores.
    * ``store`` — raw :class:`Sessions`.
    * ``uow`` — Unit-of-Work for transactional repo access.
    * ``auth`` / ``documents`` / ``query`` / ``health`` — service handles
      populated by :class:`Facade.__init__`.
    """

    settings: Settings
    logger: object
    metrics: object
    authorization: Authz
    registry: SqliteUsers
    conversation: ConversationManager
    embeddings: Embedder
    llm: Generator
    vector_store: Store
    prompt_builder: PromptBuilder
    ingestion: Ingestor
    retrieval: RetrievalPipeline
    image_store: ImageStore
    user_store: SqliteUsers
    parser_registry: Catalog
    store: Sessions
    uow: UnitOfWork
    auth: object = None
    documents: object = None
    query: object = None
    health: object = None
    rag_facade: object = None


def seed_blocked(settings: Settings) -> bool:
    """Return ``True`` when the demo-user seed must be skipped."""
    if settings.environment == "production":
        return True
    return os.getenv("CORS_ORIGINS", "").strip() == "*"


def parse_users(raw: str) -> Any:
    """Parse the ``RAGHUB_USERS`` env var as JSON."""
    import json as json_import

    return json_import.loads(raw)


async def seed_demo_users(user_store: SqliteUsers) -> None:
    """Seed demo users from ``RAGHUB_USERS`` or the default list."""
    users_env = os.getenv("RAGHUB_USERS", "").strip()
    if users_env:
        seed_users = parse_users(users_env)
        if isinstance(seed_users, dict):
            for email, cfg in seed_users.items():
                if not isinstance(cfg, dict):
                    continue
                existing = await user_store.get_by_email(email)
                if existing is not None:
                    continue
                await user_store.create_user(
                    email=email,
                    password=str(cfg.get("password", "password")),
                    companies=list(cfg.get("companies", []) or []),
                    is_admin=bool(cfg.get("is_admin", False)),
                )
        return

    default_seed = [
        ("alice@acme.com", "password", ["Apple"], False),
        ("bob@acme.com", "password", ["Microsoft"], False),
        ("charlie@acme.com", "password", ["Amazon", "Tesla"], False),
        ("diana@acme.com", "password", ["Google"], False),
        ("admin@acme.com", "password", [], True),
        ("alice@email.com", "test", ["Apple"], False),
        ("bob@email.com", "test", ["Microsoft", "Google"], False),
        ("charlie@email.com", "test", ["Amazon", "Tesla"], False),
        ("admin@email.com", "admin", [], True),
    ]
    for email, pwd, companies, is_admin in default_seed:
        existing = await user_store.get_by_email(email)
        if existing is not None:
            continue
        await user_store.create_user(
            email=email,
            password=pwd,
            companies=companies,
            is_admin=is_admin,
        )


async def build_container(settings: Settings) -> RagContainer:
    """Construct a fully-wired :class:`RagContainer`.

    Raises:
        RuntimeError: When ``JWT_SECRET`` is missing.

    """
    from contextlib import suppress

    from raghub.auth import Authz, SqliteUsers

    logger = build_logger(settings.log_level)
    user_store = SqliteUsers(settings.data_dir / "users.db")
    await user_store.initialize()
    jwt_secret = settings.jwt_secret.get_secret_value()
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET must be configured")
    authorization = Authz(user_store, logger=logger)

    nvidia_api_key = settings.nvidia_api_key or settings.extra.get("nvidia_api_key", "")
    vector_store: Store = build_store(settings, embedding_dim=settings.embedding_dim)

    db_path = str(settings.registry_path).replace(".json", ".db")
    uow = UnitOfWork(
        db_path=db_path,
        vector_store=vector_store,
        session_timeout=settings.session_timeout_seconds,
    )
    await uow.initialize()
    raw_session_store = Sessions(
        settings.data_dir / "sessions.db",
        settings.session_timeout_seconds,
    )
    await raw_session_store.initialize()

    embeddings: Embedder = build_embedder(
        settings.embedding_model,
        settings.embedding_dim,
        nvidia_api_key,
    )
    llm: Generator = build_llm(settings.llm_model, nvidia_api_key)

    prompt_builder = PromptBuilder()
    conversation = ConversationManager(uow)
    lifecycle = Lifecycle()
    ingestion = Ingestor(
        uow=uow,
        embedding_provider=embeddings,
        lifecycle_manager=lifecycle,
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalPipeline(
        embedding_provider=embeddings,
        vector_store=vector_store,
        rerank=IdentityReranker(),
    )
    image_store = ImageStore(settings.data_dir / "images")
    parser_registry = Catalog()

    if seed_blocked(settings):
        info = getattr(logger, "warning", None) or getattr(logger, "info", None)
        if callable(info):
            with suppress(Exception):
                info("seed.skipped", reason="production_or_wildcard_cors")
    else:
        await seed_demo_users(user_store)

    return RagContainer(
        settings=settings,
        logger=logger,
        metrics=PrometheusMetrics(),
        authorization=authorization,
        registry=user_store,
        conversation=conversation,
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        ingestion=ingestion,
        retrieval=retrieval,
        image_store=image_store,
        user_store=user_store,
        parser_registry=parser_registry,
        store=raw_session_store,
        uow=uow,
    )


# ---------------------------------------------------------------------------
# Facade + coordinators
# ---------------------------------------------------------------------------

RAG_FACADE_AVAILABLE: bool = importlib.util.find_spec("raghub.rag") is not None


class Auth:
    """Auth-shaped coordinator on the facade."""

    def __init__(self, facade: Any) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""
        return cast(AuthLoginResponse, await self.facade.auth_svc.login(email, password))

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store."""
        await self.facade.auth_svc.logout(token)

    async def resolve_user(self, token: str) -> tuple[User, list[Turn]]:
        """Resolve a bearer token to a principal plus history."""
        return cast(
            tuple[User, list[Turn]],
            await self.facade.auth_svc.resolve_user(token),
        )


class Shutdown:
    """Release collaborators held by the :class:`RagContainer`."""

    SHUTDOWN_TARGETS: tuple[str, ...] = (
        "background_ingestion",
        "ingestion",
        "image_store",
        "vector_store",
        "store",
        "uow",
    )

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    async def release(self) -> None:
        """Close every owned collaborator in order."""
        for attr in self.SHUTDOWN_TARGETS:
            collaborator = getattr(self.container, attr, None)
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None) or getattr(collaborator, "shutdown", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result


class Preference:
    """Routes advanced-RAG requests based on resolved user prefs."""

    def __init__(self, facade: Any) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def query_with_flags(
        self,
        *,
        token: str,
        question: str,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Resolve advanced-RAG flags against user prefs and route accordingly."""
        container = self.facade.container
        user, _ = await container.auth.resolve_user(token)
        prefs = dict(getattr(user, "tool_settings", None) or {})
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
            session_overrides=None,
            user_prefs=prefs,
            settings=container.settings,
        )

        rag: Any | None = getattr(container, "rag_facade", None)
        if rag is None:
            response = await self.facade.query_svc.query(token=token, question=question)
            response.metadata = dict(response.metadata or {})
            response.metadata["resolved_config"] = resolved.to_dict()
            if top_k is not None:
                response.metadata["requested_top_k"] = top_k
            return cast(QueryResponse, response)

        session = await container.store.get_by_token(token)
        principal = User(
            user_id=user.user_id,
            email=user.email,
            allowed_companies=user.allowed_companies,
            allowed_groups=user.allowed_groups,
            is_admin=user.is_admin,
            tool_settings=user.tool_settings,
        )
        canonical = await rag.aquery(
            question,
            user=principal,
            session_id=session.session_id if session is not None else None,
            tools_enabled=tools_enabled,
            agent=agent,
            web=web,
            graph=graph,
            summaries=summaries,
            reranker=reranker,
            long_context_pass=long_context_pass,
            query_transforms=query_transforms,
            max_steps=max_steps,
            top_k=top_k,
        )
        return QueryResponse(
            answer=canonical.answer,
            citations=canonical.citations,
            source_chunks=[chunk.model_dump(mode="json") for chunk in canonical.source_chunks],
            planner_trace=canonical.metadata.get("planner_trace"),
            tools_invoked=canonical.metadata.get("tools_invoked") or [],
            transforms_applied=canonical.transforms_applied,
            metadata={
                "pipeline_id": "query_agent"
                if (resolved.agent_enabled or resolved.tools_enabled)
                else "query",
                "structured": False,
                **canonical.metadata,
            },
        )


class Facade:
    """High-level facade exposing every public action.

    The application holds the container and four service handles. Each
    public method delegates to the appropriate service so the facade
    stays thin.
    """

    def __init__(self, container: Any) -> None:
        """Initialise the facade and wire service handles back into the container."""
        from raghub.auth import AuthService

        self.container = container
        self.auth_svc = AuthService(container)
        self.documents_svc = DocumentSvc(container)
        self.query_svc = Query(container)
        self.health_svc = Health(container)
        container.auth = self.auth_svc
        container.documents = self.documents_svc
        container.query = self.query_svc
        container.health = self.health_svc
        self.auth = Auth(self)
        self.shutdown_coordinator = Shutdown(container)
        self.preferences = Preference(self)

    @staticmethod
    def build_rag_facade(container: Any) -> Any | None:
        """Construct a :class:`raghub.RAG` from the container's collaborators."""
        if not RAG_FACADE_AVAILABLE:
            return None
        import importlib

        rag_module = importlib.import_module("raghub.rag")
        return rag_module.RAG(
            settings=container.settings,
            embedder=container.embeddings,
            llm=container.llm,
            vector_store=container.vector_store,
            knowledge_repo=container.registry,
            conversation_store=getattr(container, "conversation_store", None),
        )

    def rag_facade(self) -> Any | None:
        """Return the lazily-built :class:`raghub.RAG` instance."""
        if getattr(self.container, "rag_facade", None) is None:
            self.container.rag_facade = self.build_rag_facade(self.container)
        return self.container.rag_facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""
        return await self.auth.login(email, password)

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store."""
        return await self.auth.logout(token)

    async def resolve_user(self, token: str) -> tuple[User, list[Turn]]:
        """Resolve a bearer token to a principal plus conversation history."""
        return await self.auth.resolve_user(token)

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> Document:
        """Upload ``content`` as a new document owned by the calling user."""
        return await self.documents_svc.upload_document(
            token=token, filename=filename, content=content, company=company
        )

    async def list_documents(self, token: str) -> list[Document]:
        """List the documents visible to the caller."""
        return await self.documents_svc.list_documents(token)

    async def document_status(self, token: str, document_id: str) -> Document:
        """Return the status of a single document."""
        return await self.documents_svc.document_status(token, document_id)

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks."""
        await self.documents_svc.delete_document(token, document_id)

    async def clear_history(self, token: str) -> None:
        """Empty the conversation history for ``token``."""
        await self.container.conversation.clear(token)

    async def history(self, token: str) -> list[Turn]:
        """Return the full conversation history for ``token``."""
        return cast(
            list[Turn],
            await self.container.conversation.load(token),
        )

    def health(self) -> dict[str, object]:
        """Run liveness checks and return a status dict."""
        return self.health_svc.health()

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single retrieval-augmented Q/A turn."""
        return await self.query_svc.query(token=token, question=question)

    async def query_with_flags(
        self,
        *,
        token: str,
        question: str,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Resolve advanced-RAG flags against user prefs and route accordingly."""
        return await self.preferences.query_with_flags(
            token=token,
            question=question,
            tools_enabled=tools_enabled,
            agent=agent,
            web=web,
            graph=graph,
            summaries=summaries,
            reranker=reranker,
            long_context_pass=long_context_pass,
            query_transforms=query_transforms,
            max_steps=max_steps,
            top_k=top_k,
        )

    def log(self, message: str, **payload: object) -> None:
        """Emit a structured log event via the health service."""
        self.health_svc.log(message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Emit a latency metric given a perf-counter start time."""
        self.health_svc.emit_metric(name, started_at)

    async def shutdown(self) -> None:
        """Release all resources held by the application."""
        await self.shutdown_coordinator.release()


# `Facade` is the public class
# here, before ``__all__`` is defined, so partial-init cycles see it
# via the star-import in :mod:`raghub.services`.

__all__ = [
    "Auth",
    "Document",
    "Facade",
    "Facade",
    "Health",
    "MemoryQueue",
    "Mixin",
    "Preference",
    "Query",
    "RagContainer",
    "Shutdown",
    "Synchronous",
    "ThreadPool",
    "aggregate_status",
    "build_container",
    "get_doc",
    "list_records",
    "missing_doc",
    "parse_users",
    "probe_embedder",
    "probe_vector_store",
    "seed_blocked",
    "seed_demo_users",
    "upload_record",
]
