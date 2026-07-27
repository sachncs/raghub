"""Container and facade wiring every collaborator.

This package is the **single assembly point** for the entire RAG
application. :func:`build_container` instantiates every collaborator
(settings, logger, metrics, auth, RBAC, sessions, embeddings, LLM,
vector store, ingestion, retrieval, parsers, image store) into a
:class:`RagContainer`, then :class:`ApplicationFacade` (re-exported
as :class:`RagApplication`) wraps that container with the
high-level facade methods the API and CLI call.

The package is split into focused modules:

* :mod:`raghub.services.application.facade` — :class:`ApplicationFacade`,
  the public-facing facade.
* :mod:`raghub.services.application.shutdown` — :class:`ShutdownCoordinator`,
  the resource release loop.
* :mod:`raghub.services.application.auth` — :class:`AuthCoordinator`,
  the auth-aware façade methods (``login``, ``logout``, ``resolve_user``).
* :mod:`raghub.services.application.preferences` —
  :class:`PreferenceCoordinator`, the resolved-config and preference
  routing for ``query_with_flags``.

Production safety
-----------------

:func:`build_container` refuses to seed the demo users when
``settings.environment == "production"`` or when ``CORS_ORIGINS`` is
``"*"`` — both signal a production-style deploy that must not silently
fall back to the ``alice@acme.com / password`` defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from raghub.auth import RBACAuthorizationService, SqliteUserStore
from raghub.config import Settings
from raghub.conversation import ConversationManager
from raghub.documents import DocumentLifecycleManager
from raghub.documents.parser import Catalog
from raghub.embeddings import BaseEmbeddingProvider, build_embedding_provider
from raghub.ingestion import DocumentIngestionService
from raghub.llm import BaseLLMProvider, build_llm_provider
from raghub.observability import PrometheusMetrics, build_logger
from raghub.prompts import PromptBuilder
from raghub.repositories import UnitOfWork
from raghub.retrieval.helper import Identity as IdentityReranker, Retrieval as RetrievalPipeline
from raghub.services.application.facade import ApplicationFacade
from raghub.storage.image_store import FilesystemImageStore
from raghub.storage.sqlite_session_store import SqliteSessionStore
from raghub.vectorstore import BaseVectorStore, ZvecVectorStore


def seed_blocked(settings: Settings) -> bool:
    """Return ``True`` when the demo-user seed must be skipped.

    The seed is suppressed when either signal of a production deploy
    is present:

    * ``settings.environment == "production"`` — explicit opt-in to
      production semantics.
    * ``CORS_ORIGINS`` is ``"*"`` — the same misconfiguration that the
      CORS guard rejects at startup; if the operator left it as a
      wildcard the platform is not configured for production and
      must not silently create default accounts.

    Args:
        settings: The loaded application settings.

    Returns:
        ``True`` when the demo seed must be skipped.
    """
    if settings.environment == "production":
        return True
    cors = os.getenv("CORS_ORIGINS", "").strip()
    return cors == "*"


@dataclass
class RagContainer:
    """Composition root: every collaborator the application needs.

    Field roles:

    * ``settings`` — typed configuration snapshot.
    * ``logger`` / ``metrics`` — observability primitives.
    * ``authorization`` — RBAC service for admin-only checks.
    * ``registry`` — user store aliased for legacy call sites that
      expected a "registry" name (kept for backward compatibility).
    * ``conversation`` — chat-history manager.
    * ``embeddings`` / ``llm`` — AI providers.
    * ``vector_store`` / ``prompt_builder`` / ``ingestion`` / ``retrieval``
      — RAG pipeline pieces.
    * ``image_store`` / ``parser_registry`` — auxiliary stores.
    * ``user_store`` — same instance as ``registry``; named for clarity.
    * ``store`` — raw :class:`SqliteSessionStore` (canonical session
      store used by :class:`AuthService`).
    * ``uow`` — Unit-of-Work for transactional repo access.
    * ``auth`` / ``documents`` / ``query`` / ``health`` — service handles
      populated by :class:`ApplicationFacade.__init__`.

    Attributes:
        settings: Application configuration.
        logger: Loguru logger (see :mod:`raghub.observability`).
        metrics: Prometheus metrics sink.
        authorization: RBAC service.
        registry: Backward-compat alias for ``user_store``.
        conversation: Chat-history manager.
        embeddings: Embedding provider.
        llm: LLM provider.
        vector_store: Vector store.
        prompt_builder: Token-aware prompt builder.
        ingestion: Document ingestion service.
        retrieval: Retrieval pipeline.
        image_store: Filesystem image store.
        user_store: User CRUD store.
        parser_registry: Document format parser registry.
        store: SQLite-backed session store.
        uow: Unit-of-work for repos.
        auth: :class:`AuthService` handle (set by
            :class:`ApplicationFacade`).
        documents: :class:`DocumentService` handle.
        query: :class:`QueryService` handle.
        health: :class:`HealthService` handle.
        rag_facade: Optional :class:`raghub.RAG` instance.
    """

    settings: Settings
    logger: object
    metrics: object
    authorization: RBACAuthorizationService
    registry: SqliteUserStore
    conversation: ConversationManager
    embeddings: BaseEmbeddingProvider
    llm: BaseLLMProvider
    vector_store: BaseVectorStore
    prompt_builder: PromptBuilder
    ingestion: DocumentIngestionService
    retrieval: RetrievalPipeline
    image_store: FilesystemImageStore
    user_store: SqliteUserStore
    parser_registry: Catalog
    store: SqliteSessionStore
    uow: UnitOfWork
    auth: object = None
    documents: object = None
    query: object = None
    health: object = None
    rag_facade: object = None


# ``RagApplication`` is the legacy class name; the new class
# lives in :mod:`raghub.services.application.facade` as
# :class:`ApplicationFacade`. Re-export under the old name so external
# callers (``from raghub.services.application import RagApplication``)
# keep working.
RagApplication = ApplicationFacade


async def _build_auth_components(settings: Settings) -> tuple[Any, Any, SqliteUserStore]:
    """Build logger + user_store + RBAC.

    Returns:
        (logger, authorization_service, user_store).
    """
    logger = build_logger(settings.log_level)
    user_store = SqliteUserStore(settings.data_dir / "users.db")
    await user_store.initialize()
    jwt_secret = settings.jwt_secret.get_secret_value()
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET must be configured")
    authorization = RBACAuthorizationService(user_store, logger=logger)
    return logger, authorization, user_store


async def _build_persistence_components(
    settings: Settings, vector_store: BaseVectorStore
) -> tuple[UnitOfWork, SqliteSessionStore]:
    """Build unit-of-work + session store."""
    db_path = str(settings.registry_path).replace(".json", ".db")
    uow = UnitOfWork(
        db_path=db_path,
        vector_store=vector_store,
        session_timeout=settings.session_timeout_seconds,
    )
    await uow.initialize()
    raw_session_store = SqliteSessionStore(
        settings.data_dir / "sessions.db",
        settings.session_timeout_seconds,
    )
    await raw_session_store.initialize()
    return uow, raw_session_store


def _build_provider_components(
    settings: Settings, nvidia_api_key: str
) -> tuple[BaseEmbeddingProvider, BaseLLMProvider]:
    """Build the embedding + LLM providers via the factory helpers."""
    embeddings: BaseEmbeddingProvider = build_embedding_provider(
        settings.embedding_model,
        settings.embedding_dim,
        nvidia_api_key,
    )
    llm: BaseLLMProvider = build_llm_provider(
        settings.llm_model,
        nvidia_api_key,
    )
    return embeddings, llm


def _build_pipeline_components(
    uow: UnitOfWork,
    embeddings: BaseEmbeddingProvider,
    vector_store: BaseVectorStore,
    settings: Settings,
) -> dict[str, Any]:
    """Build the pipeline collaborators (prompt, conversation, ingestion, retrieval, image)."""
    prompt_builder = PromptBuilder()
    conversation = ConversationManager(uow)
    lifecycle = DocumentLifecycleManager()
    ingestion = DocumentIngestionService(
        uow=uow,
        embedding_provider=embeddings,
        lifecycle_manager=lifecycle,
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalPipeline(
        embedding_provider=embeddings,
        vector_store=vector_store,
        reranker=IdentityReranker(),
    )
    image_store = FilesystemImageStore(settings.data_dir / "images")
    parser_registry = Catalog()
    return {
        "prompt_builder": prompt_builder,
        "conversation": conversation,
        "ingestion": ingestion,
        "retrieval": retrieval,
        "image_store": image_store,
        "parser_registry": parser_registry,
    }


async def _seed_demo_users(logger: Any, user_store: SqliteUserStore, settings: Settings) -> None:
    """Seed demo users unless blocked (production or wildcard CORS)."""
    from contextlib import suppress

    if not seed_blocked(settings):
        await seed_demo_users(user_store)
        return
    info = getattr(logger, "warning", None) or getattr(logger, "info", None)
    if callable(info):
        with suppress(Exception):
            info("seed.skipped", reason="production_or_wildcard_cors")


async def build_container(settings: Settings) -> RagContainer:
    """Construct a fully-wired :class:`RagContainer`.

    The build is ordered so that every collaborator's dependencies are
    available when needed. The actual per-block construction is
    delegated to ``_build_*`` helpers so this orchestrator stays
    under 80 lines.

    Build order:
        1. Logger and metrics + user store + RBAC.
        2. Vector store (``ZvecVectorStore``).
        3. Unit-of-work + session store.
        4. Embeddings + LLM (via factory helpers).
        5. Pipeline collaborators (prompt, conversation, ingestion, retrieval).
        6. Demo-user seeding — skipped in production / wildcard CORS.

    Args:
        settings: The loaded application settings.

    Returns:
        A populated :class:`RagContainer` ready to be wrapped
        by :class:`ApplicationFacade`.

    Raises:
        RuntimeError: When ``JWT_SECRET`` is missing from settings.
    """
    logger, authorization, user_store = await _build_auth_components(settings)
    nvidia_api_key = settings.nvidia_api_key or settings.extra.get("nvidia_api_key", "")
    vector_store: BaseVectorStore = ZvecVectorStore(
        str(settings.zvec_dir),
        embedding_dim=settings.embedding_dim,
        require_zvec=settings.require_zvec,
    )
    uow, raw_session_store = await _build_persistence_components(settings, vector_store)
    embeddings, llm = _build_provider_components(settings, nvidia_api_key)
    pipelines = _build_pipeline_components(uow, embeddings, vector_store, settings)
    await _seed_demo_users(logger, user_store, settings)
    return RagContainer(
        settings=settings,
        logger=logger,
        metrics=PrometheusMetrics(),
        authorization=authorization,
        registry=user_store,
        conversation=pipelines["conversation"],
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        prompt_builder=pipelines["prompt_builder"],
        ingestion=pipelines["ingestion"],
        retrieval=pipelines["retrieval"],
        image_store=pipelines["image_store"],
        user_store=user_store,
        parser_registry=pipelines["parser_registry"],
        store=raw_session_store,
        uow=uow,
    )


async def seed_demo_users(user_store: SqliteUserStore) -> None:
    """Seed demo users from ``RAGHUB_USERS`` or the default list.

    Reads ``RAGHUB_USERS`` (a JSON object) when present; otherwise
    inserts the five documented demo users. Skips any user that
    already exists.

    Args:
        user_store: The user store to populate.
    """
    users_env = os.getenv("RAGHUB_USERS", "").strip()
    if users_env:
        seed_users = parse_seed_users_json(users_env)
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


def parse_seed_users_json(raw: str) -> Any:
    """Parse the ``RAGHUB_USERS`` env var as JSON.

    Args:
        raw: The raw env var value.

    Returns:
        The parsed JSON object.

    Raises:
        json.JSONDecodeError: When ``raw`` is not valid JSON. The
            stdlib error is surfaced verbatim; callers wrap it into a
            :class:`RuntimeError` when the user-facing message needs
            the ``RAGHUB_USERS`` prefix.
    """
    import json as json_import

    return json_import.loads(raw)