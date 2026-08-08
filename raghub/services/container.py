"""Composition root: container dataclass and its build helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raghub.config import Settings
from raghub.conv import ConversationHistory
from raghub.embedder import Embedder
from raghub.ingest import Ingestor
from raghub.llm import Generator
from raghub.parsers import Catalog
from raghub.prompts import Prompt
from raghub.repos import UnitOfWork
from raghub.retrieval import Retrieval as RetrievalPipeline
from raghub.services.diagnostics import (
    build_logger,
    build_models,
    seed_blocked,
    seed_demo_users,
)
from raghub.store import Store, build_store
from raghub.stores import ImageStore, Sessions

if TYPE_CHECKING:
    from raghub.auth import AuthService, Authz, SqliteUsers
    from raghub.rag.facade import RAG
    from raghub.services.documents import Documents
    from raghub.services.health import Health
    from raghub.services.query import Query


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
    conversation: ConversationHistory
    embeddings: Embedder
    llm: Generator
    vector_store: Store
    prompt_builder: Prompt
    ingestion: Ingestor
    retrieval: RetrievalPipeline
    image_store: ImageStore
    user_store: SqliteUsers
    parser_registry: Catalog
    store: Sessions
    uow: UnitOfWork
    auth: "AuthService" = None
    documents: "Documents" = None
    query: "Query" = None
    health: "Health" = None
    rag_facade: "RAG | None" = None


async def build_container(settings: Settings) -> RagContainer:
    """Construct a fully-wired :class:`RagContainer`.

    Raises:
        RuntimeError: When ``JWT_SECRET`` is missing.

    """
    logger, authorization, user_store = await build_auth_components(settings)
    raw_session_store, uow, vector_store = await build_storage_components(settings)
    nvidia_api_key = settings.nvidia_api_key or settings.extra.get("nvidia_api_key", "")
    model_components = build_models(
        settings, vector_store, uow, nvidia_api_key
    )
    (
        embeddings,
        llm,
        retrieval,
        ingestion,
        conversation,
        prompt_builder,
        image_store,
        parser_registry,
    ) = model_components
    del model_components
    await maybe_seed_demo_users(settings, logger, user_store)
    return RagContainer(
        settings=settings,
        logger=logger,
        metrics=None,
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


async def build_auth_components(
    settings: Settings,
) -> tuple[Any, Any, Any]:
    """Build the logger, ``Authz`` coordinator, and user store."""
    from raghub.auth import Authz, SqliteUsers

    logger = build_logger(settings.log_level)
    user_store = SqliteUsers(settings.data_dir / "users.db")
    await user_store.initialize()
    jwt_secret = settings.jwt_secret.get_secret_value()
    if not jwt_secret:
        raise RuntimeError("JWT_SECRET must be configured")
    authorization = Authz(user_store, logger=logger)
    return logger, authorization, user_store


async def build_storage_components(settings: Settings) -> tuple[Any, Any, Store]:
    """Build the raw session store, the unit of work, and the vector store."""
    raw_session_store = Sessions(
        settings.data_dir / "sessions.db",
        settings.session_timeout_seconds,
    )
    await raw_session_store.initialize()
    vector_store: Store = build_store(settings, embedding_dim=settings.embedding_dim)
    db_path = str(settings.registry_path).replace(".json", ".db")
    uow = UnitOfWork(
        db_path=db_path,
        vector_store=vector_store,
        session_timeout=settings.session_timeout_seconds,
    )
    await uow.initialize()
    return raw_session_store, uow, vector_store


async def maybe_seed_demo_users(
    settings: Settings,
    logger: object,
    user_store: "SqliteUsers",
) -> None:
    """Seed demo users when the deployment profile allows it."""
    from contextlib import suppress

    if seed_blocked(settings):
        info = getattr(logger, "warning", None) or getattr(logger, "info", None)
        if callable(info):
            with suppress(Exception):
                info("seed.skipped", reason="production_or_wildcard_cors")
        return
    await seed_demo_users(user_store)


__all__ = ["RagContainer", "build_container"]
