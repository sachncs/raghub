"""Tests for ``raghub.services.container`` (RagContainer + build_*_components)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from raghub.config import Settings
from raghub.services.container import (
    RagContainer,
    build_auth_components,
    build_storage_components,
    maybe_seed_demo_users,
)


def test_rag_container_dataclass_carries_required_attrs() -> None:
    """``RagContainer`` accepts every documented collaborator as a typed field."""

    settings = Settings(jwt_secret="x" * 32)
    auth = SimpleNamespace()
    user_store = SimpleNamespace()
    convo = SimpleNamespace()
    embeddings = SimpleNamespace()
    llm = SimpleNamespace()
    vector_store = SimpleNamespace()
    prompt = SimpleNamespace()
    ingestion = SimpleNamespace()
    retrieval = SimpleNamespace()
    image_store = SimpleNamespace()
    parser_registry = SimpleNamespace()
    sessions = SimpleNamespace()
    uow = SimpleNamespace()

    container = RagContainer(
        settings=settings,
        logger=None,
        metrics=None,
        authorization=auth,
        registry=user_store,
        conversation=convo,
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        prompt_builder=prompt,
        ingestion=ingestion,
        retrieval=retrieval,
        image_store=image_store,
        user_store=user_store,
        parser_registry=parser_registry,
        store=sessions,
        uow=uow,
    )
    assert container.settings is settings
    assert container.authorization is auth
    assert container.vector_store is vector_store
    # Service handles default to None and are populated by Facade.
    assert container.auth is None
    assert container.documents is None
    assert container.query is None
    assert container.health is None
    assert container.rag_facade is None


@pytest.mark.asyncio
async def test_build_auth_components_raises_when_jwt_secret_missing() -> None:
    """``build_auth_components`` raises RuntimeError when JWT_SECRET is empty."""

    settings = Settings(jwt_secret="")
    with pytest.raises(RuntimeError, match="JWT_SECRET must be configured"):
        await build_auth_components(settings)


@pytest.mark.asyncio
async def test_maybe_seed_demo_users_logs_and_skips_in_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``maybe_seed_demo_users`` returns early in production without seeding."""

    settings = Settings(jwt_secret="x" * 32, environment="production")
    user_store = SimpleNamespace()
    # The function should NOT call any user_store.create when seed_blocked.
    await maybe_seed_demo_users(settings, user_store)  # must not raise


@pytest.mark.asyncio
async def test_maybe_seed_demo_users_calls_seed_demo_users_in_development() -> None:
    """``maybe_seed_demo_users`` delegates to seed_demo_users in development."""

    settings = Settings(jwt_secret="x" * 32, environment="development")
    user_store = SimpleNamespace()

    with patch("raghub.services.container.seed_demo_users", new=AsyncMock()) as mock_seed:
        await maybe_seed_demo_users(settings, user_store)
        mock_seed.assert_awaited_once_with(user_store)


@pytest.mark.asyncio
async def test_build_storage_components_returns_three_collaborators(tmp_path) -> None:
    """``build_storage_components`` returns (sessions, uow, vector_store)."""

    settings = Settings(
        jwt_secret="x" * 32,
        data_dir=tmp_path,
        embedding_dim=16,
    )
    sessions, uow, vector_store = await build_storage_components(settings)
    assert sessions is not None, "sessions should be set by test setup"
    assert uow is not None, "uow should be set by test setup"
    assert vector_store is not None, "vector_store should be set by test setup"
