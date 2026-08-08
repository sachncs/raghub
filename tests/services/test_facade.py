"""Tests for ``raghub.services.facade`` (ApplicationFacade)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raghub.config import Settings
from raghub.services.facade import ApplicationFacade


def _make_container() -> SimpleNamespace:
    """Build a minimal container-like SimpleNamespace for facade tests."""

    settings = Settings(jwt_secret="x" * 32)
    embeddings = SimpleNamespace()
    llm = SimpleNamespace()
    retrieval = SimpleNamespace()
    ingestion = SimpleNamespace()
    conversation = SimpleNamespace(name="conv")
    prompt_builder = SimpleNamespace()
    image_store = SimpleNamespace()
    parser_registry = SimpleNamespace()
    return SimpleNamespace(
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        retrieval=retrieval,
        ingestion=ingestion,
        conversation=conversation,
        prompt_builder=prompt_builder,
        image_store=image_store,
        parser_registry=parser_registry,
    )


def test_facade_init_wires_service_handles_back_into_container() -> None:
    """``ApplicationFacade.__init__`` populates auth/documents/query/health on the container."""

    from raghub.auth import AuthService
    from raghub.services.documents import Documents
    from raghub.services.health import Health
    from raghub.services.query import Query

    container = _make_container()
    facade = ApplicationFacade(container)
    assert isinstance(container.auth, AuthService)
    assert isinstance(container.documents, Documents)
    assert isinstance(container.query, Query)
    assert isinstance(container.health, Health)
    assert facade.shutdown_coordinator is not None
    assert facade.preferences is not None


def test_facade_rag_facade_is_lazy_and_cached() -> None:
    """``rag_facade()`` builds the RAG instance on first call and caches it."""

    container = _make_container()
    facade = ApplicationFacade(container)

    with patch("raghub.services.facade.ApplicationFacade.build_rag") as mock_build:
        rag_instance = MagicMock()
        mock_build.return_value = rag_instance
        # First call triggers build.
        assert facade.rag_facade() is rag_instance
        # Second call returns cached.
        assert facade.rag_facade() is rag_instance
        mock_build.assert_called_once()


def test_facade_login_delegates_to_auth_service() -> None:
    """``ApplicationFacade.login`` returns ``auth.login``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = SimpleNamespace(token="t", user=SimpleNamespace())
    container.auth.login = AsyncMock(return_value=expected)
    import asyncio

    result = asyncio.run(facade.login("alice@example.com", "pwd"))
    assert result is expected


def test_facade_resolve_user_delegates_to_auth() -> None:
    """``ApplicationFacade.resolve_user`` returns ``auth.resolve_user``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = (SimpleNamespace(), [])
    container.auth.resolve_user = AsyncMock(return_value=expected)
    import asyncio

    assert asyncio.run(facade.resolve_user("t")) is expected


def test_facade_health_returns_container_health() -> None:
    """``ApplicationFacade.health`` delegates to ``container.health.health``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = {"status": "ok"}
    container.health.health = MagicMock(return_value=expected)
    assert facade.health() is expected


def test_facade_history_loads_conversation_history() -> None:
    """``ApplicationFacade.history`` returns ``container.conversation.load(token)``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = [SimpleNamespace()]
    container.conversation.load = AsyncMock(return_value=expected)
    import asyncio

    assert asyncio.run(facade.history("t")) is expected


def test_facade_shutdown_calls_shutdown_coordinator() -> None:
    """``ApplicationFacade.shutdown`` awaits ``shutdown_coordinator.release``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    facade.shutdown_coordinator.release = AsyncMock()
    import asyncio

    asyncio.run(facade.shutdown())
    facade.shutdown_coordinator.release.assert_awaited_once()


def test_facade_log_and_emit_metric_delegate_to_container_health() -> None:
    """``log`` and ``emit_metric`` proxy through to ``container.health``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    captured: list[tuple[str, dict[str, object]]] = []
    container.health.log = lambda message, **kwargs: captured.append((message, kwargs))
    container.health.emit_metric = lambda name, started: None
    facade.log("test.event", code=42)
    assert captured == [("test.event", {"code": 42})]


def test_facade_query_delegates_to_container_query() -> None:
    """``ApplicationFacade.query`` delegates to ``container.query.query``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = SimpleNamespace(answer="a", citations=[], source_chunks=[])
    container.query.query = AsyncMock(return_value=expected)
    import asyncio

    assert asyncio.run(facade.query(token="t", question="q")) is expected


def test_facade_query_with_flags_delegates_to_preferences() -> None:
    """``ApplicationFacade.query_with_flags`` delegates to ``preferences.query_with_flags``."""

    container = _make_container()
    facade = ApplicationFacade(container)
    expected = SimpleNamespace(answer="a", citations=[], source_chunks=[])
    facade.preferences.query_with_flags = AsyncMock(return_value=expected)
    import asyncio

    assert asyncio.run(facade.query_with_flags(token="t", question="q")) is expected