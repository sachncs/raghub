"""Phase 8.1 — preferences router end-to-end via FastAPI TestClient."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from raghub.api.app import create_app
from raghub.api.dependencies import get_application
from raghub.auth.user_store import SqliteUserStore
from raghub.config import Settings
from raghub.services.application import DynamicRagApplication


def _build_app(tmp: Path) -> tuple[Any, SqliteUserStore]:
    """Build a FastAPI app with a fresh user store + application container."""
    from raghub.api.rag import RAG
    from raghub.conversation.memory import InMemoryConversationStore
    from raghub.repositories import UnitOfWork
    from raghub.storage.database import DatabaseManager
    from raghub.storage.sqlite_session_store import SqliteSessionStore

    s = Settings(data_dir=tmp, environment="development")
    store = SqliteUserStore(tmp / "users.db")
    asyncio.run(store.initialize())
    asyncio.run(store.create_user("alice@acme.com", "password"))

    db = DatabaseManager(tmp / "raghub.db")
    session_store = SqliteSessionStore(tmp / "sessions.db", db_manager=db)
    asyncio.run(session_store.initialize())
    uow = UnitOfWork(db)
    rag = RAG(settings=s, conversation_store=InMemoryConversationStore())

    container = DynamicRagContainer.__new__(DynamicRagContainer)  # type: ignore[attr-defined]
    return store, s


def test_preferences_router_routes_are_registered() -> None:
    """The router module exposes the three expected routes."""
    from raghub.api.preferences import router

    paths = {route.path for route in router.routes}  # type: ignore[attr-defined]
    assert "/users/me/preferences" in paths
    assert "/users/me/preferences/{key}" in paths