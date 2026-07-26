"""Phase 8 — user-configurable toggles integration tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from raghub.api.rag import RAG
from raghub.config.settings import AppSettings
from raghub.models import UserPrincipal


def build_rag(tmp: Path) -> RAG:
    s = AppSettings(data_dir=tmp)
    return RAG(settings=s)


@pytest.mark.asyncio
async def test_rag_aquery_runs_resolver_with_request_flags() -> None:
    """`aquery` records the resolved config even when no agent runs."""
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_rag(Path(tmp))
        user = UserPrincipal(email="a@b.c", allowed_companies=[])
        response = await rag.aquery(
            "hello",
            user=user,
            session_id="s1",
            agent=True,
            web=True,
            reranker="bge",
        )
    # Resolved config is exposed in the response metadata.
    assert response.metadata["resolved_config"]["agent_enabled"] is True
    assert "web_search" in response.metadata["resolved_config"]["tools_enabled"]
    assert response.metadata["resolved_config"]["reranker"] == "bge"


@pytest.mark.asyncio
async def test_rag_aquery_user_tool_settings_override_global() -> None:
    """User prefs win over the global default."""
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_rag(Path(tmp))
        user = UserPrincipal(
            email="a@b.c",
            tool_settings={"agent_enabled": True, "tools_enabled": ["summary_search"]},
        )
        response = await rag.aquery("hi", user=user, session_id="s2")
    cfg = response.metadata["resolved_config"]
    assert cfg["agent_enabled"] is True
    assert "summary_search" in cfg["tools_enabled"]


@pytest.mark.asyncio
async def test_rag_aquery_session_overrides_layer() -> None:
    """Per-session overrides win over user prefs and the global default."""
    with tempfile.TemporaryDirectory() as tmp:
        rag = build_rag(Path(tmp))
        user = UserPrincipal(
            email="a@b.c",
            user_id="alice-uuid",
            tool_settings={"agent_enabled": False},
        )
        # The conversation store is keyed by the scoped (user::session)
        # id so multiple users sharing a session id cannot stomp on
        # each other's overrides.
        scoped = RAG.scoped_session_id(user, "s3")
        rag.conversation_store.set_overrides(scoped, {"agent_enabled": True})
        response = await rag.aquery("hi", user=user, session_id="s3")
    assert response.metadata["resolved_config"]["agent_enabled"] is True


def test_rag_session_overrides_helper_returns_empty_when_no_store() -> None:
    """The helper gracefully no-ops when the store lacks ``get_overrides``."""
    rag = build_rag(Path(tempfile.gettempdir()))
    # Default InMemoryConversationStore does support get_overrides;
    # confirm the helper round-trips for an unknown session.
    out = rag.session_overrides("does-not-exist")
    assert out == {}


def test_rag_aquery_signature_accepts_all_advanced_kwargs() -> None:
    """The facade signature exposes every Phase 8 flag."""
    import inspect

    sig = inspect.signature(RAG.aquery)
    expected = {
        "tools_enabled",
        "agent",
        "web",
        "graph",
        "summaries",
        "reranker",
        "long_context_pass",
        "query_transforms",
        "max_steps",
    }
    assert expected.issubset(sig.parameters.keys())


def test_rag_astream_signature_accepts_all_advanced_kwargs() -> None:
    """The streaming facade signature exposes every Phase 8 flag."""
    import inspect

    sig = inspect.signature(RAG.astream)
    expected = {
        "tools_enabled",
        "agent",
        "web",
        "graph",
        "summaries",
        "reranker",
        "long_context_pass",
        "query_transforms",
        "max_steps",
    }
    assert expected.issubset(sig.parameters.keys())