"""Coverage tests for :mod:`raghub.tools`."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from raghub.errors import ConfigurationError
from raghub.models import User
from raghub.tools import (
    DateToday,
    GraphSearch,
    HybridSearch,
    KeywordSearch,
    Tool,
    ToolContext,
    ToolRegistry,
    WebSearch,
    as_admin_user,
)


def _user(**overrides: Any) -> User:
    """Build a :class:`User` with sensible defaults."""
    defaults: dict[str, Any] = {
        "id": "u-1",
        "email": "alice@example.com",
        "is_admin": False,
        "allowed_companies": ["acme"],
        "allowed_groups": [],
    }
    defaults.update(overrides)
    return User(**defaults)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def _fake_tool(name: str) -> Any:
    """Build a fake tool with a ``name`` and ``json_schema``."""
    tool = MagicMock()
    tool.name = name
    tool.json_schema = {"name": name, "type": "object"}
    return tool


def test_tool_registry_register_and_get() -> None:
    """``register`` adds a tool; ``get`` returns it."""
    registry = ToolRegistry()
    tool = _fake_tool("web_search")
    registry.register(tool)
    assert registry.get("web_search") is tool


def test_tool_registry_register_replaces_existing() -> None:
    """Re-registering a name overwrites the prior binding."""
    registry = ToolRegistry()
    registry.register(_fake_tool("a"))
    registry.register(_fake_tool("a"))
    assert len(registry) == 1


def test_tool_registry_unregister_removes_tool() -> None:
    """``unregister`` removes a tool by name."""
    registry = ToolRegistry()
    registry.register(_fake_tool("a"))
    registry.unregister("a")
    assert "a" not in registry


def test_tool_registry_unregister_unknown_noop() -> None:
    """Unregistering an absent tool is a no-op."""
    registry = ToolRegistry()
    registry.unregister("missing")  # should not raise


def test_tool_registry_get_unknown_raises() -> None:
    """``get`` raises :class:`ConfigurationError` for an unknown name."""
    registry = ToolRegistry()
    with pytest.raises(ConfigurationError, match="not registered"):
        registry.get("missing")


def test_tool_registry_try_get_returns_none_for_unknown() -> None:
    """``try_get`` returns ``None`` for an unknown name."""
    assert ToolRegistry().try_get("missing") is None


def test_tool_registry_names_preserves_insertion_order() -> None:
    """``names`` returns tool names in insertion order."""
    registry = ToolRegistry()
    registry.register(_fake_tool("first"))
    registry.register(_fake_tool("second"))
    registry.register(_fake_tool("third"))
    assert registry.names() == ["first", "second", "third"]


def test_tool_registry_schemas_returns_json_schemas() -> None:
    """``schemas`` returns the ``json_schema`` of every tool."""
    registry = ToolRegistry()
    registry.register(_fake_tool("a"))
    registry.register(_fake_tool("b"))
    assert registry.schemas() == [
        {"name": "a", "type": "object"},
        {"name": "b", "type": "object"},
    ]


def test_tool_registry_contains() -> None:
    """``in registry`` works for strings and rejects non-strings."""
    registry = ToolRegistry()
    registry.register(_fake_tool("a"))
    assert "a" in registry
    assert "missing" not in registry
    assert 42 not in registry


def test_tool_registry_len() -> None:
    """``len(registry)`` returns the number of tools."""
    registry = ToolRegistry()
    assert len(registry) == 0
    registry.register(_fake_tool("a"))
    registry.register(_fake_tool("b"))
    assert len(registry) == 2


# ---------------------------------------------------------------------------
# as_admin_user
# ---------------------------------------------------------------------------


def test_as_admin_user_returns_existing_user() -> None:
    """A non-None user is returned unchanged."""
    user = _user()
    assert as_admin_user(user) is user


def test_as_admin_user_synthesises_admin() -> None:
    """A None user becomes a synthetic admin principal."""
    admin = as_admin_user(None)
    assert admin.is_admin is True
    assert admin.email == "__agent__@local"
    assert admin.id == "__agent__"


# ---------------------------------------------------------------------------
# DateToday
# ---------------------------------------------------------------------------


def test_date_today_metadata() -> None:
    """``DateToday.execute`` returns today's date with structured data."""
    context = ToolContext(user=_user())
    result = asyncio.run(DateToday.execute(context))
    assert result.content  # ISO date string
    assert "iso" in result.data
    assert "year" in result.data
    assert "month" in result.data


def test_date_today_name_and_schema() -> None:
    """``DateToday`` exposes a stable name and JSON schema."""
    assert DateToday.name == "date_today"
    assert "type" in DateToday.json_schema


# ---------------------------------------------------------------------------
# KeywordSearch, GraphSearch, HybridSearch, WebSearch
# ---------------------------------------------------------------------------


def test_keyword_search_uses_dependencies() -> None:
    """``KeywordSearch`` is constructed with a vector store."""
    tool = KeywordSearch(vector_store=MagicMock())
    assert tool is not None


def test_graph_search_delegates_to_graph_index() -> None:
    """``GraphSearch.execute`` delegates to the graph index."""

    class _Hit:
        chunk = MagicMock(text="graph answer", id="h1", metadata={})

    class _GraphIndex:
        def search_local(self, question: str, top_k: int = 0) -> list[Any]:
            return [_Hit()]

    graph_index = _GraphIndex()
    tool = GraphSearch(graph_index=graph_index)
    context = ToolContext(user=_user())
    result = asyncio.run(tool.execute(context, query="q", mode="local", top_k=3))
    assert "graph answer" in result.content


def test_graph_search_handles_empty_results() -> None:
    """An empty graph result yields a no-match message."""

    class _EmptyGraph:
        def search_local(self, question: str, top_k: int = 0) -> list[Any]:
            return []

    tool = GraphSearch(graph_index=_EmptyGraph())
    context = ToolContext(user=_user())
    result = asyncio.run(tool.execute(context, query="q", mode="local"))
    assert "no" in result.content.lower()


def test_hybrid_search_executes_via_retrieval_pipeline() -> None:
    """``HybridSearch.execute`` delegates to the retrieval pipeline."""

    class _Hit:
        chunk = MagicMock(text="hybrid answer", id="h1", document_id="d1", metadata={})

    class _Pipeline:
        def retrieve(self, user: Any, question: str, top_k: int) -> list[Any]:
            return [_Hit()]

    pipeline = _Pipeline()
    vector_store = MagicMock()
    vector_store.keyword_search = MagicMock(return_value=[])
    tool = HybridSearch(retrieval_pipeline=pipeline, vector_store=vector_store)
    context = ToolContext(user=_user())
    result = asyncio.run(tool.execute(context, query="q", top_k=3))
    assert "hybrid answer" in result.content


def test_web_search_uses_search_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WebSearch.execute`` with a no-result search yields the empty message."""
    tool = WebSearch(max_results=3)
    context = ToolContext(user=_user())

    class _EmptyDDGS:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def text(self, query: str, max_results: int = 5) -> list[Any]:
            return []

    monkeypatch.setattr("duckduckgo_search.DDGS", _EmptyDDGS)
    result = asyncio.run(tool.execute(context, query="nothing", max_results=3))
    assert "no web results" in result.content


def test_web_search_handles_empty_query() -> None:
    """An empty query yields an error result."""
    tool = WebSearch(max_results=3)
    context = ToolContext(user=_user())
    result = asyncio.run(tool.execute(context, query="", max_results=3))
    assert result.ok is False
    assert "empty" in (result.error or "")


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------


def test_tool_subclass_must_implement_execute() -> None:
    """A :class:`Tool` subclass without ``execute`` cannot be instantiated."""

    class _Incomplete(Tool):
        name = "incomplete"
        description = "test"
        json_schema: dict[str, Any] = {"type": "object"}

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]
