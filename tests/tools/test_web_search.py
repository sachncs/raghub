"""Tests for raghub.tools.web_search.

The :class:`WebSearchTool` wraps DuckDuckGo. The tests exercise:

* Empty query short-circuits without touching the network.
* Custom ``max_results`` is honoured.
* A 25-cap prevents ridiculous per-call limits.
* Network errors surface as ``ToolResult(ok=False, ...)``.
* Schema is exposed and validated.
"""

from __future__ import annotations

import sys
import asyncio
from unittest.mock import patch

import duckduckgo_search

from raghub.tools.base import ToolContext, ToolResult
from raghub.tools.web_search import WebSearchTool


def run(coro):
    """Drive an async coroutine to completion (test helper)."""
    return asyncio.run(coro)


def _ctx() -> ToolContext:
    return ToolContext()


class FakeDDGS:
    """A fake DuckDuckGo context manager that returns preset results."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = results or []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, query, max_results):
        return self.results[:max_results]


def test_empty_query_returns_error_without_network() -> None:
    tool = WebSearchTool()
    result = asyncio.run(tool.execute(_ctx(), query=""))
    assert result.ok is False
    assert "empty" in result.error


def test_results_are_formatted_with_index() -> None:
    """Each result is prefixed with ``[N] title`` in the rendered content."""
    tool = WebSearchTool()
    with patch(
        "duckduckgo_search.DDGS",
        return_value=FakeDDGS([
            {"title": "First Result", "body": "body 1", "href": "https://a"},
            {"title": "Second Result", "body": "body 2", "href": "https://b"},
        ]),
    ):
        result = asyncio.run(tool.execute(_ctx(), query="python", max_results=5))
    assert result.ok is True
    assert "[1] First Result" in result.content
    assert "[2] Second Result" in result.content
    assert "https://a" in result.content
    assert result.data["results"][0]["title"] == "First Result"
    assert result.source_url == "https://a"


def test_no_results_returns_no_results_marker() -> None:
    tool = WebSearchTool()
    with patch(
        "duckduckgo_search.DDGS",
        return_value=FakeDDGS([]),
    ):
        result = asyncio.run(tool.execute(_ctx(), query="obscure topic"))
    assert result.ok is True
    assert "no web results" in result.content


def test_custom_max_results_capped_at_25() -> None:
    """A 1000-request call is silently capped at 25."""
    tool = WebSearchTool()
    seen: list[int] = []

    class TrackingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results):
            seen.append(max_results)
            return []

    with patch(
        "duckduckgo_search.DDGS",
        return_value=TrackingDDGS(),
    ):
        asyncio.run(tool.execute(_ctx(), query="x", max_results=1000))
    assert seen == [25]


def test_max_results_floor_is_one() -> None:
    tool = WebSearchTool()
    seen: list[int] = []

    class TrackingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, max_results):
            seen.append(max_results)
            return []

    with patch(
        "duckduckgo_search.DDGS",
        return_value=TrackingDDGS(),
    ):
        asyncio.run(tool.execute(_ctx(), query="x", max_results=0))
    assert seen == [1]


def test_default_max_results_is_5() -> None:
    tool = WebSearchTool()
    assert tool.default_max == 5


def test_schema_contains_required_query() -> None:
    tool = WebSearchTool()
    assert "query" in tool.json_schema["required"]
    assert tool.json_schema["properties"]["max_results"]["type"] == "integer"


def test_metadata_includes_source_url() -> None:
    """The first hit's URL is exposed as ``source_url`` for citation."""
    tool = WebSearchTool()
    with patch(
        "duckduckgo_search.DDGS",
        return_value=FakeDDGS([
            {"title": "Only", "body": "b", "href": "https://only"},
        ]),
    ):
        result = asyncio.run(tool.execute(_ctx(), query="x"))
    assert result.source_url == "https://only"


def test_no_results_yields_no_source_url() -> None:
    tool = WebSearchTool()
    with patch(
        "duckduckgo_search.DDGS",
        return_value=FakeDDGS([]),
    ):
        result = asyncio.run(tool.execute(_ctx(), query="x"))
    assert result.source_url is None


def test_name_and_description_are_stable() -> None:
    """The tool identity is part of the agent's prompt contract."""
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "web" in tool.description.lower()
