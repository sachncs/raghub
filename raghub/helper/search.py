"""Module-style dispatch for every tool.

Each module-level function builds a configured :class:`Tool`, asks
it for a fresh :class:`ToolContext` (admin-equivalent unless a
``user=`` override is supplied), and synchronously :meth:`Tool.call`
once. The concrete :class:`Tool` subclasses remain reachable from
:mod:`raghub.tools` for :class:`ToolRegistry`-based agent registration.

Public functions::

    search.date.today()
    search.vector(pipeline, query="...", top_k=5)
    search.keyword(vector_store, query="...", top_k=5)
    search.graph(graph_index, query="...", mode="local", top_k=5)
    search.summary(raptor_index, query="...", top_k=5)
    search.web(query="...", max_results=5)
    search.hybrid(pipeline, vector_store, query="...", top_k=5, rrf_k=60)
"""

from __future__ import annotations

from typing import Any

from raghub.tools import (
    DateToday,
    GraphSearch,
    HybridSearch,
    KeywordSearch,
    SummarySearch,
    ToolResult,
    VectorSearch,
    WebSearch,
)


def date() -> ToolResult:
    """Return today's UTC date as a :class:`ToolResult`."""
    tool = DateToday()
    return tool.call({}, tool.context())


def vector(
    retrieval_pipeline: Any,
    *,
    query: str = "",
    top_k: int = 5,
) -> ToolResult:
    """Dense vector retrieval."""
    tool = VectorSearch(retrieval_pipeline)
    ctx = tool.context(question=query)
    return tool.call({"query": query, "top_k": top_k}, ctx)


def keyword(
    vector_store: Any,
    *,
    query: str = "",
    top_k: int = 5,
) -> ToolResult:
    """BM25 / TF keyword retrieval."""
    tool = KeywordSearch(vector_store)
    ctx = tool.context(question=query)
    return tool.call({"query": query, "top_k": top_k}, ctx)


def graph(
    graph_index: Any,
    *,
    query: str = "",
    mode: str = "local",
    top_k: int = 5,
) -> ToolResult:
    """GraphRAG local or global search."""
    tool = GraphSearch(graph_index)
    ctx = tool.context(question=query)
    return tool.call({"query": query, "mode": mode, "top_k": top_k}, ctx)


def summary(
    raptor_index: Any,
    *,
    query: str = "",
    top_k: int = 5,
) -> ToolResult:
    """RAPTOR summary-tree search."""
    tool = SummarySearch(raptor_index)
    ctx = tool.context(question=query)
    return tool.call({"query": query, "top_k": top_k}, ctx)


def web(query: str = "", *, max_results: int = 5) -> ToolResult:
    """DuckDuckGo-backed web search."""
    tool = WebSearch(max_results=max_results)
    return tool.call({"query": query, "max_results": max_results})


def hybrid(
    retrieval_pipeline: Any,
    vector_store: Any,
    *,
    query: str = "",
    top_k: int = 5,
    rrf_k: int = 60,
) -> ToolResult:
    """Dense + sparse retrieval fused with reciprocal-rank fusion."""
    tool = HybridSearch(retrieval_pipeline, vector_store)
    ctx = tool.context(question=query)
    return tool.call(
        {"query": query, "top_k": top_k, "rrf_k": rrf_k}, ctx
    )


__all__ = [
    "date",
    "graph",
    "hybrid",
    "keyword",
    "summary",
    "vector",
    "web",
]
