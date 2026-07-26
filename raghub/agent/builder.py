"""Tool-registry factory (Phase 7.8).

The :class:`RAG` facade calls this once during construction. The
returned :class:`ToolRegistry` is the planner's tool catalog; the
planner then filters by name according to the resolved config.
"""

from __future__ import annotations

from typing import Any

from raghub.agent.tools.date_today import DateTodayTool
from raghub.agent.tools.graph_search import GraphSearchTool
from raghub.agent.tools.hybrid_search import HybridSearchTool
from raghub.agent.tools.keyword_search import KeywordSearchTool
from raghub.agent.tools.registry import ToolRegistry
from raghub.agent.tools.summary_search import SummarySearchTool
from raghub.agent.tools.vector_search import VectorSearchTool
from raghub.agent.tools.web_search import WebSearchTool
from raghub.config import Settings


def build_tool_registry(
    settings: Settings,
    *,
    retrieval_pipeline: Any,
    vector_store: Any,
    raptor: Any | None = None,
    graph: Any | None = None,
) -> ToolRegistry:
    """Build the planner's tool catalog.

    Args:
        settings: Application settings. The ``web_search.enabled``,
            ``summary_search_enabled``, and ``graph_search_enabled``
            flags control which optional tools are registered.
        retrieval_pipeline: A :class:`RetrievalPipeline` for the
            dense / hybrid tools.
        vector_store: A vector store backing the keyword tool.
        raptor: Optional :class:`RaptorIndex`.
        graph: Optional :class:`GraphRagIndex`.

    Returns:
        A populated :class:`ToolRegistry`.
    """
    registry = ToolRegistry()
    registry.register(VectorSearchTool(retrieval_pipeline))
    registry.register(KeywordSearchTool(vector_store))
    registry.register(HybridSearchTool(retrieval_pipeline, vector_store))
    registry.register(DateTodayTool())
    if settings.web_search.enabled:
        registry.register(WebSearchTool(max_results=settings.web_search.max_results))
    if settings.summary_search_enabled and raptor is not None:
        registry.register(SummarySearchTool(raptor))
    if settings.graph_search_enabled and graph is not None:
        registry.register(GraphSearchTool(graph))
    return registry


__all__ = ["build_tool_registry"]