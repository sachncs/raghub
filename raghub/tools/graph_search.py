"""GraphRAG search tool (Phase 7).

Wraps a :class:`GraphRagIndex` so the planner can ask for
entity-anchored or community-summary retrieval.
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.tools.base import BaseTool, ToolContext, ToolResult


class GraphSearchTool(BaseTool):
    """GraphRAG local + global search.

    Attributes:
        name: ``"graph_search"``.
    """

    name: ClassVar[str] = "graph_search"
    description: ClassVar[str] = (
        "Search the entity / community graph built at ingest time. "
        "Use ``mode=local`` for entity-expanded retrieval and "
        "``mode=global`` for community-summary retrieval."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["local", "global"], "default": "local"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, graph_index: Any) -> None:
        """Initialise the tool.

        Args:
            graph_index: A :class:`raghub.knowledge.structures.graphrag.GraphRagIndex`.
                ``None`` is accepted; the tool reports ``"(no graph index)"``.
        """
        self.index = graph_index

    async def execute(
        self, context: ToolContext, **kwargs: Any
    ) -> ToolResult:
        """Run GraphRAG local or global search.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context (unused).
            query: The query string.
            mode: ``"local"`` for entity-anchored search,
                ``"global"`` for community-summary search.
            top_k: Maximum hits.
        """
        if self.index is None:
            return ToolResult(content="(no graph index configured)")
        mode = str(kwargs.get("mode", "local"))
        query = str(kwargs.get("query", ""))
        if mode == "local":
            fn = getattr(self.index, "search_local", None)
        elif mode == "global":
            fn = getattr(self.index, "search_global", None)
        else:
            return ToolResult(ok=False, error=f"graph_search: unknown mode {mode!r}")
        if not callable(fn):
            return ToolResult(
                ok=False, error=f"graph_search: mode {mode!r} not supported by index"
            )
        hits = fn(query, top_k=int(kwargs.get("top_k", 0)))
        if not hits:
            return ToolResult(content="(no graph matches)")
        joined = "\n\n---\n\n".join(h.chunk.text for h in hits if h.chunk.text)
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "text": h.chunk.text,
                        "metadata": h.chunk.metadata,
                    }
                    for h in hits
                ]
            },
        )


