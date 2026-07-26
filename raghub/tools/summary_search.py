"""RAPTOR summary search tool (Phase 7).

Wraps a :class:`RaptorIndex` so the planner can ask for higher-level
summaries alongside the chunk-level retrieval.
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.tools.base import BaseTool, ToolContext, ToolResult


class SummarySearchTool(BaseTool):
    """Search the RAPTOR summary tree.

    Attributes:
        name: ``"summary_search"``.
    """

    name: ClassVar[str] = "summary_search"
    description: ClassVar[str] = (
        "Search the recursive summary tree produced at ingest time. "
        "Useful for high-level questions about themes or summaries "
        "across many chunks."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, raptor_index: Any) -> None:
        """Initialise the tool.

        Args:
            raptor_index: A :class:`raghub.knowledge.structures.raptor.RaptorIndex`.
                ``None`` is accepted; the tool becomes a no-op that
                returns ``"(no summary index)"`` so the agent can
                tolerate missing knowledge structures.
        """
        self.index = raptor_index

    async def execute(
        self, context: ToolContext, *, query: str, top_k: int = 5, **_: Any
    ) -> ToolResult:
        """Run the RAPTOR summary search.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context (unused).
            query: The query string.
            top_k: Maximum hits.
        """
        if self.index is None:
            return ToolResult(content="(no summary index configured)")
        hits = self.index.search(query, top_k=int(top_k))
        if not hits:
            return ToolResult(content="(no summaries matched)")
        joined = "\n\n---\n\n".join(h.chunk.text for h in hits if h.chunk.text)
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "level": getattr(h.chunk, "metadata", {}).get("raptor_level", 0),
                        "text": h.chunk.text,
                    }
                    for h in hits
                ]
            },
        )


__all__ = ["SummarySearchTool"]