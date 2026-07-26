"""Keyword (BM25) search tool (Phase 7.4).

Uses the vector store's native keyword scorer (BM25 when the
optional ``rank_bm25`` is installed; TF fallback otherwise).
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.agent.tools.base import BaseTool, ToolContext, ToolResult


class KeywordSearchTool(BaseTool):
    """BM25 keyword search.

    Attributes:
        name: ``"keyword_search"``.
    """

    name: ClassVar[str] = "keyword_search"
    description: ClassVar[str] = (
        "Search the keyword (BM25) index for chunks containing the "
        "query terms. Useful for exact-token matches the vector "
        "search misses."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, vector_store: Any) -> None:
        """Initialise the tool.

        Args:
            vector_store: A :class:`raghub.vectorstore.base.BaseVectorStore`
                (must expose ``keyword_search``).
        """
        self.vector_store = vector_store

    async def execute(
        self, context: ToolContext, *, query: str, top_k: int = 5, **_: Any
    ) -> ToolResult:
        """Run the keyword search and return the joined hit list.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context (unused — keywords are
                query-driven, not user-driven).
            query: The query string.
            top_k: Maximum number of hits to return.
        """
        text = (query or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="keyword_search: empty query")
        keyword_search = getattr(self.vector_store, "keyword_search", None)
        if not callable(keyword_search):
            return ToolResult(
                ok=False,
                error="keyword_search: vector store lacks keyword_search()",
            )
        try:
            raw = keyword_search(text, int(top_k))
        except Exception as exc:
            return ToolResult(ok=False, error=f"keyword_search failed: {exc}")
        if not raw:
            return ToolResult(content="(no hits)")
        joined = "\n\n---\n\n".join(
            item["chunk"].text for item in raw if getattr(item.get("chunk"), "text", None)
        )
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": item["chunk"].chunk_id,
                        "document_id": item["chunk"].document_id,
                        "score": float(item["score"]),
                        "text": item["chunk"].text,
                        "metadata": item["chunk"].metadata,
                    }
                    for item in raw
                ]
            },
        )


__all__ = ["KeywordSearchTool"]