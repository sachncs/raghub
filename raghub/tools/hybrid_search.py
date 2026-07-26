"""Hybrid (dense + sparse, RRF-fused) search tool (Phase 7.5).

Combines :class:`VectorSearchTool` and :class:`KeywordSearchTool`
inside one tool so the planner gets one fused result per call.
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.retrieval.fusion import rrf
from raghub.tools.base import BaseTool, ToolContext, ToolResult
from raghub.tools.vector_search import as_admin_user


class HybridSearchTool(BaseTool):
    """Dense + sparse fused retrieval.

    Attributes:
        name: ``"hybrid_search"``.
    """

    name: ClassVar[str] = "hybrid_search"
    description: ClassVar[str] = (
        "Combine vector and keyword retrieval with reciprocal rank "
        "fusion. Returns hits from both channels, fused."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
            "rrf_k": {"type": "integer", "default": 60, "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retrieval_pipeline: Any, vector_store: Any) -> None:
        """Initialise the tool.

        Args:
            retrieval_pipeline: A :class:`RetrievalPipeline` (dense).
            vector_store: A vector store with ``keyword_search``.
        """
        self.pipeline = retrieval_pipeline
        self.vector_store = vector_store

    async def execute(
        self,
        context: ToolContext,
        *,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        **_: Any,
    ) -> ToolResult:
        """Fuse dense + sparse retrieval with reciprocal-rank fusion.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context. ``context.user`` drives RBAC.
            query: The query string.
            top_k: Maximum hits per channel.
            rrf_k: RRF damping constant.
        """
        text = (query or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="hybrid_search: empty query")
        dense = self.pipeline.retrieve(
            user=as_admin_user(context.user),
            question=text,
            top_k=int(top_k),
        )
        sparse_raw: list[dict[str, Any]] = []
        keyword_search = getattr(self.vector_store, "keyword_search", None)
        if callable(keyword_search):
            sparse_raw = keyword_search(text, int(top_k) * 2)  
        fused = rrf(
            [
                [h.chunk.chunk_id for h in dense],
                [item["chunk"].chunk_id for item in sparse_raw],
            ],
            k=int(rrf_k),
        )
        id_to_hit: dict[str, Any] = {h.chunk.chunk_id: h for h in dense}
        for item in sparse_raw:
            cid = item["chunk"].chunk_id
            if cid not in id_to_hit:
                id_to_hit[cid] = item
        if not fused:
            return ToolResult(content="(no hits)")
        joined = "\n\n---\n\n".join(
            (getattr(id_to_hit[cid], "chunk", None) and id_to_hit[cid].chunk.text) or id_to_hit[cid]["chunk"].text
            for cid, _score in fused
        )
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": cid,
                        "document_id": getattr(
                            getattr(id_to_hit[cid], "chunk", None), "document_id", None
                        )
                        or id_to_hit[cid]["chunk"].document_id,
                        "score": float(score),
                        "text": (
                            id_to_hit[cid].chunk.text
                            if hasattr(id_to_hit[cid], "chunk")
                            else id_to_hit[cid]["chunk"].text
                        ),
                    }
                    for cid, score in fused
                ]
            },
        )


