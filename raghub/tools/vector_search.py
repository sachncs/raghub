"""Vector similarity search tool (Phase 7.3).

Wraps the configured :class:`RetrievalPipeline` so the planner can
ask for ``{"query": ..., "top_k": 10}`` and get back a
:class:`ToolResult` whose ``content`` is the joined chunk texts
and whose ``data["hits"]`` is the underlying :class:`RetrievalHit`
list (used by the agentic pipeline for citation enrichment).
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.models import UserPrincipal
from raghub.tools.base import BaseTool, ToolContext, ToolResult


def as_admin_user(user: UserPrincipal | None) -> UserPrincipal:
    """Return ``user`` or a synthetic admin principal.

    The retrieval pipeline requires a :class:`UserPrincipal`; the
    RBAC layer treats an admin as "see everything", which is the
    safe default for an unauthenticated tool call.

    Args:
        user: The active principal, or ``None`` for an anonymous
            tool invocation.

    Returns:
        ``user`` unchanged when provided, otherwise a synthetic
        principal with ``is_admin=True``.
    """
    if user is None:
        return UserPrincipal(user_id="__agent__", email="__agent__@local", is_admin=True)
    return user


class VectorSearchTool(BaseTool):
    """Top-K dense retrieval scoped to the user's RBAC filter.

    Attributes:
        name: ``"vector_search"``.
    """

    name: ClassVar[str] = "vector_search"
    description: ClassVar[str] = (
        "Search the vector store for chunks relevant to the query. "
        "Returns the top-k hits with text and metadata."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retrieval_pipeline: Any) -> None:
        """Initialise the tool.

        Args:
            retrieval_pipeline: A :class:`raghub.retrieval.pipeline.RetrievalPipeline`.
        """
        self.pipeline = retrieval_pipeline

    async def execute(
        self, context: ToolContext, **kwargs: Any
    ) -> ToolResult:
        """Run the dense vector search and return the joined hit list.

        Args:
            context: Per-invocation context. ``context.user`` drives RBAC.
            query: The search query. Defaults to ``context.question`` when ``None``.
            top_k: Maximum hits.
        """
        text = (str(kwargs.get("query", "")) or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="vector_search: empty query")
        user = as_admin_user(context.user)
        hits = self.pipeline.retrieve(
            user=user, question=text, top_k=int(kwargs.get("top_k", 0))
        )
        if not hits:
            return ToolResult(content="(no hits)")
        joined = "\n\n---\n\n".join(h.chunk.text for h in hits if h.chunk.text)
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "document_id": h.chunk.document_id,
                        "score": float(h.score),
                        "text": h.chunk.text,
                        "metadata": h.chunk.metadata,
                    }
                    for h in hits
                ]
            },
            source_url=None,
        )


