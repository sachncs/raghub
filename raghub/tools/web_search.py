"""Web search tool (Phase 7.6).

DuckDuckGo-backed search via the optional ``duckduckgo-search``
package. Imported lazily so the rest of the package stays usable
when DuckDuckGo is not installed.
"""

from __future__ import annotations

from typing import Any, ClassVar

from raghub.exceptions import WebSearchError
from raghub.tools.base import BaseTool, ToolContext, ToolResult


class WebSearchTool(BaseTool):
    """DuckDuckGo web search.

    Attributes:
        name: ``"web_search"``.
    """

    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the public web via DuckDuckGo. Use when the corpus "
        "doesn't contain the answer and fresh information may help."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, *, max_results: int = 5) -> None:
        """Initialise the tool.

        Args:
            max_results: Default result count. Overridable per call.
        """
        self.default_max = int(max_results)

    async def execute(
        self,
        context: ToolContext,
        **kwargs: Any,
    ) -> ToolResult:
        """Run a DuckDuckGo web search.

        See :meth:`BaseTool.execute` for the argument contract.

        Args:
            context: Per-invocation context (unused).
            query: The search query.
            max_results: Per-call override of the configured default.
        """
        text = (str(kwargs.get("query", "")) or "").strip()
        if not text:
            return ToolResult(ok=False, error="web_search: empty query")
        from duckduckgo_search import DDGS

        max_results = kwargs.get("max_results")
        n = int(max_results) if isinstance(max_results, int) else self.default_max
        n = max(1, min(n, 25))
        with DDGS() as ddgs:
            results = list(ddgs.text(text, max_results=n))
        if not results:
            return ToolResult(content="(no web results)")
        lines: list[str] = []
        payload: list[dict[str, Any]] = []
        for idx, result in enumerate(results, start=1):
            title = result.get("title") or ""
            body = result.get("body") or ""
            href = result.get("href") or ""
            lines.append(f"[{idx}] {title}\n{body}\n{href}")
            payload.append({"title": title, "body": body, "href": href})
        return ToolResult(
            content="\n\n".join(lines),
            data={"results": payload},
            source_url=payload[0]["href"] if payload else None,
        )


