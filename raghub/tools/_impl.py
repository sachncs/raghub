"""Tool registry + concrete tool implementations.

Defines the tool contract and every concrete tool class in one helper file.
Class summary::

    ToolProtocol    - :class:`Protocol` describing the planner-facing contract.
    ToolResult      - structured Pydantic result returned by a tool.
    ToolContext     - per-invocation state (user, session, question).
    Tool            - reusable :class:`ABC` implementing :class:`ToolProtocol`.
    ToolRegistry    - name -> tool lookup.

Concrete tools (one class per format family, no ``Tool`` suffix):

    Today           - UTC date stub.
    GraphSearch     - GraphRAG local / global search.
    HybridSearch    - dense + sparse, RRF-fused.
    Keyword         - BM25 / TF keyword search.
    SummarySearch   - RAPTOR summary search.
    VectorSearch    - top-K dense retrieval.
    WebSearch       - DuckDuckGo web search.

Helpers::

    as_admin_user   - resolve :class:`User` to an admin
                      equivalent (synthetic when ``None``).
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from raghub.constants import RRF_K
from raghub.errors import ConfigurationError
from raghub.models import User
from raghub.store import Store
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.knowledge import GraphIndex, Raptor
    from raghub.models import VectorStore
    from raghub.retrieval import Retrieval

# ---------------------------------------------------------------------------
# Tool contract
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Structured result returned by a :class:`Tool`.

    Attributes:
        ok: ``True`` for a successful execution, ``False`` otherwise.
        content: Plain-text rendering of the result for the LLM.
        data: Structured payload (e.g. web search hits, citations).
        error: Error message when ``ok`` is ``False``; ``None`` on success.
        latency_ms: Wall-clock duration recorded by the tool itself.
        source_url: Optional URL the result came from (web search, API).

    """

    ok: bool = True
    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    source_url: str | None = None


@runtime_checkable
class ToolProtocol(Protocol):
    """Async tool the planner can invoke.

    Attributes:
        name: Stable tool name (e.g. ``"web_search"``). Must be unique
            within a :class:`ToolRegistry`.
        description: One-line description surfaced to the LLM.
        json_schema: JSON-Schema describing accepted ``args``. Used by
            the planner to prompt the model and to validate the parsed
            arguments before invocation.

    """

    name: ClassVar[str]
    description: ClassVar[str]
    json_schema: ClassVar[dict[str, Any]]

    async def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool.

        Args:
            args: Arguments parsed from the LLM's planner output.

        Returns:
            A :class:`ToolResult`. Never raise for expected failures —
            set ``ok=False`` and populate ``error`` so the planner can
            observe the failure and adapt.

        """
        ...


@dataclass
class ToolContext:
    """Per-invocation context passed to every tool.

    Attributes:
        user: The :class:`User` driving RBAC. ``None`` for
            unauthenticated callers.
        session_id: Scoped session id (``user::raw_session``).
        session_overrides: Session-scoped overrides copied from the
            conversation store.
        question: The user's literal question. Useful when a tool
            wants to embed it (e.g. vector_search).
        metadata: Free-form per-call state.

    """

    user: User | None = None
    session_id: str | None = None
    session_overrides: dict[str, Any] | None = None
    question: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ToolProtocol, ABC):
    """Reusable :class:`Tool` base class.

    Concrete tools override :meth:`execute` and declare the
    class-level :attr:`name`, :attr:`description`, and
    :attr:`json_schema`. The framework calls :meth:`run`, which:

    * Wraps every exception in a structured :class:`ToolResult`
      with ``ok=False``. Tools must never raise — the planner
      reacts to ``ok=False`` the same way it reacts to any other
      observation.
    * Times the call and writes ``latency_ms`` on the result.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    json_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Run the tool.

        Args:
            context: Per-invocation context.
            **kwargs: Validated arguments parsed from the LLM's
                planner output.

        """

    @staticmethod
    def context(**overrides: JSONValue) -> ToolContext:
        """Return a fresh :class:`ToolContext` for this tool.

        Use this when invoking the tool outside the agent loop. The
        returned context is admin-equivalent (synthetic admin
        principal) by default; pass ``user=`` to override and any
        other ToolContext field to override too.

        Args:
            **overrides: Field overrides for :class:`ToolContext`.

        Returns:
            A new :class:`ToolContext` with sensible defaults.

        """
        defaults: dict[str, Any] = {
            "user": as_admin_user(None),
            "session_id": None,
            "session_overrides": None,
            "question": "",
            "metadata": {},
        }
        defaults.update(overrides)
        return ToolContext(**defaults)

    async def run(
        self,
        args: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Public entry point used by the agent loop.

        Args:
            args: Argument dict parsed from the LLM output.
            context: Optional :class:`ToolContext`.

        Returns:
            A :class:`ToolResult`. ``latency_ms`` is filled in here.

        """
        ctx = context or self.context()
        started = time.perf_counter()
        result = await self.execute(ctx, **args)
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        return result

    def call(
        self,
        args: dict[str, Any],
        context: ToolContext | None = None,
    ) -> ToolResult:
        """Run the tool synchronously on a fresh event loop.

        Use this when invoking the tool from sync code; the agent
        loop calls :meth:`run` directly.

        Args:
            args: Tool argument dict.
            context: Optional :class:`ToolContext`. Defaults to
                :meth:`context`.

        Returns:
            A :class:`ToolResult`.

        """
        return asyncio.run(self.run(args, context))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Named container of :class:`Tool` instances.

    Lookup is case-sensitive. Re-registering a name overwrites the
    prior binding — the registry is intentionally permissive so
    tests can swap implementations without ceremony.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self.tools: dict[str, ToolProtocol] = {}

    def register(self, tool: ToolProtocol) -> None:
        """Add (or replace) ``tool`` under its :attr:`Tool.name`."""
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op when absent."""
        self.tools.pop(name, None)

    def get(self, name: str) -> ToolProtocol:
        """Return the tool registered under ``name``.

        Raises:
            ConfigurationError: When ``name`` is not registered.

        """
        if name not in self.tools:
            raise ConfigurationError(f"Tool {name!r} is not registered")
        return self.tools[name]

    def try_get(self, name: str) -> ToolProtocol | None:
        """Return the tool registered under ``name`` or ``None``."""
        return self.tools.get(name)

    def names(self) -> list[str]:
        """Return the list of registered tool names (insertion order)."""
        return list(self.tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """Return the JSON-Schema list for every registered tool.

        Used by the planner to render the tool catalog in the system
        prompt. Schemas are returned in insertion order so tests can
        rely on a stable ordering.
        """
        return [tool.json_schema for tool in self.tools.values()]

    def __contains__(self, name: object) -> bool:
        """Support ``"web_search" in registry``."""
        return isinstance(name, str) and name in self.tools

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self.tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def as_admin_user(user: User | None) -> User:
    """Return ``user`` or a synthetic admin principal.

    The retrieval pipeline requires a :class:`User`; the
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
        return User(id="__agent__", email="__agent__@local", is_admin=True)
    return user


# ---------------------------------------------------------------------------
# Concrete tools
# ---------------------------------------------------------------------------


class Today(Tool):
    """UTC date stub.

    Attributes:
        name: ``"date_today"``.

    """

    name: ClassVar[str] = "date_today"
    description: ClassVar[str] = (
        "Return today's date in UTC ISO 8601 format. Use when the question is time-sensitive."
    )
    json_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    @staticmethod
    async def execute(context: ToolContext, **_: Any) -> ToolResult:
        """Return today's date in UTC ISO 8601 format."""
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        return ToolResult(
            content=now.date().isoformat(),
            data={"iso": now.isoformat(), "year": now.year, "month": now.month},
        )


class GraphSearch(Tool):
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
            "mode": {
                "type": "string",
                "enum": ["local", "global"],
                "default": "local",
            },
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, graph_index: GraphIndex) -> None:
        """Initialise the tool.

        Args:
            graph_index: A GraphIndex (or ``None`` for a no-op tool).

        """
        self.index = graph_index

    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Run GraphRAG local or global search."""
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
                ok=False,
                error=f"graph_search: mode {mode!r} not supported by index",
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
                        "chunk_id": h.chunk.id,
                        "text": h.chunk.text,
                        "metadata": h.chunk.metadata,
                    }
                    for h in hits
                ]
            },
        )


class HybridSearch(Tool):
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
            "rrf_k": {"type": "integer", "default": RRF_K, "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retrieval_pipeline: Retrieval, vector_store: Store) -> None:
        """Initialise the tool."""
        self.pipeline = retrieval_pipeline
        self.vector_store = vector_store

    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Fuse dense + sparse retrieval with reciprocal-rank fusion."""
        text = (str(kwargs.get("query", "")) or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="hybrid_search: empty query")
        dense = self.pipeline.retrieve(
            user=as_admin_user(context.user),
            question=text,
            top_k=int(kwargs.get("top_k", 0)),
        )
        sparse_raw = self.fetch_sparse_results(text, kwargs)
        fused = self.fuse_results(dense, sparse_raw, kwargs)
        id_to_hit = self.merge_into_hit_map(dense, sparse_raw)
        if not fused:
            return ToolResult(content="(no hits)")
        return self.build_fused_tool_result(id_to_hit, fused)

    def fetch_sparse_results(
        self, text: str, kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Return keyword-search hits if the vector store supports it."""
        keyword_search = getattr(self.vector_store, "keyword_search", None)
        if not callable(keyword_search):
            return []
        return keyword_search(text, int(kwargs.get("top_k", 0)) * 2)

    @staticmethod
    def fuse_results(
        dense: list[Any],
        sparse_raw: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """Combine dense hits with sparse hits via reciprocal-rank fusion."""
        from raghub.retrieval import reciprocal_rank_fusion

        return reciprocal_rank_fusion(
            [
                [h.chunk.id for h in dense],
                [search_record["chunk"].chunk_id for search_record in sparse_raw],
            ],
            k=int(kwargs.get("rrf_k", RRF_K)) or RRF_K,
        )

    @staticmethod
    def merge_into_hit_map(
        dense: list[Any], sparse_raw: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build a chunk-id -> hit dict, deduplicating across dense/sparse."""
        id_to_hit: dict[str, Any] = {h.chunk.id: h for h in dense}
        for search_record in sparse_raw:
            cid = search_record["chunk"].chunk_id
            if cid not in id_to_hit:
                id_to_hit[cid] = search_record
        return id_to_hit

    @staticmethod
    def build_fused_tool_result(
        id_to_hit: dict[str, Any], fused: list[tuple[str, float]]
    ) -> ToolResult:
        """Build the ToolResult from a fused id-to-hit mapping and final ranking."""
        joined = "\n\n---\n\n".join(
            (getattr(id_to_hit[cid], "chunk", None) and id_to_hit[cid].chunk.text)
            or id_to_hit[cid]["chunk"].text
            for cid, _score in fused
        )
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": cid,
                        "document_id": (
                            getattr(getattr(id_to_hit[cid], "chunk", None), "document_id", None)
                            or id_to_hit[cid]["chunk"].document_id
                        ),
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


class Keyword(Tool):
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

    def __init__(self, vector_store: VectorStore) -> None:
        """Initialise the tool.

        Args:
            vector_store: A :class:`VectorStore` exposing ``keyword_search``.

        """
        self.vector_store = vector_store

    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Run the keyword search and return the joined hit list."""
        text = (str(kwargs.get("query", "")) or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="keyword_search: empty query")
        keyword_search = getattr(self.vector_store, "keyword_search", None)
        if not callable(keyword_search):
            return ToolResult(
                ok=False,
                error="keyword_search: vector store lacks keyword_search()",
            )
        raw = keyword_search(text, int(kwargs.get("top_k", 0)))
        if not raw:
            return ToolResult(content="(no hits)")
        joined = "\n\n---\n\n".join(
            search_record["chunk"].text
            for search_record in raw
            if getattr(search_record.get("chunk"), "text", None)
        )
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": search_record["chunk"].id,
                        "document_id": search_record["chunk"].document_id,
                        "score": float(search_record["score"]),
                        "text": search_record["chunk"].text,
                        "metadata": search_record["chunk"].metadata,
                    }
                    for search_record in raw
                ]
            },
        )


class SummarySearch(Tool):
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

    def __init__(self, raptor_index: Raptor) -> None:
        """Initialise the tool.

        Args:
            raptor_index: A Raptor (or ``None`` for a no-op tool).

        """
        self.index = raptor_index

    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Run the RAPTOR summary search."""
        if self.index is None:
            return ToolResult(content="(no summary index configured)")
        hits = self.index.search(str(kwargs.get("query", "")), top_k=int(kwargs.get("top_k", 0)))
        if not hits:
            return ToolResult(content="(no summaries matched)")
        joined = "\n\n---\n\n".join(h.chunk.text for h in hits if h.chunk.text)
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": h.chunk.id,
                        "level": getattr(h.chunk, "metadata", {}).get("raptor_level", 0),
                        "text": h.chunk.text,
                    }
                    for h in hits
                ]
            },
        )


class VectorSearch(Tool):
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

    def __init__(self, retrieval_pipeline: Retrieval) -> None:
        """Initialise the tool.

        Args:
            retrieval_pipeline: A :class:`raghub.retrieval.Retrieval`.

        """
        self.pipeline = retrieval_pipeline

    async def execute(self, context: ToolContext, **kwargs: JSONValue) -> ToolResult:
        """Run the dense vector search and return the joined hit list."""
        text = (str(kwargs.get("query", "")) or context.question or "").strip()
        if not text:
            return ToolResult(ok=False, error="vector_search: empty query")
        user = as_admin_user(context.user)
        hits = self.pipeline.retrieve(user=user, question=text, top_k=int(kwargs.get("top_k", 0)))
        if not hits:
            return ToolResult(content="(no hits)")
        joined = "\n\n---\n\n".join(h.chunk.text for h in hits if h.chunk.text)
        return ToolResult(
            content=joined,
            data={
                "hits": [
                    {
                        "chunk_id": h.chunk.id,
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


class WebSearch(Tool):
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
        **kwargs: JSONValue,
    ) -> ToolResult:
        """Run a DuckDuckGo web search."""
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


__all__ = [
    "GraphSearch",
    "HybridSearch",
    "Keyword",
    "SummarySearch",
    "Today",
    "Tool",
    "ToolContext",
    "ToolProtocol",
    "ToolRegistry",
    "ToolResult",
    "VectorSearch",
    "WebSearch",
    "as_admin_user",
]
