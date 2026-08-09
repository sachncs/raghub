"""Query mixin for the RAG facade.

Holds the query entry points (:meth:`query`, :meth:`aquery`,
:meth:`astream`, :meth:`astream_agent`, :meth:`resolve_agent`,
:meth:`fallback_planner_events`), the conversation-scoped helpers
(:meth:`scoped`, :meth:`session_overrides`), and the evaluation
entry point (:meth:`evaluate`).

The mixin assumes the host class has already wired the
collaborators it needs:

- ``self.query_pipeline`` :class:`QueryPipeline` instance
- ``self.agentic_pipeline`` :class:`AgentPipeline` (or ``None``)
- ``self.conversation_store`` for per-session overrides
- ``self.llm`` for empty-question / empty-LLM guards
- ``self.settings`` for the :func:`resolve` defaults
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from raghub.agent import PlannerEvent, resolve
from raghub.config import Settings
from raghub.coroutines import await_if_awaitable
from raghub.errors import ConfigurationError, IngestionError, RagHubError
from raghub.eval import Finance
from raghub.models import (
    LLMProvider,
    PipelineCtx,
    RagQueryRequest,
    Response,
    Result,
)
from raghub.response import ResponseBuilder
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.conv import ConversationStore
    from raghub.pipeline import AgentPipeline, QueryPipeline


class QueryMixin:
    """Mixin providing query, streaming, agent, and evaluation entry points."""

    settings: Settings
    llm: LLMProvider | None
    query_pipeline: QueryPipeline
    agentic_pipeline: AgentPipeline | None
    conversation_store: ConversationStore

    def query(self, question: str, **kwargs: Any) -> Response:
        """Ask a question and return a typed :class:`Response`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Caller is already inside an event loop — return a coroutine
            # they can await themselves, or use :meth:`aquery` directly.
            return cast(
                Response,
                self.aquery(question, **kwargs),
            )
        return asyncio.run(self.aquery(question, **kwargs))

    @staticmethod
    def scoped(user: Any, session_id: str | None) -> str | None:
        """Combine ``user`` and ``session_id`` into a single opaque key.

        The conversation store is keyed by this combined value so two
        callers who happen to share or guess a ``session_id`` cannot
        read each other's history. When ``user`` is ``None`` the
        method returns the raw ``session_id`` (matching the prior
        for tests that exercise the in-process store anonymously).

        Args:
            user: The :class:`User` (or any duck-typed
                object with ``user_id`` / ``email`` attributes).
            session_id: The caller-supplied session id.

        Returns:
            The namespaced key, or ``None`` when no session id is set.

        """
        if session_id is None:
            return None
        if user is None:
            return session_id
        uid = getattr(user, "user_id", None) or getattr(user, "email", None) or "anonymous"
        return f"{uid}::{session_id}"

    def session_overrides(
        self, scoped: str | None, user: Any | None = None
    ) -> dict[str, Any] | None:
        """Return the session's tool/agent overrides (Phase 1.12).

        Args:
            scoped: The namespaced session id produced by
                :meth:`scoped`.
            user: Optional user principal. The conversation store is
                keyed by the scoped id, so the caller must pass the
                user that was used to build that scoped id.

        Returns:
            The overrides dict, or ``None`` when the session has none
            stored. Sessions without a key resolve to the global
            default in the resolver.

        """
        if scoped is None:
            return None
        get_overrides = getattr(self.conversation_store, "get_overrides", None)
        if not callable(get_overrides):
            return None
        return cast(dict[str, Any] | None, get_overrides(scoped))

    async def aquery(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: Any,
    ) -> Response:
        """Async version of :meth:`query`.

        All the ``agent / web / graph / reranker / ...`` kwargs are
        advanced-RAG flags (Phase 8.7). When any are set the facade
        resolves them against per-session overrides and per-user
        preferences via :func:`raghub.agent.resolve`. The resolved
        config is reflected in the returned :class:`Response`'s
        ``transforms_applied`` and ``metadata`` fields.

        Args:
            question: The user's question.
            request: Optional :class:`RagQueryRequest` that bundles
                the remaining advanced-RAG overrides.
            **kwargs: Convenience overrides accepted as keyword
                arguments (``user=``, ``session_id=``,
                ``tools_enabled=``, ``agent=``, ``web=``, ``graph=``,
                ``summaries=``, ``reranker=``, ``long_context_pass=``,
                ``query_transforms=``, ``max_steps=``, ``top_k=``,
                ``metadata_filter=``, ``response_model=``).

        Returns:
            A typed :class:`Response`.

        Raises:
            IngestionError: When ``question`` is empty or whitespace-only.

        """
        merged = self._merge_query_kwargs(request, kwargs)
        self._validate_query_inputs(question, merged)
        scoped = self.scoped(merged.get("user"), merged.get("session_id"))
        context = PipelineCtx(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )
        resolved = self._resolve_query_flags(merged, scoped)
        context.metadata["resolved_config"] = resolved.to_dict()
        return await self._execute_query_pipeline(
            question=question,
            merged=merged,
            scoped=scoped,
            context=context,
            resolved=resolved,
        )

    def _merge_query_kwargs(
        self, request: RagQueryRequest | None, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge the optional ``RagQueryRequest`` and the loose kwargs into one dict."""
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        return merged

    def _validate_query_inputs(self, question: str, merged: dict[str, Any]) -> None:
        """Validate that ``question`` is non-empty and the LLM is configured."""
        if not question or not question.strip():
            raise IngestionError("query() requires a non-empty question")
        if self.llm is None:
            raise ConfigurationError(
                "No LLM API key configured; set RAG_LLM_API_KEY "
                "(or another provider key) before calling query()."
            )

    def _resolve_query_flags(
        self, merged: dict[str, Any], scoped: str | None
    ) -> Any:
        """Resolve advanced-RAG flags via ``raghub.agent.resolve``."""
        user: Any | None = merged.get("user")
        return resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )

    async def _execute_query_pipeline(
        self,
        *,
        question: str,
        merged: dict[str, Any],
        scoped: str | None,
        context: PipelineCtx,
        resolved: Any,
    ) -> Response:
        """Run the query pipeline and translate the result into a Response."""
        user: Any | None = merged.get("user")
        top_k: int = merged.get("top_k", 5)
        metadata_filter: dict[str, Any] | None = merged.get("metadata_filter")
        response_model: type | None = merged.get("response_model")
        resolved_tools = set(resolved.tools_enabled) if resolved.tools_enabled else None

        result = await self.query_pipeline.run(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            response_model=response_model,
            user=user,
            session_id=scoped,
            tools_enabled=resolved_tools,
            resolved_config=resolved.to_dict(),
        )
        if getattr(result, "error", None) is not None:
            raise RagHubError(result.error or "query failed")
        return ResponseBuilder.from_pipeline(result)

    async def astream(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer token-by-token via the LLM's ``astream``.

        Accepts the same advanced-RAG flags as :meth:`aquery`; they
        are resolved through :func:`raghub.agent.resolve` and the
        resolved config is attached to the streaming span for
        observability.
        """
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        user: Any | None = merged.get("user")
        session_id: str | None = merged.get("session_id")
        top_k: int = merged.get("top_k", 5)
        metadata_filter: dict[str, Any] | None = merged.get("metadata_filter")
        scoped = self.scoped(user, session_id)
        context = PipelineCtx(
            pipeline_name="query",
            metadata={"session_id": scoped} if scoped else {},
        )

        resolved = resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )
        context.metadata["resolved_config"] = resolved.to_dict()
        async for piece in self.query_pipeline.stream(
            context,
            question=question,
            top_k=top_k,
            metadata_filter=metadata_filter or {},
            user=user,
            session_id=scoped,
            tools_enabled=(set(resolved.tools_enabled) if resolved.tools_enabled else None),
        ):
            yield piece

    async def astream_agent(
        self,
        question: str,
        *,
        request: RagQueryRequest | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream :class:`PlannerEvent` instances from the agent loop.

        Args:
            question: The user's question.
            request: Optional :class:`RagQueryRequest` that bundles
                the remaining advanced-RAG overrides.
            **kwargs: Convenience overrides accepted as keyword
                arguments (``user=``, ``session_id=``,
                ``tools_enabled=``, ``agent=``, ``web=``, ``graph=``,
                ``summaries=``, ``reranker=``, ``long_context_pass=``,
                ``query_transforms=``, ``max_steps=``).

        Yields:
            :class:`PlannerEvent` instances. SSE encoding is the
            caller's responsibility — the FastAPI route uses
            :meth:`raghub.sse.Sse.format`.

        """
        merged: dict[str, Any] = dict(request) if request is not None else {}
        merged.update(kwargs)
        user: Any | None = merged.get("user")
        session_id: str | None = merged.get("session_id")
        scoped = self.scoped(user, session_id)
        resolved = self.resolve_agent(merged, scoped, user)
        if self.agentic_pipeline is None:
            async for event in self.fallback_planner_events(question, session_id, user):
                yield event
            return
        context = PipelineCtx(
            pipeline_name="query_agent",
            metadata={
                "session_id": scoped or "",
                "resolved_config": resolved.to_dict(),
            },
        )
        async for event in self.agentic_pipeline.astream(
            context,
            question=question,
            user=user,
            session_id=scoped,
            tools_enabled=(set(resolved.tools_enabled) if resolved.tools_enabled else None),
            history=[],
        ):
            yield event

    def resolve_agent(
        self,
        merged: dict[str, Any],
        scoped: str | None,
        user: Any,
    ) -> Any:
        """Resolve the effective advanced-RAG config for a request."""
        return resolve(
            request_overrides={
                "tools_enabled": merged.get("tools_enabled"),
                "agent": merged.get("agent"),
                "web": merged.get("web"),
                "graph": merged.get("graph"),
                "summaries": merged.get("summaries"),
                "reranker": merged.get("reranker"),
                "long_context_pass": merged.get("long_context_pass"),
                "query_transforms": merged.get("query_transforms"),
                "max_steps": merged.get("max_steps"),
            },
            session_overrides=self.session_overrides(scoped, user),
            user_prefs=getattr(user, "tool_settings", None) if user else None,
            settings=self.settings,
        )

    async def fallback_planner_events(
        self,
        question: str,
        session_id: str | None,
        user: Any | None = None,
    ) -> AsyncIterator[Any]:
        """Yield planner events from the non-agentic path.

        When the agentic pipeline is not configured the facade
        wraps each token of the streaming answer as a planner
        event so SSE consumers see a uniform stream.
        """
        async for piece in self.astream(
            question,
            user=user,
            session_id=session_id,
            top_k=5,
            metadata_filter=None,
        ):
            yield PlannerEvent(
                kind="answer_chunk",
                step=0,
                payload={"text": piece},
            )

    def evaluate(
        self,
        benchmark: str = "financebench",
        *,
        response_factory: Callable[[dict[str, Any]], Any] | None = None,
        examples: Sequence[dict[str, Any]] | None = None,
        evaluator: Any = None,
    ) -> list[Result]:
        """Run a benchmark evaluation.

        Args:
            benchmark: ``"financebench"`` is the only supported name today.
            response_factory: Optional callable producing an answer per
                example (sync or async); when ``None`` the facade calls
                :meth:`aquery`.
            examples: The example list to evaluate.
            evaluator: Optional pre-built evaluator instance. When
                supplied the method skips the default ``Finance()``
                construction, which is what tests use to inject fakes
                without touching module globals.

        Returns:
            The list of evaluation :class:`Result` records.

        """
        if benchmark != "financebench":
            raise ConfigurationError(f"Unknown benchmark: {benchmark!r}")

        if evaluator is None:
            evaluator = Finance()
        factory = response_factory

        async def coerce_answer(example: dict[str, Any]) -> Any:
            """Coerce the result of ``response_factory`` to a coroutine."""
            if factory is None:
                return await self.aquery(example.get("question", ""))
            result = factory(example)
            if inspect.isawaitable(result):
                return await result
            return result

        return asyncio.run(self._arun_evaluation(
            evaluator=evaluator,
            examples=examples,
            response_factory=coerce_answer,
        ))

    async def _arun_evaluation(
        self,
        *,
        evaluator: Any,
        examples: Sequence[dict[str, Any]] | None,
        response_factory: Callable[[dict[str, Any]], Any],
    ) -> list[Result]:
        """Run an evaluator with a normalised ``response_factory``.

        Kept as a helper so :meth:`evaluate` can drive it via
        :func:`asyncio.run` while callers already in an event loop can
        await it directly.
        """
        examples_list = list(examples or [])
        return await evaluator.evaluate(examples_list, response_factory=response_factory)
