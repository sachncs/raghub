"""Preference routing for advanced-RAG flags.

Encapsulates :meth:`ApplicationFacade.query_with_flags` so the
facade's surface stays uniform with the rest of the auth/documents
delegators. The coordinator is responsible for:

1. Resolving the user's ``tool_settings`` preferences against the
   per-request overrides.
2. Routing the request through the wired :class:`raghub.RAG`
   instance when one is available (Phase 7+ agent / tool path).
3. Falling back to the legacy :class:`QueryService` when the
   :class:`raghub.RAG` facade is missing or fails to build.
"""

from __future__ import annotations

from typing import Any

from raghub.agent.resolver import resolve
from raghub.models import QueryResponse, UserPrincipal


class PreferenceCoordinator:
    """Routes advanced-RAG requests based on resolved user prefs.

    Attributes:
        facade: The owning :class:`ApplicationFacade`.
    """

    def __init__(self, facade: Any) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def query_with_flags(
        self,
        *,
        token: str,
        question: str,
        tools_enabled: list[str] | None = None,
        agent: bool | None = None,
        web: bool | None = None,
        graph: bool | None = None,
        summaries: bool | None = None,
        reranker: str | None = None,
        long_context_pass: bool | None = None,
        query_transforms: list[str] | None = None,
        max_steps: int | None = None,
        top_k: int | None = None,
    ) -> QueryResponse:
        """Resolve advanced-RAG flags against user prefs and route accordingly.

        Args:
            token: Bearer token.
            question: The user's question.
            tools_enabled: Per-request tool allow-list override.
            agent: Force the agent loop on (Phase 7).
            web: ``"web_search"`` shortcut.
            graph: ``"graph_search"`` shortcut.
            summaries: ``"summary_search"`` shortcut.
            reranker: Per-request reranker provider override.
            long_context_pass: Per-request long-context rerank toggle.
            query_transforms: Per-request list of transform names.
            max_steps: Per-request cap on planner steps.
            top_k: Per-request override of retrieval depth.

        Returns:
            A :class:`QueryResponse` carrying the resolved config in
            ``metadata``. When a :class:`raghub.RAG` instance is
            available (Phase 7+), the request is routed through
            :meth:`RAG.aquery` so the agent loop and tools actually
            run end-to-end. Otherwise the legacy ``QueryService`` is
            used and the response carries the resolved config for
            observability.
        """
        container = self.facade.container
        user, _ = await container.auth.resolve_user(token)
        prefs = dict(getattr(user, "tool_settings", None) or {})
        resolved = resolve(
            request_overrides={
                "tools_enabled": tools_enabled,
                "agent": agent,
                "web": web,
                "graph": graph,
                "summaries": summaries,
                "reranker": reranker,
                "long_context_pass": long_context_pass,
                "query_transforms": query_transforms,
                "max_steps": max_steps,
            },
            session_overrides=None,
            user_prefs=prefs,
            settings=container.settings,
        )

        rag: Any | None = getattr(container, "rag_facade", None)
        if rag is None:
            response = await self.facade.query_svc.query(token=token, question=question)
            response.metadata = dict(response.metadata or {})
            response.metadata["resolved_config"] = resolved.to_dict()
            if top_k is not None:
                response.metadata["requested_top_k"] = top_k
            return response

        session = await container.store.get_by_token(token)
        principal = UserPrincipal(
            user_id=user.user_id,
            email=user.email,
            allowed_companies=user.allowed_companies,
            allowed_groups=user.allowed_groups,
            is_admin=user.is_admin,
            tool_settings=user.tool_settings,
        )
        canonical = await rag.aquery(
            question,
            user=principal,
            session_id=session.session_id if session is not None else None,
            tools_enabled=tools_enabled,
            agent=agent,
            web=web,
            graph=graph,
            summaries=summaries,
            reranker=reranker,
            long_context_pass=long_context_pass,
            query_transforms=query_transforms,
            max_steps=max_steps,
            top_k=top_k,
        )
        return QueryResponse(
            answer=canonical.answer,
            citations=canonical.citations,
            source_chunks=[
                chunk.model_dump(mode="json")
                for chunk in canonical.source_chunks
            ],
            planner_trace=canonical.metadata.get("planner_trace"),
            tools_invoked=canonical.metadata.get("tools_invoked") or [],
            transforms_applied=canonical.transforms_applied,
            metadata={
                "pipeline_id": "query_agent"
                if (resolved.agent_enabled or resolved.tools_enabled)
                else "query",
                "structured": False,
                **canonical.metadata,
            },
        )


__all__ = ["PreferenceCoordinator"]