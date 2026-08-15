"""Preference router: advanced-RAG requests based on resolved user prefs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from raghub.agent import ResolvedConfig, resolve
from raghub.models import QueryResponse, User
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.rag.facade import RAG
    from raghub.services.container import RagContainer
    from raghub.services.facade import ApplicationFacade, Facade  # noqa: F401  - kept for backwards compatibility in tests


class Preference:
    """Routes advanced-RAG requests based on resolved user prefs."""

    def __init__(self, facade: ApplicationFacade) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def query_with_flags(
        self,
        *,
        token: str,
        question: str,
        **flags: JSONValue,
    ) -> QueryResponse:
        """Resolve advanced-RAG flags against user prefs and route accordingly."""
        container = self.facade.container
        user, _ = await container.auth.resolve_user(token)
        resolved = self.resolve_flags(user, flags, container)
        rag = getattr(container, "rag_facade", None)
        if rag is None:
            return await self.query_basic(
                token=token, question=question, flags=flags, resolved=resolved
            )
        return await self.query_advanced(
            token=token,
            question=question,
            flags=flags,
            resolved=resolved,
            user=user,
            rag=rag,
        )

    @staticmethod
    def resolve_flags(user: User, flags: dict[str, Any], container: RagContainer) -> ResolvedConfig:
        """Resolve per-request, session, and user-prefs into a single :class:`ResolvedConfig`."""
        prefs = dict(getattr(user, "tool_settings", None) or {})
        return resolve(
            request_overrides={
                "tools_enabled": flags.get("tools_enabled"),
                "agent": flags.get("agent"),
                "web": flags.get("web"),
                "graph": flags.get("graph"),
                "summaries": flags.get("summaries"),
                "reranker": flags.get("reranker"),
                "long_context_pass": flags.get("long_context_pass"),
                "query_transforms": flags.get("query_transforms"),
                "max_steps": flags.get("max_steps"),
            },
            session_overrides=None,
            user_prefs=prefs,
            settings=container.settings,
        )

    async def query_basic(
        self,
        *,
        token: str,
        question: str,
        flags: dict[str, Any],
        resolved: ResolvedConfig,
    ) -> QueryResponse:
        """Run a basic RAG query without the agent loop, attaching resolved config metadata."""
        response = cast(QueryResponse, await self.facade.query(token=token, question=question))
        metadata = dict(response.metadata or {})
        metadata["resolved_config"] = resolved.to_dict()
        if flags.get("top_k") is not None:
            metadata["requested_top_k"] = flags["top_k"]
        return cast(QueryResponse, response.copy(metadata=metadata))

    async def query_advanced(  # noqa: PLR0913 - one facade method fanning out query inputs
        self,
        *,
        token: str,
        question: str,
        flags: dict[str, Any],
        resolved: ResolvedConfig,
        user: User,
        rag: RAG,
    ) -> QueryResponse:
        """Run the full agent loop with the resolved RAG instance.

        Returns a structured response with the resolved config attached.
        """
        container = self.facade.container
        session = await container.store.get_by_token(token)
        principal = User(
            id=user.id,
            email=user.email,
            allowed_companies=user.allowed_companies,
            allowed_groups=user.allowed_groups,
            is_admin=user.is_admin,
            tool_settings=user.tool_settings,
        )
        canonical = await rag.aquery(
            question,
            user=principal,
            session_id=session.id if session is not None else None,
            **flags,
        )
        return QueryResponse(
            answer=canonical.answer,
            citations=[c.dump(mode="json") for c in canonical.citations],
            source_chunks=[chunk.dump(mode="json") for chunk in canonical.source_chunks],
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


__all__ = ["Preference"]
