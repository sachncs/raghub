"""High-level application facade.

:mod:`raghub.services.application.facade` exposes every public
action the API and CLI call. Three thin coordinators carry the
specifics:

* :class:`AuthCoordinator` — login / logout / principal resolution.
* :class:`PreferenceCoordinator` — advanced-RAG flag resolution +
  raghub.RAG dispatch.
* :class:`ShutdownCoordinator` — collaborator teardown order.

The legacy alias ``DynamicRagApplication`` points at
:class:`ApplicationFacade`; it is re-exported from
:mod:`raghub.services.application.__init__`.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any, cast

from raghub.agent import resolve
from raghub.auth import AuthService
from raghub.models import (
    AuthLoginResponse,
    ConversationTurn,
    DocumentRecord,
    QueryResponse,
    UserPrincipal,
)
from raghub.services.document import DocumentService
from raghub.services.health import HealthService
from raghub.services.query_service import QueryService

RAG_FACADE_AVAILABLE: bool = importlib.util.find_spec("raghub.api.rag") is not None


class AuthCoordinator:
    """Facade for the auth-shaped :class:`ApplicationFacade` methods.

    Attributes:
        facade: The owning :class:`ApplicationFacade`.
    """

    def __init__(self, facade: Any) -> None:
        """Store the facade reference."""
        self.facade = facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            The :class:`AuthLoginResponse` produced by
            :meth:`AuthService.login`.
        """
        return cast(AuthLoginResponse, await self.facade.auth_svc.login(email, password))

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store.

        Args:
            token: The bearer token presented by the client.
        """
        await self.facade.auth_svc.logout(token)

    async def resolve_user(
        self, token: str
    ) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve a bearer token to a principal plus conversation history.

        Args:
            token: The bearer token.

        Returns:
            A tuple of (UserPrincipal, history).
        """
        return cast(
            tuple[UserPrincipal, list[ConversationTurn]],
            await self.facade.auth_svc.resolve_user(token),
        )


class ShutdownCoordinator:
    """Release collaborators held by the :class:`DynamicRagContainer`.

    The coordinator is intentionally stateless — every collaborator
    owns the resource it backs. The container is the single source of
    truth, so the coordinator just iterates the well-known list.

    Attributes:
        container: The application container whose collaborators
            will be released.
    """

    SHUTDOWN_TARGETS: tuple[str, ...] = (
        "background_ingestion",
        "ingestion",
        "image_store",
        "vector_store",
        "store",
        "uow",
    )

    def __init__(self, container: Any) -> None:
        """Store the container reference."""
        self.container = container

    async def release(self) -> None:
        """Close every owned collaborator in order."""
        for attr in self.SHUTDOWN_TARGETS:
            collaborator = getattr(self.container, attr, None)
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None) or getattr(collaborator, "shutdown", None)
            if close is None:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result


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
            available, the request is routed through :meth:`RAG.aquery`
            so the agent loop and tools actually run end-to-end.
            Otherwise the legacy :class:`QueryService` is used and the
            response carries the resolved config for observability.
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
            return cast(QueryResponse, response)

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


class ApplicationFacade:
    """High-level facade exposing every public action.

    The application holds the container and four service handles. Each
    public method delegates to the appropriate service so the facade
    stays thin. Use :meth:`health` for liveness, :meth:`query` for
    retrieval-augmented Q/A, and the ``upload_/list_/delete_document``
    trio for document management.
    """

    def __init__(self, container: Any) -> None:
        """Initialise the facade and wire service handles back into the container.

        Args:
            container: The fully-wired application container.
        """
        self.container = container
        self.auth_svc = AuthService(container)
        self.documents_svc = DocumentService(container)
        self.query_svc = QueryService(container)
        self.health_svc = HealthService(container)
        container.auth = self.auth_svc
        container.documents = self.documents_svc
        container.query = self.query_svc
        container.health = self.health_svc
        self.auth = AuthCoordinator(self)
        self._shutdown_coordinator = ShutdownCoordinator(container)
        self.preferences = PreferenceCoordinator(self)

    @staticmethod
    def build_rag_facade(container: Any) -> Any | None:
        """Construct a :class:`raghub.RAG` from the container's collaborators.

        Args:
            container: The wired :class:`DynamicRagContainer`.

        Returns:
            A configured :class:`raghub.RAG`, or ``None`` when the
            :mod:`raghub.api.rag` module is unavailable (e.g. the
            optional web dependencies are not installed).
        """
        if not RAG_FACADE_AVAILABLE:
            return None
        import importlib as _importlib

        rag_module = _importlib.import_module("raghub.api.rag")
        return rag_module.RAG(
            settings=container.settings,
            embedder=container.embeddings,
            llm=container.llm,
            vector_store=container.vector_store,
            knowledge_repo=container.registry,
            conversation_store=getattr(container, "conversation_store", None),
        )

    def rag_facade(self) -> Any | None:
        """Return the lazily-built :class:`raghub.RAG` instance.

        First call constructs the facade and caches it on the
        container; subsequent calls return the cached instance. The
        cache is filled lazily so a missing optional dep does not
        prevent the application from booting.
        """
        if getattr(self.container, "rag_facade", None) is None:
            self.container.rag_facade = self.build_rag_facade(self.container)
        return self.container.rag_facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""
        return await self.auth.login(email, password)

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store."""
        await self.auth.logout(token)

    async def resolve_user(
        self, token: str
    ) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve a bearer token to a principal plus conversation history."""
        return await self.auth.resolve_user(token)

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentRecord:
        """Upload ``content`` as a new document owned by the calling user."""
        return await self.documents_svc.upload_document(
            token=token, filename=filename, content=content, company=company
        )

    async def list_documents(self, token: str) -> list[DocumentRecord]:
        """List the documents visible to the calling user."""
        return await self.documents_svc.list_documents(token)

    async def document_status(self, token: str, document_id: str) -> DocumentRecord:
        """Return the status of a single document."""
        return await self.documents_svc.document_status(token, document_id)

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks."""
        await self.documents_svc.delete_document(token, document_id)

    async def clear_history(self, token: str) -> None:
        """Empty the conversation history for ``token``."""
        await self.container.conversation.clear(token)

    async def history(self, token: str) -> list[ConversationTurn]:
        """Return the full conversation history for ``token``."""
        return cast(
            list[ConversationTurn],
            await self.container.conversation.load(token),
        )

    def health(self) -> dict[str, object]:
        """Run liveness checks and return a status dict."""
        return self.health_svc.health()

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single retrieval-augmented Q/A turn."""
        return await self.query_svc.query(token=token, question=question)

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
        """Resolve advanced-RAG flags against user prefs and route accordingly."""
        return await self.preferences.query_with_flags(
            token=token,
            question=question,
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

    def log(self, message: str, **payload: object) -> None:
        """Emit a structured log event via the health service."""
        self.health_svc.log(message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Emit a latency metric given a perf-counter start time."""
        self.health_svc.emit_metric(name, started_at)

    async def shutdown(self) -> None:
        """Release all resources held by the application."""
        await self._shutdown_coordinator.release()