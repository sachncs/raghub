"""High-level application facade.

:class:`ApplicationFacade` wraps a fully-wired
:class:`DynamicRagContainer` with the public methods the API and CLI
call. The facade is intentionally thin — each public method delegates
to a coordinator (:class:`AuthCoordinator`,
:class:`ShutdownCoordinator`, :class:`PreferenceCoordinator`) or a
service handle so the surface stays uniform.

The legacy alias ``DynamicRagApplication`` points at this class; it
is re-exported from :mod:`raghub.services.application.__init__`.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from raghub.models import (
    AuthLoginResponse,
    ConversationTurn,
    DocumentRecord,
    QueryResponse,
    UserPrincipal,
)
from raghub.services.application.auth import AuthCoordinator
from raghub.services.application.preferences import PreferenceCoordinator
from raghub.services.application.shutdown import ShutdownCoordinator
from raghub.auth import AuthService
from raghub.services.document import DocumentService
from raghub.services.health import HealthService
from raghub.services.query_service import QueryService

_RAG_FACADE_AVAILABLE: bool = importlib.util.find_spec("raghub.api.rag") is not None


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
        self.shutdown = ShutdownCoordinator(container)
        self.preferences = PreferenceCoordinator(self)

    @staticmethod
    def build_rag_facade(container: Any) -> Any | None:
        """Construct a :class:`raghub.RAG` from the container's collaborators.

        Args:
            container: The wired :class:`DynamicRagContainer`.

        Returns:
            A configured :class:`raghub.RAG`, or ``None`` when the
            :mod:`raghub.api.rag` module is unavailable (e.g. the
            optional web dependencies are not installed). Real
            construction failures propagate to the caller so the
            boot path surfaces them.
        """
        if not _RAG_FACADE_AVAILABLE:
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
        return await self.container.conversation.load(token)

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
        await self.shutdown.release()


__all__ = ["ApplicationFacade"]