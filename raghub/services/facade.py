"""High-level facade exposing every public action."""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import TYPE_CHECKING, Any, cast

from raghub.auth import AuthService
from raghub.models import AuthLoginResponse, Document, QueryResponse, Turn, User
from raghub.services.documents import Documents
from raghub.services.health import Health
from raghub.services.preference import Preference
from raghub.services.query import Query
from raghub.services.shutdown import Shutdown

if TYPE_CHECKING:
    from raghub.rag.facade import RAG
    from raghub.services.container import RagContainer

RAG_FACADE_AVAILABLE: bool = importlib.util.find_spec("raghub.rag") is not None


class ApplicationFacade:  # ruff: ignore[too-many-public-methods] -- facade aggregating the service surface
    """High-level application facade exposing every public action.

    Renamed from :class:`Facade` (the bare name violated AGENTS.md §602-630
    as a forbidden class name; see :class:`Facade` for the deprecation alias).

    The application holds the container and four service handles. Each
    public method delegates to the appropriate service so the facade
    stays thin.
    """

    def __init__(self, container: RagContainer) -> None:
        """Initialise the facade and wire service handles back into the container."""
        self.container = container
        self.auth = AuthService(container)
        self.documents = Documents(container)
        query_service = Query(container)
        health_service = Health(container)
        container.auth = self.auth
        container.documents = self.documents
        container.query = query_service
        container.health = health_service
        self.shutdown_coordinator = Shutdown(container)
        self.preferences = Preference(self)

    @staticmethod
    def build_rag(container: RagContainer) -> RAG | None:
        """Construct a :class:`raghub.RAG` from the container's collaborators."""
        if not RAG_FACADE_AVAILABLE:
            return None
        rag_module = importlib.import_module("raghub.rag")
        return rag_module.RAG(
            settings=container.settings,
            embedder=container.embeddings,
            llm=container.llm,
            vector_store=container.vector_store,
            knowledge_repo=container.registry,
            conversation_store=getattr(container, "conversation_store", None),
        )

    def rag_facade(self) -> RAG | None:
        """Return the lazily-built :class:`raghub.RAG` instance."""
        if getattr(self.container, "rag_facade", None) is None:
            self.container.rag_facade = self.build_rag(self.container)
        return self.container.rag_facade

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""
        return await self.auth.login(email, password)

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store."""
        return await self.auth.logout(token)

    async def resolve_user(self, token: str) -> tuple[User, list[Turn]]:
        """Resolve a bearer token to a principal plus conversation history."""
        return await self.auth.resolve_user(token)

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> Document:
        """Upload ``content`` as a new document owned by the calling user."""
        return await self.documents.upload_document(
            token=token, filename=filename, content=content, company=company
        )

    async def list_documents(self, token: str) -> list[Document]:
        """List the documents visible to the caller."""
        return await self.documents.list_documents(token)

    # ------------------------------------------------------------------
    # Demeter-chain-breaking pass-through methods (routes/admin only)
    # ------------------------------------------------------------------

    async def list_all_documents(self) -> list[Document]:
        """List every document in the repository (admin only).

        Breaks the ``app_service.container.uow.document_repo.list_all()``
        Demeter chain by giving routes a single named entry point.
        """
        return cast(list[Document], await self.container.uow.document_repo.list_all())

    async def list_all_users(self) -> list[User]:
        """List every user (admin only)."""
        return cast(list[User], await self.container.user_store.list_users())

    async def vector_store_health(self) -> dict[str, Any]:
        """Return the vector store health snapshot."""
        return cast(dict[str, Any], self.container.vector_store.health())

    def get_max_upload_bytes(self) -> int:
        """Return the configured upload-byte limit (0 = unlimited)."""
        return int(getattr(self.container.settings, "max_upload_bytes", 0) or 0)

    def get_rag_facade(self) -> Any | None:
        """Return the cached RAG facade, or None if not yet built."""
        return getattr(self.container, "rag_facade", None)

    def get_vector_chunk_count(self) -> int:
        """Return the vector store's chunk count from the health snapshot."""
        return self.vector_store_health().get("chunks", 0)

    async def document_status(self, token: str, document_id: str) -> Document:
        """Return the status of a single document."""
        return await self.documents.document_status(token, document_id)

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks."""
        await self.documents.delete_document(token, document_id)

    async def clear_history(self, token: str) -> None:
        """Empty the conversation history for ``token``."""
        await self.container.conversation.clear(token)

    async def history(self, token: str) -> list[Turn]:
        """Return the full conversation history for ``token``."""
        return cast(
            list[Turn],
            await self.container.conversation.load(token),
        )

    def health(self) -> dict[str, object]:
        """Run liveness checks and return a status dict."""
        return self.container.health.health()

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single retrieval-augmented Q/A turn."""
        return await self.container.query.query(token=token, question=question)

    async def query_with_flags(
        self,
        *,
        token: str,
        question: str,
        **flags: Any,
    ) -> QueryResponse:
        """Resolve advanced-RAG flags against user prefs and route accordingly.

        Args:
            token: The session token.
            question: The user's question.
            **flags: Optional advanced-RAG overrides forwarded to
                :meth:`preferences.query_with_flags`.

        """
        return await self.preferences.query_with_flags(
            token=token,
            question=question,
            **flags,
        )

    def log(self, message: str, **payload: object) -> None:
        """Emit a structured log event via the health service."""
        self.container.health.log(message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Emit a latency metric given a perf-counter start time."""
        self.container.health.emit_metric(name, started_at)

    async def shutdown(self) -> None:
        """Release all resources held by the application."""
        await self.shutdown_coordinator.release()


__all__ = ["RAG_FACADE_AVAILABLE", "ApplicationFacade", "Facade"]


# Deprecated alias preserved for one minor version. Use ApplicationFacade in new code.
class FacadeDeprecationMeta(type):
    """Metaclass that emits DeprecationWarning on first instantiation."""

    _warned: bool = False

    def __call__(cls, *args, **kwargs):
        if not cls._warned:
            warnings.warn(
                "raghub.services.Facade has been renamed to "
                "raghub.services.ApplicationFacade; import the new name. "
                "This compatibility alias will be removed in the next minor release.",
                DeprecationWarning,
                stacklevel=2,
            )
            cls._warned = True
        return super().__call__(*args, **kwargs)


class Facade(ApplicationFacade, metaclass=FacadeDeprecationMeta):
    """Deprecated alias for :class:`ApplicationFacade`."""
