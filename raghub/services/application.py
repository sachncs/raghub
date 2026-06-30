"""Application container and facade wiring every collaborator.

This module is the **single assembly point** for the entire RAG
application. :func:`build_container` instantiates every collaborator
(settings, logger, metrics, auth, RBAC, sessions, embeddings, LLM, vector
store, ingestion, retrieval, parsers, image store) into a
:class:`DynamicRagContainer`, then :class:`DynamicRagApplication` wraps
that container with the high-level facade methods the API and CLI call.

Why a dataclass container? It gives us:

* **Single dependency surface.** Services receive the container and ask
  it for collaborators they need; we avoid deep parameter lists.
* **Easy introspection.** Tests can swap any field by reassigning it on
  the container before exercising the system under test.
* **Lazy services.** The ``auth``, ``documents``, ``query``, ``health``
  fields are filled in by :class:`DynamicRagApplication.__init__` so the
  container can be built without depending on the service modules.

Failure modes: :func:`build_container` raises :class:`RuntimeError` when
``JWT_SECRET`` is missing from settings or when no LLM credentials are
provided. SQLite-backed stores are initialised eagerly during build
(``initialize`` is called) so a misconfigured schema surfaces immediately
rather than on the first request.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from raghub.auth import (
    JwtAuthenticator,
    JwtSessionManager,
    RBACAuthorizationService,
    SqliteUserStore,
)
from raghub.config.settings import AppSettings
from raghub.conversation.manager import ConversationManager
from raghub.documents.chunker import ChunkingPlan
from raghub.documents.lifecycle import DocumentLifecycleManager
from raghub.documents.parsers import ParserRegistry
from raghub.embeddings import BaseEmbeddingProvider, build_embedding_provider
from raghub.ingestion.service import DocumentIngestionService
from raghub.llm import BaseLLMProvider, build_llm_provider
from raghub.observability.logging import build_logger
from raghub.observability.metrics import PrometheusMetrics
from raghub.prompts.builder import PromptBuilder
from raghub.retrieval.pipeline import RetrievalPipeline
from raghub.retrieval.reranker import IdentityReranker
from raghub.storage.image_store import FilesystemImageStore
from raghub.storage.sqlite_session_store import SqliteSessionStore
from raghub.models import (
    AuthLoginResponse,
    ConversationTurn,
    DocumentRecord,
    QueryResponse,
    UserPrincipal,
)
from raghub.repositories import UnitOfWork
from raghub.vectorstore.base import BaseVectorStore
from raghub.vectorstore.zvec import ZvecVectorStore


@dataclass
class DynamicRagContainer:
    """Composition root: every collaborator the application needs.

    Field roles:

    * ``settings`` — typed configuration snapshot.
    * ``logger`` / ``metrics`` — observability primitives.
    * ``authenticator`` / ``authorization`` / ``sessions`` — auth trio.
    * ``registry`` — user store aliased for legacy call sites that
      expected a "registry" name (kept for backward compatibility).
    * ``conversation`` — chat-history manager.
    * ``embeddings`` / ``llm`` — AI providers.
    * ``vector_store`` / ``prompt_builder`` / ``ingestion`` / ``retrieval``
      — RAG pipeline pieces.
    * ``image_store`` / ``parser_registry`` — auxiliary stores.
    * ``user_store`` — same instance as ``registry``; named for clarity.
    * ``store`` — raw :class:`SqliteSessionStore` (used by ``sessions``).
    * ``uow`` — Unit-of-Work for transactional repo access.
    * ``auth`` / ``documents`` / ``query`` / ``health`` — service handles
      populated by :class:`DynamicRagApplication.__init__`.

    Attributes:
        settings: Application configuration.
        logger: Structured logger (see :mod:`raghub.observability.logging`).
        metrics: Prometheus metrics sink.
        authenticator: JWT authenticator.
        authorization: RBAC service.
        sessions: Login / logout facade.
        registry: Backward-compat alias for ``user_store``.
        conversation: Chat-history manager.
        embeddings: Embedding provider.
        llm: LLM provider.
        vector_store: Vector store.
        prompt_builder: Token-aware prompt builder.
        ingestion: Document ingestion service.
        retrieval: Retrieval pipeline.
        image_store: Filesystem image store.
        user_store: User CRUD store.
        parser_registry: Document format parser registry.
        store: SQLite-backed session store.
        uow: Unit-of-work for repos.
        auth: :class:`AuthService` handle (set by
            :class:`DynamicRagApplication`).
        documents: :class:`DocumentService` handle.
        query: :class:`QueryService` handle.
        health: :class:`HealthService` handle.
    """

    settings: AppSettings
    logger: object
    metrics: object
    authenticator: JwtAuthenticator
    authorization: RBACAuthorizationService
    sessions: JwtSessionManager
    registry: SqliteUserStore
    conversation: ConversationManager
    embeddings: BaseEmbeddingProvider
    llm: BaseLLMProvider
    vector_store: BaseVectorStore
    prompt_builder: PromptBuilder
    ingestion: DocumentIngestionService
    retrieval: RetrievalPipeline
    image_store: FilesystemImageStore
    user_store: SqliteUserStore
    parser_registry: ParserRegistry
    store: SqliteSessionStore
    uow: UnitOfWork
    auth: object = None
    documents: object = None
    query: object = None
    health: object = None


from raghub.services.auth_service import AuthService  # noqa: E402
from raghub.services.document_service import DocumentService  # noqa: E402
from raghub.services.health_service import HealthService  # noqa: E402
from raghub.services.query_service import QueryService  # noqa: E402


class DynamicRagApplication:
    """High-level facade exposing every public action.

    The application holds the container and four service handles. Each
    public method delegates to the appropriate service so the facade
    stays thin. Use :meth:`health` for liveness, :meth:`query` for
    retrieval-augmented Q/A, and the ``upload_/list_/delete_document``
    trio for document management.
    """

    def __init__(self, container: DynamicRagContainer) -> None:
        """Initialise the facade and wire service handles back into the container.

        Args:
            container: The fully-wired application container.
        """
        self.container = container
        self.auth_svc = AuthService(container)
        self.documents_svc = DocumentService(container)
        self.query_svc = QueryService(container)
        self.health_svc = HealthService(container)
        # Cross-link: the container also exposes the service handles so
        # downstream code that only has the container can reach the
        # services without going through this facade.
        container.auth = self.auth_svc
        container.documents = self.documents_svc
        container.query = self.query_svc
        container.health = self.health_svc

    async def login(self, email: str, password: str) -> AuthLoginResponse:
        """Authenticate a user and return a session token.

        Args:
            email: User email.
            password: Plaintext password.

        Returns:
            The :class:`AuthLoginResponse` produced by
            :meth:`AuthService.login`.
        """
        return await self.auth_svc.login(email, password)

    async def logout(self, token: str) -> None:
        """Invalidate ``token`` in the session store.

        Args:
            token: The bearer token presented by the client.
        """
        await self.auth_svc.logout(token)

    async def resolve_user(self, token: str) -> tuple[UserPrincipal, list[ConversationTurn]]:
        """Resolve a bearer token to a principal plus conversation history.

        Args:
            token: The bearer token.

        Returns:
            A tuple of (UserPrincipal, history). The history comes from
            the session record, **not** from the conversation manager.
        """
        return await self.auth_svc.resolve_user(token)

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentRecord:
        """Upload ``content`` as a new document owned by the calling user.

        Args:
            token: Bearer token of the uploading user.
            filename: Original filename; used for MIME detection.
            content: Raw bytes of the upload.
            company: Optional explicit tenant. When ``None``, the company
                is derived from ``filename.split("_", 1)[0]``.

        Returns:
            The persisted :class:`DocumentRecord`.
        """
        return await self.documents_svc.upload_document(
            token=token, filename=filename, content=content, company=company
        )

    async def list_documents(self, token: str) -> list[DocumentRecord]:
        """List the documents visible to the calling user.

        Admins see every document; non-admins see only those whose
        ``organization`` is in their allow-list.

        Args:
            token: Bearer token.

        Returns:
            A list of :class:`DocumentRecord`.
        """
        return await self.documents_svc.list_documents(token)

    async def document_status(self, token: str, document_id: str) -> DocumentRecord:
        """Return the status of a single document.

        Args:
            token: Bearer token.
            document_id: The document id.

        Returns:
            The :class:`DocumentRecord`.
        """
        return await self.documents_svc.document_status(token, document_id)

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks.

        Admin-only: non-admin callers receive :class:`AuthorizationError`.

        Args:
            token: Bearer token.
            document_id: The document id.
        """
        await self.documents_svc.delete_document(token, document_id)

    async def clear_history(self, token: str) -> None:
        """Empty the conversation history for ``token``.

        Args:
            token: Bearer token identifying the session.
        """
        await self.container.conversation.clear(token)

    async def history(self, token: str) -> list[ConversationTurn]:
        """Return the full conversation history for ``token``.

        Args:
            token: Bearer token.

        Returns:
            The chronological list of turns. Empty when the token is
            unknown.
        """
        return await self.container.conversation.load(token)

    def health(self) -> dict[str, object]:
        """Run liveness checks and return a status dict.

        Returns:
            A dict whose ``status`` field is ``"ok"`` when every
            collaborator is healthy, otherwise a per-component report.
        """
        return self.health_svc.health()

    async def query(self, *, token: str, question: str) -> QueryResponse:
        """Run a single retrieval-augmented Q/A turn.

        Args:
            token: Bearer token.
            question: The user's question.

        Returns:
            A :class:`QueryResponse` containing the answer, citations,
            and source chunks.
        """
        return await self.query_svc.query(token=token, question=question)

    def log(self, message: str, **payload: object) -> None:
        """Emit a structured log event via the health service.

        Args:
            message: Event name.
            **payload: Arbitrary structured fields.
        """
        self.health_svc.log(message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Emit a latency metric given a perf-counter start time.

        Args:
            name: Metric name.
            started_at: The value returned by :func:`time.perf_counter`
                at the start of the measured operation.
        """
        self.health_svc.emit_metric(name, started_at)

    async def shutdown(self) -> None:
        """Release all resources held by the application.

        Closes the database manager, the in-memory vector store, and
        any background ingestion service that the application owns.
        Safe to call multiple times.
        """
        for attr in ("uow", "store", "vector_store", "image_store", "ingestion"):
            collaborator = getattr(self.container, attr, None)
            if collaborator is None:
                continue
            close = getattr(collaborator, "close", None) or getattr(collaborator, "shutdown", None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                continue


async def build_container(settings: AppSettings) -> DynamicRagContainer:
    """Construct a fully-wired :class:`DynamicRagContainer`.

    The build is ordered so that every collaborator's dependencies are
    available when needed:

    1. Logger and metrics.
    2. User store (initialised against ``data_dir/users.db``).
    3. JWT authenticator (validates ``settings.jwt_secret`` is non-empty).
    4. RBAC service.
    5. Vector store (``ZvecVectorStore`` with optional fallback).
    6. Unit-of-work (initialised against the registry SQLite db).
    7. Session store (initialised against ``data_dir/sessions.db``).
    8. Embeddings, LLM (built via factory helpers).
    9. Prompt builder, conversation, ingestion, retrieval, image store.

    Args:
        settings: The loaded application settings.

    Returns:
        A populated :class:`DynamicRagContainer` ready to be wrapped
        by :class:`DynamicRagApplication`.

    Raises:
        RuntimeError: If ``JWT_SECRET`` is missing from settings.
    """
    logger = build_logger(settings.log_level)
    metrics = PrometheusMetrics()

    user_store = SqliteUserStore(settings.data_dir / "users.db")
    await user_store.initialize()

    jwt_secret = settings.jwt_secret
    if not jwt_secret:
        # ``RuntimeError`` rather than a typed exception so it surfaces
        # in startup logs without callers needing to catch a custom
        # hierarchy.
        raise RuntimeError("JWT_SECRET must be configured")
    nvidia_api_key = settings.nvidia_api_key or settings.extra.get("nvidia_api_key")

    authenticator = JwtAuthenticator(
        secret_key=jwt_secret,
        user_store=user_store,
    )
    authorization = RBACAuthorizationService(user_store)

    vector_store: BaseVectorStore = ZvecVectorStore(
        str(settings.zvec_dir),
        embedding_dim=settings.embedding_dim,
        require_zvec=settings.require_zvec,
    )

    # The registry SQLite db lives next to the JSON registry, sharing
    # the same name with the extension swapped.
    db_path = str(settings.registry_path).replace(".json", ".db")
    uow = UnitOfWork(
        db_path=db_path,
        vector_store=vector_store,
        session_timeout=settings.session_timeout_seconds,
    )
    await uow.initialize()

    raw_session_store = SqliteSessionStore(
        settings.data_dir / "sessions.db",
        settings.session_timeout_seconds,
    )
    await raw_session_store.initialize()
    sessions = JwtSessionManager(
        session_store=raw_session_store,
        authenticator=authenticator,
    )

    embeddings: BaseEmbeddingProvider = build_embedding_provider(
        settings.embedding_model,
        settings.embedding_dim,
        nvidia_api_key,
    )
    llm: BaseLLMProvider = build_llm_provider(
        settings.llm_model,
        nvidia_api_key,
    )

    prompt_builder = PromptBuilder()
    conversation = ConversationManager(uow)
    lifecycle = DocumentLifecycleManager()
    ingestion = DocumentIngestionService(
        uow=uow,
        embedding_provider=embeddings,
        lifecycle_manager=lifecycle,
        plan=ChunkingPlan(settings.chunk_size_words, settings.chunk_overlap_words),
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalPipeline(
        embedding_provider=embeddings,
        vector_store=vector_store,
        reranker=IdentityReranker(),
    )
    image_store = FilesystemImageStore(settings.data_dir / "images")
    parser_registry = ParserRegistry()

    # Idempotent demo-user seeding so the application works out of
    # the box without an external auth bootstrap step. Users with
    # ``is_admin: true`` see every company; non-admin users are
    # scoped to the listed companies. Passwords are bcrypt-hashed.
    import json as _json
    import os as _os

    _users_env = _os.getenv("RAGHUB_USERS", "").strip()
    if _users_env:
        try:
            _seed_users = _json.loads(_users_env)
        except _json.JSONDecodeError as exc:
            raise RuntimeError(
                f"RAGHUB_USERS is not valid JSON: {exc}"
            ) from exc
        if isinstance(_seed_users, dict):
            for _email, _cfg in _seed_users.items():
                if not isinstance(_cfg, dict):
                    continue
                _existing = await user_store.get_by_email(_email)
                if _existing is not None:
                    continue
                await user_store.create_user(
                    email=_email,
                    password=str(_cfg.get("password", "password")),
                    companies=list(_cfg.get("companies", []) or []),
                    is_admin=bool(_cfg.get("is_admin", False)),
                )
    else:
        # Default seed: the demo users documented in the README.
        # The default password is ``"password"``; operators are
        # expected to rotate it (or override via ``RAGHUB_USERS``)
        # before any production exposure.
        _default_seed = [
            ("alice@acme.com", "password", ["Apple"], False),
            ("bob@acme.com", "password", ["Microsoft"], False),
            ("charlie@acme.com", "password", ["Amazon", "Tesla"], False),
            ("diana@acme.com", "password", ["Google"], False),
            ("admin@acme.com", "password", [], True),
        ]
        for _email, _pwd, _companies, _is_admin in _default_seed:
            _existing = await user_store.get_by_email(_email)
            if _existing is not None:
                continue
            await user_store.create_user(
                email=_email,
                password=_pwd,
                companies=_companies,
                is_admin=_is_admin,
            )

    return DynamicRagContainer(
        settings=settings,
        logger=logger,
        metrics=metrics,
        authenticator=authenticator,
        authorization=authorization,
        sessions=sessions,
        registry=user_store,
        conversation=conversation,
        embeddings=embeddings,
        llm=llm,
        vector_store=vector_store,
        prompt_builder=prompt_builder,
        ingestion=ingestion,
        retrieval=retrieval,
        image_store=image_store,
        user_store=user_store,
        parser_registry=parser_registry,
        store=raw_session_store,
        uow=uow,
    )