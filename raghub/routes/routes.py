"""HTTP routes and exception handlers for the RAGHub API.

The :class:`RouteGroup` composes the focused route classes
(:class:`HealthRoute`, :class:`AuthRoute`, :class:`DocumentRoute`,
:class:`QueryRoute`, :class:`AdminRoute`, :class:`PreferenceRoute`,
:class:`FeedbackRoute`) under the ``/v1`` prefix. Class summary::

    HealthRoute         - liveness probe endpoint.
    AuthRoute           - auth and session-history endpoints.
    DocumentRoute       - document upload, listing, status, delete.
    QueryRoute          - synchronous and streaming query endpoints.
    AdminRoute          - admin-only endpoints mounted under /admin.
    PreferenceRoute     - per-user preferences endpoints.
    FeedbackRoute       - feedback capture endpoints.
    Exceptions          - typed-exception -> HTTP response handlers.
    RouteGroup          - composite that mounts every focused route.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from loguru import logger

from raghub.auth import (
    App,
    Auth,
    Bearer,
)
from raghub.constants import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_202_ACCEPTED,
    HTTP_204_NO_CONTENT,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)
from raghub.errors import (
    AuthenticationError,
    AuthorizationError,
    IngestionError,
    RagHubError,
)
from raghub.ingest import Batch
from raghub.models import (
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestItem,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    Snap,
    User,
)
from raghub.response import Redaction
from raghub.services import Facade
from raghub.sse import Sse

__all__ = [
    "AdminRoute",
    "AuthRoute",
    "DocumentRoute",
    "Exceptions",
    "FeedbackAggregateResponse",
    "FeedbackRoute",
    "FeedbackSubmission",
    "HealthRoute",
    "PreferenceRoute",
    "PreferencesPatch",
    "PreferencesResponse",
    "QueryRoute",
    "RouteGroup",
]


def user_store_or_raise(app_service: Facade) -> Any:
    """Return the configured user store or raise 503."""
    store = getattr(app_service.container, "user_store", None)
    if store is None:
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="user store unavailable"
        )
    if not hasattr(store, "get_pref") or not hasattr(store, "set_pref"):
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE, detail="user store lacks prefs API"
        )
    return store


def has_flags(payload: QueryRequest) -> bool:
    """Return ``True`` when any advanced-RAG flag is set on the payload."""
    return any(
        getattr(payload, field) is not None
        for field in (
            "tools_enabled",
            "agent",
            "web",
            "graph",
            "summaries",
            "reranker",
            "long_context_pass",
            "query_transforms",
            "max_steps",
            "top_k",
        )
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


class Exceptions:
    """Install typed-exception -> HTTP response handlers on the app."""

    @staticmethod
    def install(app: FastAPI) -> None:
        """Register exception handlers on ``app``.

        Args:
            app: The FastAPI instance.

        """

        @app.exception_handler(AuthenticationError)
        def auth_error(_: Any, exc: AuthenticationError) -> Any:
            """Return 401 for any :class:`AuthenticationError`."""
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=HTTP_401_UNAUTHORIZED, content={"detail": str(exc)})

        @app.exception_handler(AuthorizationError)
        def authz_error(_: Any, exc: AuthorizationError) -> Any:
            """Return 403 for any :class:`AuthorizationError`."""
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=HTTP_403_FORBIDDEN, content={"detail": str(exc)})

        @app.exception_handler(IngestionError)
        def ingestion_error_handler(_: Any, exc: IngestionError) -> Any:
            """Return 400 for any :class:`IngestionError`."""
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=HTTP_400_BAD_REQUEST, content={"detail": str(exc)})

        @app.exception_handler(RagHubError)
        def generic_error_handler(_: Any, exc: RagHubError) -> Any:
            """Return 500 for any uncategorised :class:`RagHubError`."""
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": str(exc)}
            )


# ---------------------------------------------------------------------------
# Preferences request/response models
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PreferencesResponse(Snap):
    """Preferences for the authenticated user.

    Attributes:
        prefs: Mapping of preference key -> JSON value. The reserved
            key ``"tool_settings"`` carries the ChatGPT-style tool
            toggles consumed by :func:`raghub.agent.resolve`.

    """

    prefs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PreferencesPatch(Snap):
    """Preferences update payload.

    Attributes:
        prefs: Replacement mapping for the supplied keys. Keys absent
            from the payload are left unchanged on disk.

    """

    prefs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FeedbackSubmission(Snap):
    """Inbound feedback payload."""

    session_id: str = ""
    query_id: str = ""
    chunk_id: str | None = None
    answer_id: str | None = None
    rating: int = 0
    comment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class FeedbackAggregateResponse(Snap):
    """Aggregate counts response."""

    tenant_id: str | None = None
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    by_chunk: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Route classes
# ---------------------------------------------------------------------------


class HealthRoute:
    """Health probe endpoint."""

    router: APIRouter

    def __init__(self) -> None:
        """Build the router."""
        self.router = APIRouter()
        self.register_health()

    def register_health(self) -> None:
        """Register ``GET /health`` returning the liveness probe response."""

        @self.router.get("/health")
        def handler(
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            """Report liveness."""
            return app_service.health()


class AuthRoute:
    """Auth and session-history endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        """Wire the auth router and register every auth/history endpoint."""
        self.router = APIRouter()
        self.register_login()
        self.register_logout()
        self.register_session_history()
        self.register_clear_history()

    def register_login(self) -> None:
        """Register ``POST /auth/login`` for credentials-based sign-in."""

        @self.router.post("/auth/login", response_model=AuthLoginResponse)
        async def handler(
            payload: AuthLoginRequest,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> AuthLoginResponse:
            """Handle ``POST /auth/login``."""
            return await app_service.login(payload.email, payload.password)

    def register_logout(self) -> None:
        """Register ``POST /auth/logout`` to invalidate the bearer token."""

        @self.router.post("/auth/logout")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            """Handle ``POST /auth/login``."""
            await app_service.logout(token)
            return {"status": "logged_out"}

    def register_session_history(self) -> None:
        """Register ``GET /session/history`` returning the stored turns."""

        @self.router.get("/session/history")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, list[dict[str, Any]]]:
            """Handle ``POST /auth/logout``."""
            history = await app_service.history(token)
            return {"history": [turn.dump(mode="json") for turn in history]}

    def register_clear_history(self) -> None:
        """Register ``DELETE /session/history`` to drop the caller's turns."""

        @self.router.delete(
            "/session/history", status_code=HTTP_204_NO_CONTENT, response_class=Response
        )
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            """Handle ``DELETE /session/history``."""
            await app_service.clear_history(token)
            return Response(status_code=HTTP_204_NO_CONTENT)


class DocumentRoute:
    """Document upload, listing, status, and delete endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        """Wire the documents router and register every document endpoint."""
        self.router = APIRouter()
        self.register_upload()
        self.register_ingest_batch()
        self.register_list()
        self.register_status()
        self.register_delete()
        self.register_ingest_async()

    def register_upload(self, *, enforce_limit: Any | None = None) -> None:
        """Register ``POST /documents/upload`` for synchronous single uploads."""

        @self.router.post(
            "/documents/upload",
            status_code=HTTP_202_ACCEPTED,
            response_model=DocumentUploadResponse,
        )
        async def handler(
            request: Request,
            file: Annotated[UploadFile, File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> DocumentUploadResponse:
            """Handle ``POST /documents/upload``."""
            from raghub.routes.limits import enforce_limit

            enforce_limit(request, app_service.container)
            content = await file.read()
            enforce_limit(request, app_service.container, payload=content)
            document = await app_service.upload_document(
                token=token,
                filename=file.filename or "upload.pdf",
                content=content,
                company=company,
            )
            return DocumentUploadResponse(
                document_id=document.id,
                version=document.version,
                status=document.status.value,
                company=document.organization,
                filename=document.filename,
            )

    def register_ingest_batch(self) -> None:
        """Register ``POST /documents/ingest/batch`` for multi-file uploads."""

        @self.router.post(
            "/documents/ingest/batch",
            status_code=HTTP_200_OK,
            response_model=BatchIngestResponse,
        )
        async def handler(
            request: Request,
            files: Annotated[list[UploadFile], File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> BatchIngestResponse:
            """Handle ``POST /documents/ingest/batch``."""
            from raghub.routes.limits import enforce_limit

            enforce_limit(request, app_service.container)
            results: list[BatchIngestItem] = []
            for file in files:
                try:
                    content = await file.read()
                    max_bytes = app_service.get_max_upload_bytes()
                    if max_bytes > 0 and len(content) > max_bytes:
                        results.append(
                            BatchIngestItem(
                                filename=file.filename or "upload.pdf",
                                status="error",
                                error=f"Upload exceeds maximum size of {max_bytes} bytes",
                            )
                        )
                        continue
                    document = await app_service.upload_document(
                        token=token,
                        filename=file.filename or "upload.pdf",
                        content=content,
                        company=company,
                    )
                    results.append(
                        BatchIngestItem(
                            filename=file.filename or "upload.pdf",
                            document_id=document.id,
                            status="ok",
                        )
                    )
                except (IngestionError, RagHubError, ValueError, TypeError, OSError) as exc:
                    logger.warning(
                        "api.batch_upload.item_failed", file=file.filename, error=str(exc)
                    )
                    results.append(
                        BatchIngestItem(
                            filename=file.filename or "upload.pdf",
                            status="error",
                            error=str(exc),
                        )
                    )
            return BatchIngestResponse(documents=results)

    def register_list(self) -> None:
        """Register ``GET /documents`` to list the caller's documents."""

        @self.router.get("/documents")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, list[dict[str, Any]]]:
            """Handle ``GET /documents``."""
            documents = await app_service.list_documents(token)
            return {"documents": [document.dump(mode="json") for document in documents]}

    def register_status(self) -> None:
        """Register ``GET /documents/{document_id}/status`` for a single doc."""

        @self.router.get("/documents/{document_id}/status")
        async def handler(
            document_id: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            """Handle ``GET /documents``."""
            document = await app_service.document_status(token, document_id)
            return document.dump(mode="json")

    def register_delete(self) -> None:
        """Register ``DELETE /documents/{document_id}`` for removal."""

        @self.router.delete(
            "/documents/{document_id}", status_code=HTTP_204_NO_CONTENT, response_class=Response
        )
        async def handler(
            document_id: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            """Handle ``DELETE /documents/{document_id}``."""
            await app_service.delete_document(token, document_id)
            return Response(status_code=HTTP_204_NO_CONTENT)

    def register_ingest_async(self) -> None:
        """Register ``POST /ingest/async`` for background ingestion jobs."""

        @self.router.post("/ingest/async")
        async def handler(
            request: Request,
            file: Annotated[UploadFile, File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            """Handle ``POST /ingest/async``."""
            from raghub.routes.limits import enforce_limit

            enforce_limit(request, app_service.container)
            content = await file.read()
            enforce_limit(request, app_service.container, payload=content)
            background: Batch = request.app.state.background_ingestion
            job_id = background.submit(
                app_service.upload_document,
                token=token,
                filename=file.filename or "upload.pdf",
                content=content,
                company=company,
            )
            return {"job_id": job_id}


class QueryRoute:
    """Synchronous and streaming query endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        """Wire the query router and register query/stream/agent endpoints."""
        self.router = APIRouter()
        self.register_query()
        self.register_stream()
        self.register_agent_run()

    def register_query(self) -> None:
        """Register ``POST /query`` (non-streaming, with optional flags)."""

        @self.router.post("/query", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> QueryResponse:
            """Handle ``POST /query``."""
            if not has_flags(payload):
                return await app_service.query(token=token, question=payload.question)
            return await app_service.query_with_flags(
                token=token,
                question=payload.question,
                tools_enabled=payload.tools_enabled,
                agent=payload.agent,
                web=payload.web,
                graph=payload.graph,
                summaries=payload.summaries,
                reranker=payload.reranker,
                long_context_pass=payload.long_context_pass,
                query_transforms=payload.query_transforms,
                max_steps=payload.max_steps,
                top_k=payload.top_k,
            )

    def register_stream(self) -> None:
        """Register ``POST /query/stream`` returning SSE-encoded events."""

        @self.router.post("/query/stream")
        def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> StreamingResponse:
            """Handle ``POST /query/stream``."""
            resolved_tools = set(payload.tools_enabled) if payload.tools_enabled else set()

            async def gen() -> AsyncIterator[bytes]:
                """Yield the SSE-encoded stream for the streaming ``/query/stream`` POST."""
                yield Sse.comment("raghub-query-stream")
                user, _ = await app_service.auth.resolve_user(token)
                rag = app_service.get_rag_facade()
                if rag is None:
                    yield Sse.format("error", {"message": "RAG facade unavailable"})
                    return
                async for event in rag.astream_agent(
                    payload.question,
                    user=user,
                    session_id=None,
                    tools_enabled=list(resolved_tools) or None,
                    agent=payload.agent,
                    web=payload.web,
                    graph=payload.graph,
                    summaries=payload.summaries,
                    reranker=payload.reranker,
                    long_context_pass=payload.long_context_pass,
                    query_transforms=payload.query_transforms,
                    max_steps=payload.max_steps,
                ):
                    yield Sse.format(event.kind, event.dump(mode="json"))

            return StreamingResponse(gen(), media_type="text/event-stream")

    def register_agent_run(self) -> None:
        """Register ``POST /agent/run`` to run the agent with the given flags."""

        @self.router.post("/agent/run", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> QueryResponse:
            """Handle ``POST /agent/run``."""
            return await app_service.query_with_flags(
                token=token,
                question=payload.question,
                tools_enabled=payload.tools_enabled,
                agent=payload.agent,
                web=payload.web,
                graph=payload.graph,
                summaries=payload.summaries,
                reranker=payload.reranker,
                long_context_pass=payload.long_context_pass,
                query_transforms=payload.query_transforms,
                max_steps=payload.max_steps,
                top_k=payload.top_k,
            )


class AdminRoute:
    """Admin-only endpoints mounted under ``/admin``."""

    router: APIRouter

    def __init__(self) -> None:
        """Wire the admin router (prefix=/admin, tag=admin)."""
        self.router = APIRouter(prefix="/admin", tags=["admin"])
        self.register_documents()
        self.register_users()
        self.register_stats()

    def register_documents(self) -> None:
        """Register ``GET /admin/documents`` listing every document."""

        @self.router.get("/documents")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> list[dict[str, Any]]:
            """Handle ``GET /admin``."""
            docs = await app_service.list_all_documents()
            return [doc.dump(mode="json") for doc in docs]

    def register_users(self) -> None:
        """Register ``GET /admin/users`` listing redacted user records."""

        @self.router.get("/users")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> list[dict[str, Any]]:
            """Handle ``GET /documents``."""
            users = await app_service.list_all_users()
            return [Redaction.user(user.dump(mode="json")) for user in users]

    def register_stats(self) -> None:
        """Register ``GET /admin/stats`` returning aggregate counts."""

        @self.router.get("/stats")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            """Handle ``GET /users``."""
            docs = await app_service.list_all_documents()
            users = await app_service.list_all_users()
            vector_health = app_service.vector_store_health()
            chunk_count = vector_health.get("chunks", 0)
            return {
                "document_count": len(docs),
                "user_count": len(users),
                "chunk_count": chunk_count,
                "vector_store_size": vector_health.get("size", "unknown"),
            }


class PreferenceRoute:
    """Per-user preferences endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        """Wire the preferences router and register get/patch/delete."""
        self.router = APIRouter()
        self.register_get()
        self.register_patch()
        self.register_delete()

    def register_get(self) -> None:
        """Register ``GET /users/me/preferences`` returning the user's prefs."""

        @self.router.get(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> PreferencesResponse:
            """Handle ``GET /users/me/preferences``."""
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_raise(app_service)
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

    def register_patch(self) -> None:
        """Register ``PATCH /users/me/preferences`` to merge new keys."""

        @self.router.patch(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            payload: PreferencesPatch,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> PreferencesResponse:
            """Handle ``PATCH /users/me/preferences``."""
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_raise(app_service)
            await store.set_prefs(user_id, dict(payload.prefs or {}))
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

    def register_delete(self) -> None:
        """Register ``DELETE /users/me/preferences/{key}`` to remove a key."""

        @self.router.delete(
            "/users/me/preferences/{key}",
            status_code=HTTP_204_NO_CONTENT,
        )
        async def handler(
            key: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> None:
            """Handle ``DELETE /users/me/preferences/{key}``."""
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_raise(app_service)
            await store.delete_pref(user_id, key)


class FeedbackRoute:
    """Feedback capture endpoints (Tier 3 Item 18).

    Endpoints:
        - ``POST /feedback`` -- record feedback
        - ``GET /feedback/{id}`` -- retrieve one record
        - ``DELETE /feedback/{id}`` -- delete one record
        - ``GET /feedback/aggregate?tenant_id=...`` -- aggregate counts
    """

    router: APIRouter

    def __init__(self) -> None:
        """Wire the feedback router and register every feedback endpoint.

        Register aggregate BEFORE the ``{feedback_id}`` catch-all so the
        ``/feedback/aggregate`` route is not treated as a lookup for an
        id literally named ``"aggregate"``.
        """
        self.router = APIRouter()
        # Register aggregate BEFORE the {feedback_id} catch-all so
        # ``/feedback/aggregate`` does not get treated as a lookup for
        # feedback id "aggregate".
        self.register_aggregate()
        self.register_submit()
        self.register_get()
        self.register_delete()

    @staticmethod
    def feedback_store(app_service: Facade) -> Any:
        """Return the configured FeedbackStore or raise ``503`` if absent."""
        store = getattr(app_service.container, "feedback_store", None)
        if store is None:
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="feedback_store not configured",
            )
        return store

    def register_submit(self) -> None:
        """Register ``POST /feedback`` to record a new feedback entry."""

        @self.router.post(
            "/feedback",
            status_code=HTTP_201_CREATED,
        )
        async def handler(
            payload: FeedbackSubmission,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            """Handle ``POST /feedback``."""
            from raghub.feedback import Feedback, FeedbackStore, Rating
            from raghub.tenants import current

            store = self.feedback_store(app_service)
            user, _ = await app_service.auth.resolve_user(token)
            tenant_ctx = current()
            tenant_id = (
                tenant_ctx.tenant_id
                if tenant_ctx is not None
                else getattr(user, "tenant_id", None) or "default"
            )
            feedback = Feedback(
                id=FeedbackStore.new_id(),
                session_id=payload.session_id,
                query_id=payload.query_id,
                chunk_id=payload.chunk_id,
                answer_id=payload.answer_id,
                user_id=user.email,
                tenant_id=tenant_id,
                rating=Rating(payload.rating),
                comment=payload.comment,
                created_at=FeedbackStore.now_utc(),
                metadata=payload.metadata,
            )
            await store.record(feedback)
            return {"id": feedback.id}

    def register_get(self) -> None:
        """Register ``GET /feedback/{feedback_id}`` to fetch one record."""

        @self.router.get("/feedback/{feedback_id}")
        async def handler(
            feedback_id: str,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            """Handle ``GET /feedback/{feedback_id}``."""
            from dataclasses import asdict

            store = self.feedback_store(app_service)
            feedback = await store.get(feedback_id)
            if feedback is None:
                raise HTTPException(status_code=HTTP_404_NOT_FOUND, detail="feedback not found")
            return asdict(feedback)

    def register_delete(self) -> None:
        """Register ``DELETE /feedback/{feedback_id}`` to remove one record."""

        @self.router.delete("/feedback/{feedback_id}", status_code=HTTP_204_NO_CONTENT)
        async def handler(
            feedback_id: str,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            """Handle ``DELETE /feedback/{feedback_id}``."""
            store = self.feedback_store(app_service)
            await store.delete(feedback_id)
            return Response(status_code=HTTP_204_NO_CONTENT)

    def register_aggregate(self) -> None:
        """Register ``GET /feedback/aggregate`` returning per-tenant counts."""

        @self.router.get(
            "/feedback/aggregate",
            response_model=FeedbackAggregateResponse,
        )
        async def handler(
            app_service: Annotated[Facade, Depends(App.get)],
            tenant_id: str | None = None,
        ) -> FeedbackAggregateResponse:
            """Handle ``GET /feedback/aggregate``."""
            store = self.feedback_store(app_service)
            aggregate = await store.aggregate(tenant_id)
            return FeedbackAggregateResponse(
                tenant_id=aggregate.tenant_id,
                positive=aggregate.positive,
                negative=aggregate.negative,
                neutral=aggregate.neutral,
                by_chunk=aggregate.by_chunk,
            )


# ---------------------------------------------------------------------------
# Route group: composite that mounts every focused route under /v1.
# ---------------------------------------------------------------------------


class RouteGroup:
    """Compose the focused routes under the ``/v1`` prefix.

    Attributes:
        router: The :class:`APIRouter` carrying the v1 routes.
        admin_router: The :class:`APIRouter` mounted under
            ``/v1/admin`` for admin-only endpoints.
        preferences_router: The :class:`APIRouter` mounted under
            ``/v1`` for the per-user preferences endpoints.

    """

    def __init__(self) -> None:
        """Construct the routes and compose them."""
        health = HealthRoute()
        auth = AuthRoute()
        documents = DocumentRoute()
        query = QueryRoute()
        feedback = FeedbackRoute()
        admin = AdminRoute()
        preferences = PreferenceRoute()
        self.router = APIRouter()
        for sub in (health.router, auth.router, documents.router, query.router, feedback.router):
            self.router.include_router(sub)
        self.admin_router = admin.router
        self.preferences_router = preferences.router

    def register_all(self, app: FastAPI, prefix: str) -> None:
        """Mount the v1, admin, and preferences routes under ``prefix``.

        Args:
            app: The FastAPI instance.
            prefix: The URL prefix (e.g. ``"/v1"``). The admin route
                carries its own ``/admin`` segment on top of this
                prefix, so admin endpoints land at ``/v1/admin/*``.

        """
        app.include_router(self.router, prefix=prefix)
        app.include_router(self.admin_router, prefix=prefix)
        app.include_router(self.preferences_router, prefix=prefix)
