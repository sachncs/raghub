"""FastAPI surface for the RAGHub framework.

Defines the :func:`create_app` factory, the :class:`RouteGroup` that
mounts every ``/v1/*`` route, the API-level exception handlers, the
lifespan coordinator, and the CORS / upload-guard helpers.

Auth-side helpers (``App``, ``Auth``, ``Bearer``) live in
:mod:`raghub.api_auth`; response shaping and rate-limiting live in
:mod:`raghub.api_response` and :mod:`raghub.api_ratelimit`;
streaming helpers live in :mod:`raghub.api_sse`.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, ClassVar

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger as loguru_logger
from pydantic import BaseModel, Field

from raghub.api_auth import (
    App,
    Auth,
    Bearer,
)
from raghub.api_ratelimit import RateLimiterMiddleware
from raghub.api_response import Redaction
from raghub.api_sse import (
    Sse,
)
from raghub.await_sync import capture
from raghub.errors import (
    AuthenticationError,
    AuthorizationError,
    IngestionError,
    RagHubError,
)
from raghub.ingest import Batch
from raghub.models import (
    QueryRequest,
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestItem,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    User,
)
from raghub.services import Facade as Facade

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


__all__ = [
    "AdminRouter",
    "AppFactory",
    "AuthRouter",
    "DocumentRouter",
    "ExceptionHandlers",
    "FeedbackAggregateResponse",
    "FeedbackRouter",
    "FeedbackSubmission",
    "HealthRouter",
    "Lifespan",
    "PreferencesPatch",
    "PreferencesResponse",
    "PreferencesRouter",
    "QueryRouter",
    "RouteGroup",
    "check_upload_size",
    "cors_origins_from_env",
    "create_app",
    "enforce_upload_limit",
    "package_metadata",
    "root_health_route",
    "upload_content_length",
    "user_store_or_503",
    "validate_cors",
]


def cors_origins_from_env() -> list[str]:
    """Return the parsed CORS_ORIGINS list.

    Reads ``CORS_ORIGINS`` (comma-separated). Falls back to a sane
    default of ``["*"]`` for development convenience; production
    deployments must override the env var.
    """
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


def validate_cors(origins: list[str]) -> None:
    """Refuse to start with ``allow_credentials=True`` + wildcard origins.

    Browsers reject ``Access-Control-Allow-Origin: *`` together with
    credentials, and FastAPI's CORS middleware silently passes the
    configuration through to the response. We catch the misconfiguration
    here so misdeployments fail loud instead of returning headers
    browsers discard.

    Args:
        origins: The list of origins the middleware would advertise.

    Raises:
        ConfigurationError: When ``origins`` contains ``"*"``.

    """
    if any(origin == "*" for origin in origins):
        from raghub.errors import ConfigurationError

        raise ConfigurationError(
            "CORS_ORIGINS='*' is incompatible with allow_credentials=True; "
            "set CORS_ORIGINS to a comma-separated list of explicit origins."
        )


# ---------------------------------------------------------------------------
# Upload guards
# ---------------------------------------------------------------------------


def check_upload_size(content_length: int | None, max_bytes: int) -> bool:
    """Pre-flight guard for upload size.

    Called by upload endpoints with the value of the ``Content-Length``
    request header before the multipart body is read into memory.
    Returning ``True`` causes the caller to raise HTTP 413; returning
    ``False`` allows the upload to proceed (a second check after
    reading catches chunked-transfer uploads that omit the header).

    Args:
        content_length: The value of the request's ``Content-Length``
            header. ``None`` when the client did not send the header
            (chunked transfer encoding); in that case the function
            returns ``False`` so the post-read check fires.
        max_bytes: The configured maximum accepted upload size.

    Returns:
        ``True`` when the declared upload is over the limit; ``False``
        otherwise.

    """
    if content_length is None:
        return False
    return content_length > max_bytes


def upload_content_length(request: Request) -> int | None:
    """Return the parsed ``Content-Length`` header or ``None``.

    Args:
        request: The incoming request.

    Returns:
        The integer value, or ``None`` when the header is missing or
        cannot be parsed as an integer.

    """
    declared = request.headers.get("content-length")
    if declared is None:
        return None
    value, _ = capture(int, declared)
    return value if isinstance(value, int) else None


def enforce_upload_limit(
    request: Request,
    container: "RagContainer",
    payload: bytes | None = None,
) -> None:
    """Raise HTTP 413 when ``request`` (or ``payload``) exceeds the limit.

    Args:
        request: The incoming request (used to read ``Content-Length``).
        container: The application container holding ``settings``.
        payload: Optional in-memory payload. When provided, the
            post-read check runs against the actual bytes.

    Raises:
        HTTPException: 413 when the upload exceeds ``max_upload_bytes``.

    """
    max_bytes = int(getattr(container.settings, "max_upload_bytes", 0) or 0)
    if max_bytes <= 0:
        return
    if check_upload_size(upload_content_length(request), max_bytes):
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )
    if payload is not None and len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum size of {max_bytes} bytes",
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class Lifespan:
    """FastAPI startup/shutdown coordinator.

    The class is instantiated with the :class:`Facade`
    facade and wired into the :class:`FastAPI` instance. On shutdown
    it calls :meth:`Facade.shutdown` and then closes
    the shared background-ingestion service.

    Attributes:
        application: The application facade.

    """

    def __init__(self, application: Facade) -> None:
        """Store the application facade for the lifespan handlers."""
        self.application = application

    @asynccontextmanager
    async def __call__(self, app: FastAPI) -> AsyncIterator[None]:
        """Drive the FastAPI lifespan protocol.

        Args:
            app: The FastAPI instance whose ``state`` carries the
                application facade and the background-ingestion pool.

        Yields:
            Nothing; the context manager signals lifecycle transitions
            to FastAPI.

        """
        try:
            yield
        finally:
            shutdown_app = getattr(self.application, "shutdown", None)
            if shutdown_app is not None:
                try:
                    await shutdown_app()
                except (RuntimeError, OSError, ConnectionError, TimeoutError) as exc:
                    loguru_logger.warning("api.shutdown.failed", error=str(exc))
            background = getattr(app.state, "background_ingestion", None)
            if background is not None and hasattr(background, "shutdown"):
                try:
                    background.shutdown()
                except (RuntimeError, OSError, ConnectionError) as exc:
                    loguru_logger.warning("background.shutdown.failed", error=str(exc))


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


class ExceptionHandlers:
    """Install typed-exception → HTTP response handlers on the app."""

    @staticmethod
    def install(app: FastAPI) -> None:
        """Register exception handlers on ``app``.

        Args:
            app: The FastAPI instance.

        """

        @app.exception_handler(AuthenticationError)
        def auth_error(_: Any, exc: AuthenticationError) -> JSONResponse:
            """Return 401 for any :class:`AuthenticationError`."""
            return JSONResponse(status_code=401, content={"detail": str(exc)})

        @app.exception_handler(AuthorizationError)
        def authz_error(_: Any, exc: AuthorizationError) -> JSONResponse:
            """Return 403 for any :class:`AuthorizationError`."""
            return JSONResponse(status_code=403, content={"detail": str(exc)})

        @app.exception_handler(IngestionError)
        def ingestion_error_handler(_: Any, exc: IngestionError) -> JSONResponse:
            """Return 400 for any :class:`IngestionError`."""
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        @app.exception_handler(RagHubError)
        def generic_error_handler(_: Any, exc: RagHubError) -> JSONResponse:
            """Return 500 for any uncategorised :class:`RagHubError`."""
            return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Preferences request/response models
# ---------------------------------------------------------------------------


class PreferencesResponse(BaseModel):
    """Preferences for the authenticated user.

    Attributes:
        prefs: Mapping of preference key → JSON value. The reserved
            key ``"tool_settings"`` carries the ChatGPT-style tool
            toggles consumed by :func:`raghub.agent.resolve`.

    """

    prefs: dict[str, Any] = Field(default_factory=dict)


class PreferencesPatch(BaseModel):
    """Preferences update payload.

    Attributes:
        prefs: Replacement mapping for the supplied keys. Keys absent
            from the payload are left unchanged on disk.

    """

    prefs: dict[str, Any] = Field(default_factory=dict)


def user_store_or_503(app_service: Facade) -> Any:
    """Return the configured user store or raise 503."""
    store = getattr(app_service.container, "user_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="user store unavailable")
    if not hasattr(store, "get_pref") or not hasattr(store, "set_pref"):
        raise HTTPException(status_code=503, detail="user store lacks prefs API")
    return store


def query_request_has_flags(payload: "QueryRequest") -> bool:
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
# Versioned routes (v1) + admin + preferences
# ---------------------------------------------------------------------------


class HealthRouter:
    """Health probe endpoint."""

    router: APIRouter

    def __init__(self) -> None:
        """Build the router."""
        self.router = APIRouter()
        self.__register()

    def __register(self) -> None:
        @self.router.get("/health")
        def handler(
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            """Report liveness."""
            return app_service.health()


class AuthRouter:
    """Auth and session-history endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter()
        self._register_login()
        self._register_logout()
        self._register_session_history()
        self._register_clear_history()

    def _register_login(self) -> None:
        @self.router.post("/auth/login", response_model=AuthLoginResponse)
        async def handler(
            payload: AuthLoginRequest,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> AuthLoginResponse:
            return await app_service.login(payload.email, payload.password)

    def _register_logout(self) -> None:
        @self.router.post("/auth/logout")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            await app_service.logout(token)
            return {"status": "logged_out"}

    def _register_session_history(self) -> None:
        @self.router.get("/session/history")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, list[dict[str, Any]]]:
            history = await app_service.history(token)
            return {"history": [turn.model_dump(mode="json") for turn in history]}

    def _register_clear_history(self) -> None:
        @self.router.delete("/session/history", status_code=204, response_class=Response)
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            await app_service.clear_history(token)
            return Response(status_code=204)


class DocumentRouter:
    """Document upload, listing, status, and delete endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter()
        self._register_upload()
        self._register_ingest_batch()
        self._register_list()
        self._register_status()
        self._register_delete()
        self._register_ingest_async()

    def _register_upload(self) -> None:
        @self.router.post(
            "/documents/upload", status_code=202, response_model=DocumentUploadResponse
        )
        async def handler(
            request: Request,
            file: Annotated[UploadFile, File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> DocumentUploadResponse:
            enforce_upload_limit(request, app_service.container)
            content = await file.read()
            enforce_upload_limit(request, app_service.container, payload=content)
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

    def _register_ingest_batch(self) -> None:
        @self.router.post(
            "/documents/ingest/batch",
            status_code=200,
            response_model=BatchIngestResponse,
        )
        async def handler(
            request: Request,
            files: Annotated[list[UploadFile], File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> BatchIngestResponse:
            enforce_upload_limit(request, app_service.container)
            results: list[BatchIngestItem] = []
            for file in files:
                try:
                    content = await file.read()
                    max_bytes = int(
                        getattr(app_service.container.settings, "max_upload_bytes", 0) or 0
                    )
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
                    loguru_logger.warning(
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

    def _register_list(self) -> None:
        @self.router.get("/documents")
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, list[dict[str, Any]]]:
            documents = await app_service.list_documents(token)
            return {
                "documents": [document.model_dump(mode="json") for document in documents]
            }

    def _register_status(self) -> None:
        @self.router.get("/documents/{document_id}/status")
        async def handler(
            document_id: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            document = await app_service.document_status(token, document_id)
            return document.model_dump(mode="json")

    def _register_delete(self) -> None:
        @self.router.delete(
            "/documents/{document_id}", status_code=204, response_class=Response
        )
        async def handler(
            document_id: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            await app_service.delete_document(token, document_id)
            return Response(status_code=204)

    def _register_ingest_async(self) -> None:
        @self.router.post("/ingest/async")
        async def handler(
            request: Request,
            file: Annotated[UploadFile, File(...)],
            company: Annotated[str | None, Form(default=None)],
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            enforce_upload_limit(request, app_service.container)
            content = await file.read()
            enforce_upload_limit(request, app_service.container, payload=content)
            background = request.app.state.background_ingestion
            job_id = background.submit(
                app_service.upload_document,
                token=token,
                filename=file.filename or "upload.pdf",
                content=content,
                company=company,
            )
            return {"job_id": job_id}


class QueryRouter:
    """Synchronous and streaming query endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter()
        self._register_query()
        self._register_stream()
        self._register_agent_run()

    def _register_query(self) -> None:
        @self.router.post("/query", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> QueryResponse:
            if not query_request_has_flags(payload):
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

    def _register_stream(self) -> None:
        @self.router.post("/query/stream")
        def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> StreamingResponse:
            resolved_tools = set(payload.tools_enabled) if payload.tools_enabled else set()

            async def gen() -> AsyncIterator[bytes]:
                yield Sse.comment("raghub-query-stream")
                user, _ = await app_service.auth_svc.resolve_user(token)
                rag = app_service.container.rag_facade
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
                    yield Sse.format(event.kind, event.model_dump(mode="json"))

            return StreamingResponse(gen(), media_type="text/event-stream")

    def _register_agent_run(self) -> None:
        @self.router.post("/agent/run", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> QueryResponse:
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


class AdminRouter:
    """Admin-only endpoints mounted under ``/admin``."""

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter(prefix="/admin", tags=["admin"])
        self._register_documents()
        self._register_users()
        self._register_stats()

    def _register_documents(self) -> None:
        @self.router.get("/documents")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> list[dict[str, Any]]:
            docs = await app_service.container.uow.document_repo.list_all()
            return [doc.model_dump(mode="json") for doc in docs]

    def _register_users(self) -> None:
        @self.router.get("/users")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> list[dict[str, Any]]:
            users = await app_service.container.user_store.list_users()
            return [Redaction.user(user.model_dump(mode="json")) for user in users]

    def _register_stats(self) -> None:
        @self.router.get("/stats")
        async def handler(
            admin_user: Annotated[User, Depends(Auth.admin)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            docs = await app_service.container.uow.document_repo.list_all()
            users = await app_service.container.user_store.list_users()
            vector_health = app_service.container.vector_store.health()
            chunk_count = vector_health.get("chunks", 0)
            return {
                "document_count": len(docs),
                "user_count": len(users),
                "chunk_count": chunk_count,
                "vector_store_size": vector_health.get("size", "unknown"),
            }


class PreferencesRouter:
    """Per-user preferences endpoints."""

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter()
        self._register_get()
        self._register_patch()
        self._register_delete()

    def _register_get(self) -> None:
        @self.router.get(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> PreferencesResponse:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

    def _register_patch(self) -> None:
        @self.router.patch(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            payload: PreferencesPatch,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> PreferencesResponse:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            await store.set_prefs(user_id, dict(payload.prefs or {}))
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

    def _register_delete(self) -> None:
        @self.router.delete(
            "/users/me/preferences/{key}",
            status_code=204,
        )
        async def handler(
            key: str,
            token: Annotated[str, Depends(Bearer.dependency)],
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> None:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            await store.delete_pref(user_id, key)


class FeedbackSubmission(BaseModel):
    """Inbound feedback payload."""

    session_id: str
    query_id: str
    chunk_id: str | None = None
    answer_id: str | None = None
    rating: int  # -1, 0, or 1
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackAggregateResponse(BaseModel):
    """Aggregate counts response."""

    tenant_id: str | None
    positive: int
    negative: int
    neutral: int
    by_chunk: dict[str, int]


class FeedbackRouter:
    """Feedback capture endpoints (Tier 3 Item 18).

    Endpoints:
        - ``POST /feedback`` — record feedback
        - ``GET /feedback/{id}`` — retrieve one record
        - ``DELETE /feedback/{id}`` — delete one record
        - ``GET /feedback/aggregate?tenant_id=…`` — aggregate counts
    """

    router: APIRouter

    def __init__(self) -> None:
        self.router = APIRouter()
        # Register aggregate BEFORE the {feedback_id} catch-all so
        # ``/feedback/aggregate`` does not get treated as a lookup for
        # feedback id "aggregate".
        self._register_aggregate()
        self._register_submit()
        self._register_get()
        self._register_delete()

    def __feedback_store(self, app_service: Facade) -> Any:
        """Return the configured FeedbackStore or raise ``503`` if absent."""
        from fastapi import HTTPException

        store = getattr(app_service.container, "feedback_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="feedback_store not configured",
            )
        return store

    def _register_submit(self) -> None:
        @self.router.post(
            "/feedback",
            status_code=201,
        )
        async def handler(
            payload: FeedbackSubmission,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, str]:
            from raghub.feedback import Feedback, Rating, new_feedback_id, now_utc

            store = self.__feedback_store(app_service)
            feedback = Feedback(
                id=new_feedback_id(),
                session_id=payload.session_id,
                query_id=payload.query_id,
                chunk_id=payload.chunk_id,
                answer_id=payload.answer_id,
                user_id="anonymous",
                tenant_id="default",
                rating=Rating(payload.rating),
                comment=payload.comment,
                created_at=now_utc(),
                metadata=payload.metadata,
            )
            await store.record(feedback)
            return {"id": feedback.id}

    def _register_get(self) -> None:
        @self.router.get("/feedback/{feedback_id}")
        async def handler(
            feedback_id: str,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> dict[str, Any]:
            from fastapi import HTTPException

            store = self.__feedback_store(app_service)
            feedback = await store.get(feedback_id)
            if feedback is None:
                raise HTTPException(status_code=404, detail="feedback not found")
            from dataclasses import asdict

            return asdict(feedback)

    def _register_delete(self) -> None:
        @self.router.delete("/feedback/{feedback_id}", status_code=204)
        async def handler(
            feedback_id: str,
            app_service: Annotated[Facade, Depends(App.get)],
        ) -> Response:
            store = self.__feedback_store(app_service)
            await store.delete(feedback_id)
            return Response(status_code=204)

    def _register_aggregate(self) -> None:
        @self.router.get(
            "/feedback/aggregate",
            response_model=FeedbackAggregateResponse,
        )
        async def handler(
            app_service: Annotated[Facade, Depends(App.get)],
            tenant_id: str | None = None,
        ) -> FeedbackAggregateResponse:
            store = self.__feedback_store(app_service)
            aggregate = await store.aggregate(tenant_id)
            return FeedbackAggregateResponse(
                tenant_id=aggregate.tenant_id,
                positive=aggregate.positive,
                negative=aggregate.negative,
                neutral=aggregate.neutral,
                by_chunk=aggregate.by_chunk,
            )


class RouteGroup:
    """Compose the focused routers under the ``/v1`` prefix.

    Attributes:
        router: The :class:`APIRouter` carrying the v1 routes.
        admin_router: The :class:`APIRouter` mounted under
            ``/v1/admin`` for admin-only endpoints.
        preferences_router: The :class:`APIRouter` mounted under
            ``/v1`` for the per-user preferences endpoints.

    """

    def __init__(self) -> None:
        """Construct the routers and compose them."""
        health = HealthRouter()
        auth = AuthRouter()
        documents = DocumentRouter()
        query = QueryRouter()
        feedback = FeedbackRouter()
        admin = AdminRouter()
        preferences = PreferencesRouter()
        self.router = APIRouter()
        for sub in (health.router, auth.router, documents.router, query.router, feedback.router):
            self.router.include_router(sub)
        self.admin_router = admin.router
        self.preferences_router = preferences.router

    def register_all(self, app: FastAPI, prefix: str) -> None:
        """Mount the v1, admin, and preferences routers under ``prefix``.

        Args:
            app: The FastAPI instance.
            prefix: The URL prefix (e.g. ``"/v1"``). The admin router
                carries its own ``/admin`` segment on top of this
                prefix, so admin endpoints land at ``/v1/admin/*``.

        """
        app.include_router(self.router, prefix=prefix)
        app.include_router(self.admin_router, prefix=prefix)
        app.include_router(self.preferences_router, prefix=prefix)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def package_metadata() -> tuple[str, str, str]:
    """Return ``(name, version, summary)`` for the raghub distribution.

    Falls back to a static default when the package is not installed
    in editable mode and the metadata lookup fails.

    Returns:
        A 3-tuple of ``(title, version, description)``.

    """
    pkg, error = capture(importlib.metadata.metadata, "raghub")
    if error is not None or not isinstance(pkg, dict):
        return (
            "RAGHub",
            "0.3.3",
            "RAGHub — production-grade multi-user retrieval-augmented generation platform",
        )
    return (
        pkg["Name"].replace("-", " ").title(),
        pkg["Version"],
        pkg.get(
            "Summary",
            "RAGHub — production-grade multi-user retrieval-augmented generation platform",
        ),
    )


def root_health_route(app: FastAPI) -> None:
    """Mount an unversioned liveness probe for orchestrator health checks.

    Args:
        app: The FastAPI instance.

    """

    @app.get("/health", include_in_schema=False)
    def handler() -> dict[str, str]:
        """Mirror ``GET /v1/health`` so Docker/Kubernetes probes skip the prefix."""
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_app(application: Facade) -> FastAPI:
    """Build a :class:`FastAPI` instance wired to ``application``.

    Args:
        application: The pre-wired application facade.

    Returns:
        A fully-configured FastAPI app ready to be served by
        ``uvicorn`` or any ASGI server.

    Raises:
        RuntimeError: When CORS configuration is invalid (wildcard
            origins with credentials).

    """
    title, version, description = package_metadata()
    app = FastAPI(
        title=title, version=version, description=description, lifespan=Lifespan(application)
    )
    app.state.application = application

    app.state.background_ingestion = Batch(max_workers=2)

    cors_origins = cors_origins_from_env()
    validate_cors(cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimiterMiddleware, rate=10.0, burst=20)

    ExceptionHandlers.install(app)
    RouteGroup().register_all(app, prefix="/v1")
    root_health_route(app)
    return app


# ---------------------------------------------------------------------------
# Factory (replaces module-level singleton)
# ---------------------------------------------------------------------------


class AppFactory:
    """Encapsulates the lazily-built singleton :class:`FastAPI`.

    Replaces the prior module-level ``app_singleton`` global. The
    factory holds a single class-level :class:`AppFactory` instance
    whose ``cached`` attribute is populated on the first call to
    :meth:`create_app`. Tests can reset the cache via
    :meth:`reset` to force a rebuild.
    """

    instance: ClassVar[AppFactory | None] = None

    def __init__(self) -> None:
        """Store an empty cache."""
        self.cached: FastAPI | None = None

    @classmethod
    def create_app(cls) -> FastAPI:
        """Build the app via :func:`build_container` if not cached.

        Returns:
            The cached :class:`FastAPI` instance.

        """
        if cls.instance is None:
            cls.instance = cls()
        if cls.instance.cached is None:
            import asyncio

            from raghub.config import Settings
            from raghub.services import Facade, build_container

            settings = Settings.load()
            container = asyncio.run(build_container(settings))
            application = Facade(container)
            cls.instance.cached = create_app(application)
        return cls.instance.cached

    @classmethod
    def reset(cls) -> None:
        """Drop the cached :class:`FastAPI` so the next build is fresh."""
        if cls.instance is not None:
            cls.instance.cached = None


app_singleton = AppFactory()
