"""api package.

Implementation lives in :mod:`raghub.helper` (auth, sse, response, rate_limit); local entry-point modules: ['app'].
"""

from __future__ import annotations

# --- app.py content ---
import importlib.metadata
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar

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

from raghub.exceptions import (
    AuthenticationError,
    AuthorizationError,
    DocumentError,
    StorageError,
)
from raghub.helper.auth import (
    App,
    Auth,
    Bearer,
)
from raghub.helper.rate_limit import (
    RateLimiterMiddleware,
    TokenBucket,
)
from raghub.helper.response import (
    Redaction,
    ResponseBuilder,
)
from raghub.helper.sse import (
    Sse,
)
from raghub.ingestion import BackgroundIngestionService
from raghub.models import (
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestItem,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    UserPrincipal,
)
from raghub.services import Facade as Facade
from raghub.utils import capture

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


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


def validate_cors_for_credentials(origins: list[str]) -> None:
    """Refuse to start with ``allow_credentials=True`` + wildcard origins.

    Browsers reject ``Access-Control-Allow-Origin: *`` together with
    credentials, and FastAPI's CORS middleware silently passes the
    configuration through to the response. We catch the misconfiguration
    here so misdeployments fail loud instead of returning headers
    browsers discard.

    Args:
        origins: The list of origins the middleware would advertise.

    Raises:
        RuntimeError: When ``origins`` contains ``"*"``.
    """
    if any(origin == "*" for origin in origins):
        raise RuntimeError(
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
    container: Any,
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
        def authentication_error_handler(_: Any, exc: AuthenticationError) -> JSONResponse:
            """Return 401 for any :class:`AuthenticationError`."""
            return JSONResponse(status_code=401, content={"detail": str(exc)})

        @app.exception_handler(AuthorizationError)
        def authorization_error_handler(_: Any, exc: AuthorizationError) -> JSONResponse:
            """Return 403 for any :class:`AuthorizationError`."""
            return JSONResponse(status_code=403, content={"detail": str(exc)})

        @app.exception_handler(DocumentError)
        def document_error_handler(_: Any, exc: DocumentError) -> JSONResponse:
            """Return 400 for any :class:`DocumentError`."""
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        @app.exception_handler(StorageError)
        def storage_error_handler(_: Any, exc: StorageError) -> JSONResponse:
            """Return 500 for any :class:`StorageError`."""
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


# ---------------------------------------------------------------------------
# Versioned routes (v1) + admin + preferences
# ---------------------------------------------------------------------------


class RouteGroup:
    """Register the ``/v1/*`` route group plus the admin and preferences sub-routers.

    Centralises every route definition so :func:`create_app` stays a
    pure wiring step. ``__init__`` triggers :meth:`build`, so a fresh
    :class:`RouteGroup` always has every route already decorated onto
    its three routers. :meth:`register_all` is the only step that
    needs the live :class:`FastAPI` instance.

    Attributes:
        router: The :class:`APIRouter` carrying the v1 routes.
        admin_router: The :class:`APIRouter` mounted under
            ``/v1/admin`` for admin-only endpoints.
        preferences_router: The :class:`APIRouter` mounted under
            ``/v1`` for the per-user preferences endpoints.
    """

    def __init__(self) -> None:
        """Create the routers and decorate every route on them."""
        self.router = APIRouter()
        self.admin_router = APIRouter(prefix="/admin", tags=["admin"])
        self.preferences_router = APIRouter()
        self.build()

    # ----- v1: auth / session -----------------------------------------

    def health(self) -> Callable[..., Any]:
        """Liveness probe; delegates to :meth:`Facade.health`."""

        @self.router.get("/health")
        def handler(
            app_service: Facade = Depends(App.get),
        ) -> dict[str, Any]:
            """Report liveness."""
            return app_service.health()

        return handler

    def login(self) -> Callable[..., Any]:
        """Authenticate a user and return a session token."""

        @self.router.post("/auth/login", response_model=AuthLoginResponse)
        async def handler(
            payload: AuthLoginRequest,
            app_service: Facade = Depends(App.get),
        ) -> AuthLoginResponse:
            return await app_service.login(payload.email, payload.password)

        return handler

    def logout(self) -> Callable[..., Any]:
        """Invalidate the bearer token presented in the ``Authorization`` header."""

        @self.router.post("/auth/logout")
        async def handler(
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> dict[str, str]:
            await app_service.logout(token)
            return {"status": "logged_out"}

        return handler

    def session_history(self) -> Callable[..., Any]:
        """Return the conversation history for the current session."""

        @self.router.get("/session/history")
        async def handler(
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> dict[str, list[dict[str, Any]]]:
            history = await app_service.history(token)
            return {"history": [turn.model_dump(mode="json") for turn in history]}

        return handler

    def clear_history(self) -> Callable[..., Any]:
        """Empty the conversation history for the current session."""

        @self.router.delete("/session/history", status_code=204, response_class=Response)
        async def handler(
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> Response:
            await app_service.clear_history(token)
            return Response(status_code=204)

        return handler

    # ----- v1: documents ----------------------------------------------

    def upload_document(self) -> Callable[..., Any]:
        """Upload a PDF document and synchronously index it."""

        @self.router.post("/documents/upload", status_code=202, response_model=DocumentUploadResponse)
        async def handler(
            request: Request,
            file: UploadFile = File(...),
            company: str | None = Form(default=None),
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
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
                document_id=document.document_id,
                version=document.version,
                status=document.status.value,
                company=document.organization,
                filename=document.filename,
            )

        return handler

    def ingest_documents_batch(self) -> Callable[..., Any]:
        """Ingest multiple documents in a single request.

        Accepts one or more files as multipart upload. Each file is
        ingested independently; a failure in one does not affect the
        others. This is a pipeline boundary method — failures on
        individual files are captured as :class:`BatchIngestItem`
        entries and the surrounding batch loop continues.
        """

        @self.router.post(
            "/documents/ingest/batch",
            status_code=200,
            response_model=BatchIngestResponse,
        )
        async def handler(
            request: Request,
            files: list[UploadFile] = File(...),
            company: str | None = Form(default=None),
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
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
                            document_id=document.document_id,
                            status="ok",
                        )
                    )
                except (DocumentError, StorageError, ValueError, TypeError, OSError) as exc:
                    loguru_logger.warning("api.batch_upload.item_failed", file=file.filename, error=str(exc))
                    results.append(
                        BatchIngestItem(
                            filename=file.filename or "upload.pdf",
                            status="error",
                            error=str(exc),
                        )
                    )
            return BatchIngestResponse(documents=results)

        return handler

    def list_documents(self) -> Callable[..., Any]:
        """List the documents visible to the calling user."""

        @self.router.get("/documents")
        async def handler(
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> dict[str, list[dict[str, Any]]]:
            documents = await app_service.list_documents(token)
            return {"documents": [document.model_dump(mode="json") for document in documents]}

        return handler

    def document_status(self) -> Callable[..., Any]:
        """Return the latest status for a single document."""

        @self.router.get("/documents/{document_id}/status")
        async def handler(
            document_id: str,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> dict[str, Any]:
            document = await app_service.document_status(token, document_id)
            return document.model_dump(mode="json")

        return handler

    def delete_document(self) -> Callable[..., Any]:
        """Delete a document and all of its chunks. Admin-only."""

        @self.router.delete("/documents/{document_id}", status_code=204, response_class=Response)
        async def handler(
            document_id: str,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> Response:
            await app_service.delete_document(token, document_id)
            return Response(status_code=204)

        return handler

    # ----- v1: query ---------------------------------------------------

    def query(self) -> Callable[..., Any]:
        """Answer a question using the application service.

        Advanced-RAG flags (``agent``, ``web``, ``tools_enabled``
        etc.) are forwarded to the resolver when any of them are
        supplied; otherwise the early-exit path runs.
        """

        @self.router.post("/query", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> QueryResponse:
            if payload.tools_enabled is None and payload.agent is None and payload.web is None \
                    and payload.graph is None and payload.summaries is None \
                    and payload.reranker is None and payload.long_context_pass is None \
                    and payload.query_transforms is None and payload.max_steps is None \
                    and payload.top_k is None:
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

        return handler

    def ingest_async(self) -> Callable[..., Any]:
        """Queue a document for asynchronous ingestion."""

        @self.router.post("/ingest/async")
        async def handler(
            request: Request,
            file: UploadFile = File(...),
            company: str | None = Form(default=None),
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
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

        return handler

    # ----- v1: streaming ----------------------------------------------

    def query_stream(self) -> Callable[..., Any]:
        """Stream agent / planner events as Server-Sent Events."""

        @self.router.post("/query/stream")
        async def handler(
            payload: QueryRequest,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> StreamingResponse:
            resolved_tools = (
                set(payload.tools_enabled) if payload.tools_enabled else set()
            )

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

        return handler

    def agent_run(self) -> Callable[..., Any]:
        """Run the agent end-to-end and return the full :class:`QueryResponse`."""

        @self.router.post("/agent/run", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
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

        return handler

    # ----- v1/admin ---------------------------------------------------

    def admin_documents(self) -> Callable[..., Any]:
        """Admin-only: list every document in the registry."""

        @self.admin_router.get("/documents")
        async def handler(
            _admin: UserPrincipal = Depends(Auth.admin),
            app_service: Facade = Depends(App.get),
        ) -> list[dict[str, Any]]:
            docs = await app_service.container.uow.document_repo.list_all()
            return [doc.model_dump(mode="json") for doc in docs]

        return handler

    def admin_users(self) -> Callable[..., Any]:
        """Admin-only: list every user in the user store (sensitive fields redacted)."""

        @self.admin_router.get("/users")
        async def handler(
            _admin: UserPrincipal = Depends(Auth.admin),
            app_service: Facade = Depends(App.get),
        ) -> list[dict[str, Any]]:
            users = await app_service.container.user_store.list_users()
            return [Redaction.user(user.model_dump(mode="json")) for user in users]

        return handler

    def admin_stats(self) -> Callable[..., Any]:
        """Admin-only: high-level system counters."""

        @self.admin_router.get("/stats")
        async def handler(
            _admin: UserPrincipal = Depends(Auth.admin),
            app_service: Facade = Depends(App.get),
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

        return handler

    # ----- v1/users/me/preferences ------------------------------------

    def preferences_get(self) -> Callable[..., Any]:
        """Return every stored preference for the authenticated user."""

        @self.preferences_router.get(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> PreferencesResponse:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

        return handler

    def preferences_patch(self) -> Callable[..., Any]:
        """Upsert one or more preferences for the authenticated user."""

        @self.preferences_router.patch(
            "/users/me/preferences",
            response_model=PreferencesResponse,
        )
        async def handler(
            payload: PreferencesPatch,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> PreferencesResponse:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            await store.set_prefs(user_id, dict(payload.prefs or {}))
            prefs = await store.get_prefs(user_id)
            return PreferencesResponse(prefs=prefs or {})

        return handler

    def preferences_delete(self) -> Callable[..., Any]:
        """Delete a single preference by key."""

        @self.preferences_router.delete(
            "/users/me/preferences/{key}",
            status_code=204,
        )
        async def handler(
            key: str,
            token: str = Depends(Bearer.dependency),
            app_service: Facade = Depends(App.get),
        ) -> None:
            user_id = await Auth.user_id(app_service, token)
            store = user_store_or_503(app_service)
            await store.delete_pref(user_id, key)

        return handler

    # ----- wiring -----------------------------------------------------

    def build(self) -> None:
        """Decorate every route on the three routers.

        Called once from :meth:`__init__`. Public so that tests can
        re-decorate on a fresh router if they need to.
        """
        for builder in (
            self.health,
            self.login,
            self.logout,
            self.session_history,
            self.clear_history,
            self.upload_document,
            self.ingest_documents_batch,
            self.list_documents,
            self.document_status,
            self.delete_document,
            self.query,
            self.ingest_async,
            self.query_stream,
            self.agent_run,
            self.admin_documents,
            self.admin_users,
            self.admin_stats,
            self.preferences_get,
            self.preferences_patch,
            self.preferences_delete,
        ):
            builder()

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

    metrics = getattr(application.container, "metrics", None)
    register = getattr(metrics, "register_app", None)
    if callable(register):
        register(app)

    app.state.background_ingestion = BackgroundIngestionService(max_workers=2)

    cors_origins = cors_origins_from_env()
    validate_cors_for_credentials(cors_origins)
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
        """Build the app via :func:`build_application` if not cached.

        Returns:
            The cached :class:`FastAPI` instance.
        """
        if cls.instance is None:
            cls.instance = cls()
        if cls.instance.cached is None:
            import asyncio

            from raghub.core import build_application

            application = asyncio.run(build_application())
            cls.instance.cached = create_app(application)
        return cls.instance.cached

    @classmethod
    def reset(cls) -> None:
        """Drop the cached :class:`FastAPI` so the next build is fresh."""
        if cls.instance is not None:
            cls.instance.cached = None


app_singleton = AppFactory()




__all__ = ['App', 'Auth', 'Bearer', 'RateLimiterMiddleware', 'Redaction', 'ResponseBuilder', 'Sse', 'TokenBucket']
