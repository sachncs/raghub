"""FastAPI reference server.

Defines :func:`create_app`, which builds a :class:`FastAPI` instance
bound to a fully-wired :class:`DynamicRagApplication`. The factory
wires in:

* CORS middleware (origins from ``CORS_ORIGINS`` env, comma-separated).
  Wildcard origins combined with credentials are refused at startup
  because browsers reject the combination at runtime.
* The :class:`RateLimiterMiddleware` (default 10 rps, burst 20).
* The admin router from :mod:`raghub.api.admin`.
* Exception handlers for ``AuthenticationError`` (401),
  ``AuthorizationError`` (403), ``DocumentError`` (400), and
  ``StorageError`` (500).
* A ``/metrics`` Prometheus endpoint registered via the
  :class:`raghub.observability.PrometheusMetrics` instance
  shared with the application container.
* A shared :class:`BackgroundIngestionService` placed on
  ``app.state.background_ingestion`` for the ``/ingest/async`` endpoint.

Also exposes :func:`require_bearer` (used by routes to extract the
bearer token from the ``Authorization`` header), :func:`check_upload_size`
(a pre-flight guard that rejects oversize uploads with HTTP 413
before the multipart body is read into memory), and :func:`get_app`
(a lazy singleton convenience used by tooling that needs the app
without going through the FastAPI CLI).

Section map:

* :class:`Lifespan` — FastAPI startup/shutdown context.
* :func:`cors_origins_from_env` / :func:`validate_cors_for_credentials`
  — CORS configuration helpers.
* :func:`check_upload_size` — pre-flight upload size guard.
* :func:`_upload_content_length` — helper for parsing the
  ``Content-Length`` header safely.
* :class:`ExceptionHandlers` — installs handlers for the typed
  application errors.
* :class:`RouteGroup` — registers the versioned ``/v1/*`` routes.
* :func:`create_app` — the public factory.
"""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from raghub.api.admin import router as admin_router
from raghub.api.dependencies import get_application
from raghub.api.preferences import router as preferences_router
from raghub.api.rate_limiter import RateLimiterMiddleware
from raghub.api.streaming import sse_comment, sse_format
from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError, StorageError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.models.api import (
    AuthLoginRequest,
    AuthLoginResponse,
    BatchIngestItem,
    BatchIngestResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
)
from raghub.services.application import DynamicRagApplication


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
    try:
        return int(declared)
    except ValueError:
        return None


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

    The class is instantiated with the :class:`DynamicRagApplication`
    facade and wired into the :class:`FastAPI` instance. On shutdown
    it calls :meth:`DynamicRagApplication.shutdown` and then closes
    the shared background-ingestion service.

    Attributes:
        application: The application facade.
    """

    def __init__(self, application: DynamicRagApplication) -> None:
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
                except Exception:
                    pass
            background = getattr(app.state, "background_ingestion", None)
            if background is not None and hasattr(background, "shutdown"):
                try:
                    background.shutdown()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Bearer / dependency helpers
# ---------------------------------------------------------------------------


def require_bearer(authorization: str | None) -> str:
    """Extract the bearer token from an ``Authorization`` header.

    Args:
        authorization: The raw header value (``"Bearer xxx"``) or ``None``.

    Returns:
        The trimmed token string.

    Raises:
        HTTPException: 401 if the header is missing or not bearer-formatted.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


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
# Versioned routes
# ---------------------------------------------------------------------------


class RouteGroup:
    """Register the ``/v1/*`` route group on the application.

    Centralises every route definition so :func:`create_app` stays a
    pure wiring step. The methods are pure builders; each one returns
    the decorated route function so tests can introspect them.

    Attributes:
        router: The :class:`APIRouter` carrying every v1 route.
    """

    def __init__(self) -> None:
        self.router = APIRouter()

    # ----- auth / session ----------------------------------------------

    def health(self) -> Callable[..., Any]:
        """Liveness probe; delegates to :meth:`DynamicRagApplication.health`."""

        @self.router.get("/health")
        def handler(app_service: DynamicRagApplication = Depends(get_application)) -> dict[str, Any]:
            return app_service.health()

        return handler

    def login(self) -> Callable[..., Any]:
        """Authenticate a user and return a session token."""

        @self.router.post("/auth/login", response_model=AuthLoginResponse)
        async def handler(
            payload: AuthLoginRequest,
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> AuthLoginResponse:
            return await app_service.login(payload.email, payload.password)

        return handler

    def logout(self) -> Callable[..., Any]:
        """Invalidate the bearer token presented in the ``Authorization`` header."""

        @self.router.post("/auth/logout")
        async def handler(
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> dict[str, str]:
            token = require_bearer(authorization)
            await app_service.logout(token)
            return {"status": "logged_out"}

        return handler

    def session_history(self) -> Callable[..., Any]:
        """Return the conversation history for the current session."""

        @self.router.get("/session/history")
        async def handler(
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> dict[str, list[dict[str, Any]]]:
            token = require_bearer(authorization)
            history = await app_service.history(token)
            return {"history": [turn.model_dump(mode="json") for turn in history]}

        return handler

    def clear_history(self) -> Callable[..., Any]:
        """Empty the conversation history for the current session."""

        @self.router.delete("/session/history", status_code=204, response_class=Response)
        async def handler(
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> Response:
            token = require_bearer(authorization)
            await app_service.clear_history(token)
            return Response(status_code=204)

        return handler

    # ----- documents ---------------------------------------------------

    def upload_document(self) -> Callable[..., Any]:
        """Upload a PDF document and synchronously index it."""

        @self.router.post("/documents/upload", status_code=202, response_model=DocumentUploadResponse)
        async def handler(
            request: Request,
            file: UploadFile = File(...),
            company: str | None = Form(default=None),
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> DocumentUploadResponse:
            token = require_bearer(authorization)
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
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> BatchIngestResponse:
            token = require_bearer(authorization)
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
                except Exception as exc:
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
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> dict[str, list[dict[str, Any]]]:
            token = require_bearer(authorization)
            documents = await app_service.list_documents(token)
            return {"documents": [document.model_dump(mode="json") for document in documents]}

        return handler

    def document_status(self) -> Callable[..., Any]:
        """Return the latest status for a single document."""

        @self.router.get("/documents/{document_id}/status")
        async def handler(
            document_id: str,
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> dict[str, Any]:
            token = require_bearer(authorization)
            document = await app_service.document_status(token, document_id)
            return document.model_dump(mode="json")

        return handler

    def delete_document(self) -> Callable[..., Any]:
        """Delete a document and all of its chunks. Admin-only."""

        @self.router.delete("/documents/{document_id}", status_code=204, response_class=Response)
        async def handler(
            document_id: str,
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> Response:
            token = require_bearer(authorization)
            await app_service.delete_document(token, document_id)
            return Response(status_code=204)

        return handler

    # ----- query -------------------------------------------------------

    def query(self) -> Callable[..., Any]:
        """Answer a question using the application service.

        Advanced-RAG flags (``agent``, ``web``, ``tools_enabled``
        etc.) are forwarded to the resolver when any of them are
        supplied; otherwise the legacy fast path runs.
        """

        @self.router.post("/query", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> QueryResponse:
            token = require_bearer(authorization)
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
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> dict[str, str]:
            token = require_bearer(authorization)
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

    # ----- streaming ---------------------------------------------------

    def query_stream(self) -> Callable[..., Any]:
        """Stream agent / planner events as Server-Sent Events."""

        @self.router.post("/query/stream")
        async def handler(
            payload: QueryRequest,
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> StreamingResponse:
            token = require_bearer(authorization)
            resolved_tools = (
                set(payload.tools_enabled) if payload.tools_enabled else set()
            )

            async def gen() -> Iterable[bytes]:
                yield sse_comment("raghub-query-stream")
                user, _ = await app_service.auth_svc.resolve_user(token)
                rag = app_service.container.rag_facade
                if rag is None:
                    yield sse_format("error", {"message": "RAG facade unavailable"})
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
                    yield sse_format(event.kind, event.model_dump(mode="json"))

            return StreamingResponse(gen(), media_type="text/event-stream")

        return handler

    def agent_run(self) -> Callable[..., Any]:
        """Run the agent end-to-end and return the full :class:`QueryResponse`."""

        @self.router.post("/agent/run", response_model=QueryResponse)
        async def handler(
            payload: QueryRequest,
            authorization: str | None = Header(default=None),
            app_service: DynamicRagApplication = Depends(get_application),
        ) -> QueryResponse:
            token = require_bearer(authorization)
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

    def register_all(self, app: FastAPI, prefix: str) -> None:
        """Build every route and mount the router under ``prefix``.

        Args:
            app: The FastAPI instance.
            prefix: The URL prefix (e.g. ``"/v1"``).
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
        ):
            builder()
        app.include_router(self.router, prefix=prefix)


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
    try:
        pkg = importlib.metadata.metadata("raghub")
    except Exception:
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
        """Mirrors ``GET /v1/health`` so Docker/Kubernetes probes skip the prefix."""
        return {"status": "ok"}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_app(application: DynamicRagApplication) -> FastAPI:
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
    app.include_router(admin_router, prefix="/v1")
    app.include_router(preferences_router, prefix="/v1")
    root_health_route(app)
    return app


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


app_singleton: FastAPI | None = None


def get_app() -> FastAPI:
    """Lazily build and return the singleton :class:`FastAPI` instance.

    Used by tooling (e.g. test fixtures, ASGI clients) that needs a
    fully-configured app but doesn't want to wire it themselves. The
    application container is built on the first call and reused on
    every subsequent call.

    Returns:
        The cached :class:`FastAPI` instance.
    """
    import asyncio

    from raghub.core import build_application

    global app_singleton
    if app_singleton is None:
        application = asyncio.run(build_application())
        app_singleton = create_app(application)
    return app_singleton


__all__ = [
    "ExceptionHandlers",
    "Lifespan",
    "RouteGroup",
    "app_singleton",
    "check_upload_size",
    "cors_origins_from_env",
    "create_app",
    "enforce_upload_limit",
    "get_app",
    "package_metadata",
    "require_bearer",
    "root_health_route",
    "upload_content_length",
    "validate_cors_for_credentials",
]