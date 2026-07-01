"""FastAPI reference server.

Defines :func:`create_app`, which builds a :class:`FastAPI` instance
bound to a fully-wired :class:`DynamicRagApplication`. The factory
wires in:

* CORS middleware (origins from ``CORS_ORIGINS`` env, comma-separated).
* The :class:`RateLimiterMiddleware` (default 10 rps, burst 20).
* The admin router from :mod:`raghub.api.admin`.
* Exception handlers for ``AuthenticationError`` (401),
  ``AuthorizationError`` (403), ``DocumentError`` (400), and
  ``StorageError`` (500).
* A shared :class:`BackgroundIngestionService` placed on
  ``app.state.background_ingestion`` for the ``/ingest/async`` endpoint.

Also exposes :func:`require_bearer` (used by routes to extract the
bearer token from the ``Authorization`` header) and :func:`get_app`
(a lazy singleton convenience used by tooling that needs the app
without going through the FastAPI CLI).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from raghub.api.admin import router as admin_router
from raghub.api.dependencies import get_application
from raghub.api.rate_limiter import RateLimiterMiddleware
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: graceful shutdown of collaborators."""
    try:
        yield
    finally:
        application: DynamicRagApplication = app.state.application
        try:
            await application.shutdown()
        except Exception:
            pass
        background = getattr(app.state, "background_ingestion", None)
        if background is not None and hasattr(background, "shutdown"):
            try:
                background.shutdown()
            except Exception:
                pass


def create_app(application: DynamicRagApplication) -> FastAPI:
    """Build a :class:`FastAPI` instance wired to ``application``.

    Args:
        application: The pre-wired application facade.

    Returns:
        A fully-configured FastAPI app ready to be served by
        ``uvicorn`` or any ASGI server.
    """
    from importlib.metadata import metadata as get_metadata

    try:
        pkg = get_metadata("retrieval-augmented-generation")
        app_title = pkg["Name"].replace("-", " ").title()
        app_version = pkg["Version"]
        app_description = pkg.get("Summary", "Production-grade Dynamic RAG framework")
    except Exception:
        app_title = "Dynamic RAG Platform"
        app_version = "0.2.0"
        app_description = "Production-grade Dynamic RAG framework"

    app = FastAPI(title=app_title, version=app_version, description=app_description, lifespan=lifespan)
    app.state.application = application
    # Shared background ingestion pool (2 workers by default); the
    # ``/ingest/async`` endpoint submits jobs to it.
    app.state.background_ingestion = BackgroundIngestionService(max_workers=2)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Token-bucket rate limiter; per-client IP by default.
    app.add_middleware(RateLimiterMiddleware, rate=10.0, burst=20)
    router = APIRouter()

    # Exception handlers translate typed application errors into HTTP
    # responses without leaking internal exception class names.
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

    @router.get("/health")
    def health(app_service: DynamicRagApplication = Depends(get_application)) -> dict[str, Any]:
        """Liveness probe; delegates to :meth:`DynamicRagApplication.health`."""
        return app_service.health()

    @router.post("/auth/login", response_model=AuthLoginResponse)
    async def login(
        payload: AuthLoginRequest,
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> AuthLoginResponse:
        """Authenticate a user and return a session token.

        Args:
            payload: The :class:`AuthLoginRequest` body.

        Returns:
            The :class:`AuthLoginResponse` with a session token, email,
            and allowed companies.
        """
        return await app_service.login(payload.email, payload.password)

    @router.post("/auth/logout")
    async def logout(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, str]:
        """Invalidate the bearer token presented in the ``Authorization`` header.

        Args:
            authorization: The raw ``Authorization`` header.

        Returns:
            A ``{"status": "logged_out"}`` payload.
        """
        token = require_bearer(authorization)
        await app_service.logout(token)
        return {"status": "logged_out"}

    @router.get("/session/history")
    async def session_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """Return the conversation history for the current session.

        Args:
            authorization: The raw ``Authorization`` header.

        Returns:
            ``{"history": [...]}`` where ``...`` is the serialised
            :class:`ConversationTurn` list (oldest first).
        """
        token = require_bearer(authorization)
        history = await app_service.history(token)
        return {"history": [turn.model_dump(mode="json") for turn in history]}

    @router.delete("/session/history", status_code=204, response_class=Response)
    async def clear_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> Response:
        """Empty the conversation history for the current session.

        Args:
            authorization: The raw ``Authorization`` header.
        """
        token = require_bearer(authorization)
        await app_service.clear_history(token)
        return Response(status_code=204)

    @router.post("/documents/upload", status_code=202, response_model=DocumentUploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> DocumentUploadResponse:
        """Upload a PDF document and synchronously index it.

        Args:
            file: The multipart upload.
            company: Optional tenant override; derived from the filename
                when omitted.
            authorization: The raw ``Authorization`` header.

        Returns:
            A :class:`DocumentUploadResponse` with the document id,
            version, status, company, and filename.
        """
        token = require_bearer(authorization)
        content = await file.read()
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

    @router.post(
        "/documents/ingest/batch",
        status_code=200,
        response_model=BatchIngestResponse,
    )
    async def ingest_documents_batch(
        files: list[UploadFile] = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> BatchIngestResponse:
        """Ingest multiple documents in a single request.

        Accepts one or more files as multipart upload. Each file is
        ingested independently; a failure in one does not affect the
        others.

        Memory characteristics: large files are buffered entirely in
        memory before ingestion. The per-file peak memory usage
        depends on the vector-store backend — zvec and Qdrant store
        vectors server-side so client memory is O(file_size), while
        the memory backend keeps everything in-process so peak RSS
        grows with total batch size.

        Args:
            files: One or more multipart file uploads.
            company: Optional tenant override applied to **all** files.
            authorization: The raw ``Authorization`` header.

        Returns:
            A :class:`BatchIngestResponse` with one item per file.
        """
        token = require_bearer(authorization)
        results: list[BatchIngestItem] = []
        for file in files:
            try:
                content = await file.read()
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

    @router.get("/documents")
    async def list_documents(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """List the documents visible to the calling user.

        Args:
            authorization: The raw ``Authorization`` header.

        Returns:
            ``{"documents": [...]}`` containing the user's accessible
            :class:`DocumentRecord` list.
        """
        token = require_bearer(authorization)
        documents = await app_service.list_documents(token)
        return {"documents": [document.model_dump(mode="json") for document in documents]}

    @router.get("/documents/{document_id}/status")
    async def document_status(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, Any]:
        """Return the latest status for a single document.

        Args:
            document_id: The document id.
            authorization: The raw ``Authorization`` header.

        Returns:
            The serialised :class:`DocumentRecord`.
        """
        token = require_bearer(authorization)
        document = await app_service.document_status(token, document_id)
        return document.model_dump(mode="json")

    @router.delete("/documents/{document_id}", status_code=204, response_class=Response)
    async def delete_document(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> Response:
        """Delete a document and all of its chunks. Admin-only.

        Args:
            document_id: The document id.
            authorization: The raw ``Authorization`` header.
        """
        token = require_bearer(authorization)
        await app_service.delete_document(token, document_id)
        return Response(status_code=204)

    @router.post("/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> QueryResponse:
        """Answer a question using the application service.

        Args:
            payload: The :class:`QueryRequest` body.
            authorization: The raw ``Authorization`` header.

        Returns:
            A :class:`QueryResponse` with the answer, citations, and
            source chunks.
        """
        token = require_bearer(authorization)
        response = await app_service.query(token=token, question=payload.question)
        return response

    @router.post("/ingest/async")
    async def ingest_async(
        request: Request,
        file: UploadFile = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, str]:
        """Queue a document for asynchronous ingestion.

        Args:
            file: The multipart upload.
            company: Optional tenant override.
            authorization: The raw ``Authorization`` header.
            app_service: The application facade.
            request: The current FastAPI request (used to reach
                ``app.state.background_ingestion``).

        Returns:
            ``{"job_id": "<uuid>"}`` that can later be polled via the
            background ingestion service.
        """
        token = require_bearer(authorization)
        content = await file.read()
        background = request.app.state.background_ingestion
        job_id = background.submit(
            app_service.upload_document,
            token=token,
            filename=file.filename or "upload.pdf",
            content=content,
            company=company,
        )
        return {"job_id": job_id}

    app.include_router(router, prefix="/v1")
    app.include_router(admin_router, prefix="/v1")

    return app


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


# Module-level singleton used by :func:`get_app`. Avoid importing
# build_application at module load time so this module stays cheap to
# import in unit tests that don't need the full app.
app_instance: FastAPI | None = None


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

    from raghub.core.container import build_application

    global app_instance
    if app_instance is None:
        application = asyncio.run(build_application())
        app_instance = create_app(application)
    return app_instance