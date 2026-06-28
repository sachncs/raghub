"""FastAPI reference server."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from raghub.api.admin import router as admin_router
from raghub.api.dependencies import get_application
from raghub.api.rate_limiter import RateLimiterMiddleware
from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError, StorageError
from raghub.ingestion.background import BackgroundIngestionService
from raghub.models.api import AuthLoginRequest, AuthLoginResponse, DocumentUploadResponse, QueryRequest, QueryResponse
from raghub.services.application import DynamicRagApplication


def create_app(application: DynamicRagApplication) -> FastAPI:
    """Create a FastAPI app bound to the application service."""

    app = FastAPI(title="Dynamic RAG Platform", version="1.0.0")
    app.state.application = application
    app.state.background_ingestion = BackgroundIngestionService(max_workers=2)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimiterMiddleware, rate=10.0, burst=20)
    app.include_router(admin_router)

    @app.exception_handler(AuthenticationError)
    def authentication_error_handler(
        _: Any,
        exc: AuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(AuthorizationError)
    def authorization_error_handler(
        _: Any,
        exc: AuthorizationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(DocumentError)
    def document_error_handler(
        _: Any,
        exc: DocumentError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(StorageError)
    def storage_error_handler(
        _: Any,
        exc: StorageError,
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/health")
    def health(app_service: DynamicRagApplication = Depends(get_application)) -> dict[str, Any]:
        return app_service.health()

    @app.post("/auth/login", response_model=AuthLoginResponse)
    async def login(
        payload: AuthLoginRequest,
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""

        return await app_service.login(payload.email, payload.password)

    @app.post("/auth/logout")
    async def logout(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, str]:
        """Invalidate the bearer token."""

        token = require_bearer(authorization)
        await app_service.logout(token)
        return {"status": "logged_out"}

    @app.get("/session/history")
    async def session_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """Return conversation history for the current session."""

        token = require_bearer(authorization)
        history = await app_service.history(token)
        return {"history": [turn.model_dump(mode="json") for turn in history]}

    @app.delete("/session/history", status_code=204)
    async def clear_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> None:
        token = require_bearer(authorization)
        await app_service.clear_history(token)

    @app.post("/documents/upload", status_code=202, response_model=DocumentUploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> DocumentUploadResponse:
        """Upload a PDF document and index it."""

        token = require_bearer(authorization)
        content = await file.read()
        document = await app_service.upload_document(
            token=token,
            filename=file.filename or "upload.pdf",
            content=content,
            company=company,
        )
        response = DocumentUploadResponse(
            document_id=document.document_id,
            version=document.version,
            status=document.status.value,
            company=document.organization,
            filename=document.filename,
        )
        return response

    @app.get("/documents")
    async def list_documents(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """List documents accessible to the current user."""

        token = require_bearer(authorization)
        documents = await app_service.list_documents(token)
        return {"documents": [document.model_dump(mode="json") for document in documents]}

    @app.get("/documents/{document_id}/status")
    async def document_status(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, Any]:
        """Return the latest status for a document."""

        token = require_bearer(authorization)
        document = await app_service.document_status(token, document_id)
        return document.model_dump(mode="json")

    @app.delete("/documents/{document_id}", status_code=204)
    async def delete_document(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> None:
        """Delete a document."""

        token = require_bearer(authorization)
        await app_service.delete_document(token, document_id)

    @app.post("/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> QueryResponse:
        """Answer a question using the application service."""

        token = require_bearer(authorization)
        response = await app_service.query(token=token, question=payload.question)
        return response

    @app.post("/ingest/async")
    async def ingest_async(
        file: UploadFile = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
        request: Request = Depends(lambda r: r),
    ) -> dict[str, str]:
        """Queue a document for background ingestion."""

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

    return app


def require_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


_app_instance: FastAPI | None = None


def get_app() -> FastAPI:
    import asyncio

    from raghub.core.container import build_application

    global _app_instance
    if _app_instance is None:
        application = asyncio.run(build_application())
        _app_instance = create_app(application)
    return _app_instance
