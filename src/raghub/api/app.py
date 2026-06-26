"""FastAPI reference server."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from raghub.core.container import build_application
from raghub.api.dependencies import get_application
from raghub.exceptions import AuthenticationError, AuthorizationError, DocumentError, StorageError
from raghub.models.api import AuthLoginRequest, AuthLoginResponse, DocumentUploadResponse, QueryRequest, QueryResponse
from raghub.services.application import DynamicRagApplication


def create_app(application: DynamicRagApplication | None = None) -> FastAPI:
    """Create a FastAPI app bound to the application service."""

    app = FastAPI(title="Dynamic RAG Platform", version="1.0.0")
    app.state.application = application or build_application()

    @app.exception_handler(AuthenticationError)
    def _authentication_error(
        _: Any,
        exc: AuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(AuthorizationError)
    def _authorization_error(
        _: Any,
        exc: AuthorizationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(DocumentError)
    def _document_error(
        _: Any,
        exc: DocumentError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(StorageError)
    def _storage_error(
        _: Any,
        exc: StorageError,
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/health")
    def health(app_service: DynamicRagApplication = Depends(get_application)) -> dict[str, Any]:
        return app_service.health()

    @app.post("/auth/login", response_model=AuthLoginResponse)
    def login(
        payload: AuthLoginRequest,
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> AuthLoginResponse:
        """Authenticate a user and return a session token."""

        return app_service.login(payload.email)

    @app.post("/auth/logout")
    def logout(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, str]:
        """Invalidate the bearer token."""

        token = _require_bearer(authorization)
        app_service.logout(token)
        return {"status": "logged_out"}

    @app.get("/session/history")
    def session_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """Return conversation history for the current session."""

        token = _require_bearer(authorization)
        return {"history": [turn.model_dump(mode="json") for turn in app_service.history(token)]}

    @app.delete("/session/history", status_code=204)
    def clear_history(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> None:
        token = _require_bearer(authorization)
        app_service.clear_history(token)

    @app.post("/documents/upload", status_code=202, response_model=DocumentUploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        company: str | None = Form(default=None),
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> DocumentUploadResponse:
        """Upload a PDF document and index it."""

        token = _require_bearer(authorization)
        content = await file.read()
        document = app_service.upload_document(
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
    def list_documents(
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, list[dict[str, Any]]]:
        """List documents accessible to the current user."""

        token = _require_bearer(authorization)
        documents = app_service.list_documents(token)
        return {"documents": [document.model_dump(mode="json") for document in documents]}

    @app.get("/documents/{document_id}/status")
    def document_status(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> dict[str, Any]:
        """Return the latest status for a document."""

        token = _require_bearer(authorization)
        document = app_service.document_status(token, document_id)
        return document.model_dump(mode="json")

    @app.delete("/documents/{document_id}", status_code=204)
    def delete_document(
        document_id: str,
        authorization: str | None = Header(default=None),
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> None:
        """Delete a document."""

        token = _require_bearer(authorization)
        app_service.delete_document(token, document_id)

    @app.post("/query", response_model=QueryResponse)
    def query(
        payload: QueryRequest,
        app_service: DynamicRagApplication = Depends(get_application),
    ) -> QueryResponse:
        """Answer a question using the application service."""

        response = app_service.query(token=payload.session_token, question=payload.question)
        return response

    return app


def _require_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.split(" ", 1)[1].strip()


app = create_app()
