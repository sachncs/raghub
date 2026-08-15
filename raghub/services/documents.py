"""Document service: upload, listing, status, and deletion."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from raghub.core import can_access_company
from raghub.errors import AuthorizationError, IngestionError
from raghub.lifecycle import detect_mime_type
from raghub.models import Document
from raghub.services.diagnostics import emit_log, emit_metric, upload_record
from raghub.types import JSONValue

if TYPE_CHECKING:
    from raghub.auth import AuthService
    from raghub.repos import UnitOfWork
    from raghub.services.container import RagContainer


async def list_records(uow: UnitOfWork) -> list[Document]:
    """Return every document from the repository."""
    return cast(list[Document], await uow.document_repo.list_all())


async def get_doc(uow: UnitOfWork, document_id: str) -> Document:
    """Return a single document by id or raise :class:`IngestionError`."""
    record = await uow.document_repo.get(document_id)
    if record is None:
        from raghub.services.diagnostics import missing_doc

        missing_doc(document_id)
        raise IngestionError(f"Unknown document id: {document_id}")
    return record


class Documents:
    """Document upload, listing, status, and deletion."""

    def __init__(self, container: RagContainer) -> None:
        """Store the container reference."""
        self.container = container

    def log(self, message: str, **payload: JSONValue) -> None:
        """Emit a structured log event."""
        emit_log(self.container, message, **payload)

    def emit_metric(self, name: str, started_at: float) -> None:
        """Record a latency metric."""
        emit_metric(self.container, name, started_at)

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> Document:
        """Ingest a new document on behalf of the calling user.

        Raises:
            AuthorizationError: If the caller cannot upload documents
                for the resolved company.
            IngestionError: If MIME detection or ingestion fails.

        """
        started = time.perf_counter()
        auth: AuthService | None = self.container.auth
        if auth is None:
            raise RuntimeError("container.auth must be set before this call")
        auth = auth
        user, _ = await auth.resolve_user(token)
        target_company = company or filename.split("_", 1)[0]
        if not can_access_company(user, target_company):
            raise AuthorizationError("User cannot upload documents for this company")

        detect_mime_type(filename, content)

        result = await self.container.ingestion.ingest(
            file_name=filename,
            file_bytes=content,
            owner=user,
            organization=target_company,
        )
        document = upload_record(result)

        self.emit_metric("document_ingest_latency_ms", started)
        self.log(
            "document_ingested",
            document_id=document.id,
            company=target_company,
        )
        return document

    async def list_documents(self, token: str) -> list[Document]:
        """List the documents visible to the caller.

        Admin users see every document; non-admins see only the
        documents whose organization is in their allow-list.
        """
        auth: AuthService | None = self.container.auth
        if auth is None:
            raise RuntimeError("container.auth must be set before this call")
        auth = auth
        user, _ = await auth.resolve_user(token)
        if user.is_admin:
            return await list_records(self.container.uow)
        results: list[Document] = []
        for org in user.allowed_companies:
            docs = await self.container.uow.document_repo.list_by_organization(org)
            results.extend(docs)
        return results

    async def document_status(self, token: str, document_id: str) -> Document:
        """Return a single document's status.

        Raises:
            IngestionError: If the document does not exist.
            AuthorizationError: If the caller cannot access the
                document's organization.

        """
        auth: AuthService | None = self.container.auth
        if auth is None:
            raise RuntimeError("container.auth must be set before document_status")
        user, _ = await auth.resolve_user(token)
        document = await get_doc(self.container.uow, document_id)
        if document is None:
            raise IngestionError(f"document not found: {document_id}")
        if not can_access_company(user, document.organization):
            raise AuthorizationError("Forbidden")
        return document

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks. Admin-only.

        Delegates to the RAG facade so the same deletion path used by
        the programmatic API also retires the manifest entry, the
        knowledge repo records, and the structured indexes
        (Raptor / Graph). Admin check is preserved here because the
        facade itself does not enforce it.
        """
        auth: AuthService | None = self.container.auth
        if auth is None:
            raise RuntimeError("container.auth must be set before this call")
        auth = auth
        user, _ = await auth.resolve_user(token)
        if not user.is_admin:
            raise AuthorizationError("Admin only")
        rag = getattr(self.container, "rag_facade", None)
        if rag is not None and hasattr(rag, "delete"):
            rag.delete(document_id)
            return
        self.container.vector_store.delete_document(document_id)
        await self.container.uow.document_repo.delete(document_id)


__all__ = ["Documents", "get_doc", "list_records"]
