"""Document-management service (upload, list, status, delete).

This service is the single entry point for document operations. It
combines authentication, RBAC checks, MIME detection, and ingestion
into a small set of methods that mirror the public API surface.

Repository access goes through three small typed helpers —
:func:`_upload_record`, :func:`_list_all_records`, and
:func:`_document_by_id` — so the returned values are statically
typed as :class:`DocumentRecord` / ``list[DocumentRecord]`` rather
than ``Any`` flowing through the dynamic container.
"""

from __future__ import annotations

import time
from typing import Any

from raghub.core import can_access_company
from raghub.documents import detect_mime_type
from raghub.exceptions import AuthorizationError, DocumentError
from raghub.ingestion.service import IngestionResult
from raghub.models import DocumentRecord
from raghub.services import ServiceMixin


async def _upload_record(result: IngestionResult | Any) -> DocumentRecord:
    """Return the :class:`DocumentRecord` from an ingestion result.

    Args:
        result: The :class:`IngestionResult` (or compatible object)
            returned by the ingestion service.

    Returns:
        The :class:`DocumentRecord` carried by ``result``.
    """
    return result.document


async def _list_all_records(uow: Any) -> list[DocumentRecord]:
    """Return every document from the repository.

    Args:
        uow: The unit-of-work exposing ``document_repo.list_all()``.

    Returns:
        The full list of :class:`DocumentRecord`.
    """
    return await uow.document_repo.list_all()


async def _document_by_id(uow: Any, document_id: str) -> DocumentRecord:
    """Return a single document by id.

    Args:
        uow: The unit-of-work exposing ``document_repo.get(id)``.
        document_id: The document id.

    Returns:
        The matching :class:`DocumentRecord` (``None`` is converted
        into :class:`DocumentError` by the caller).
    """
    return await uow.document_repo.get(document_id)


class DocumentService(ServiceMixin):
    """Document upload, listing, status, and deletion."""

    def __init__(self, container: Any) -> None:
        """Store the container reference.

        Args:
            container: The application container.
        """
        self.container = container

    async def upload_document(
        self,
        *,
        token: str,
        filename: str,
        content: bytes,
        company: str | None = None,
    ) -> DocumentRecord:
        """Ingest a new document on behalf of the calling user.

        Args:
            token: Bearer token.
            filename: Original filename; used for MIME detection and,
                when ``company`` is omitted, to derive the tenant via
                ``filename.split("_", 1)[0]``.
            content: Raw bytes of the upload.
            company: Optional explicit tenant override.

        Returns:
            The persisted :class:`DocumentRecord`.

        Raises:
            AuthorizationError: If the caller cannot upload documents
                for the resolved company.
            DocumentError: If MIME detection or ingestion fails.
        """
        started = time.perf_counter()
        auth: Any = self.container.auth
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
        document = await _upload_record(result)

        self.emit_metric("document_ingest_latency_ms", started)
        self.log(
            "document_ingested", document_id=document.document_id, company=target_company
        )
        return document

    async def list_documents(self, token: str) -> list[DocumentRecord]:
        """List the documents visible to the caller.

        Admin users see every document; non-admins see only the
        documents whose ``organization`` is in their allow-list.

        Args:
            token: Bearer token.

        Returns:
            A list of :class:`DocumentRecord`. Empty when the user has
            no allowed companies and is not an admin.
        """
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        if user.is_admin:
            return await _list_all_records(self.container.uow)
        results: list[DocumentRecord] = []
        for org in user.allowed_companies:
            docs = await self.container.uow.document_repo.list_by_organization(org)
            results.extend(docs)
        return results

    async def document_status(self, token: str, document_id: str) -> DocumentRecord:
        """Return a single document's status.

        Args:
            token: Bearer token.
            document_id: The document id.

        Returns:
            The :class:`DocumentRecord`.

        Raises:
            DocumentError: If the document does not exist.
            AuthorizationError: If the caller cannot access the
                document's organization.
        """
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        document = await _document_by_id(self.container.uow, document_id)
        if document is None:
            raise DocumentError("Unknown document")
        if not can_access_company(user, document.organization):
            raise AuthorizationError("Forbidden")
        return document

    async def delete_document(self, token: str, document_id: str) -> None:
        """Delete a document and all of its chunks.

        Admin-only.

        Args:
            token: Bearer token.
            document_id: The document id.

        Raises:
            AuthorizationError: If the caller is not an admin.
        """
        auth: Any = self.container.auth
        user, _ = await auth.resolve_user(token)
        if not user.is_admin:
            raise AuthorizationError("Admin only")
        self.container.vector_store.delete_document(document_id)
        await self.container.uow.document_repo.delete(document_id)


__all__ = ["DocumentService"]