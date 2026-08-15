"""JSON-backed document registry with version history.

Versions are stored in append order; each new version that exceeds
the previous latest's number automatically archives the prior latest
(see :meth:`save_version`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from raghub.io import atomic_write_json, load_json
from raghub.models import Document, DocumentLifecycleStatus

__all__ = ["Documents", "Snapshot"]


@dataclass
class Snapshot:
    """In-memory snapshot of a document registry."""

    documents: dict[str, list[Document]]
    checksum_index: dict[str, tuple[str, int]]


class Documents:
    """JSON-backed persistent registry for versioned documents.

    Versions are stored in append order; each new version that exceeds
    the previous latest's number automatically archives the prior latest
    (see :meth:`save_version`).
    """

    def __init__(self, path: Path) -> None:
        """Initialise the registry and load the existing JSON state."""
        self.path = path
        self.lock = RLock()
        self.documents: dict[str, list[Document]] = {}
        self.checksum_index: dict[str, tuple[str, int]] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk.

        Tolerates a missing or malformed file by resetting to empty
        state; this is the behaviour we want for first-run startup.
        """
        if self.path.exists() and not self.path.read_text(encoding="utf-8").lstrip().startswith(
            "{"
        ):
            self.documents = {}
            self.checksum_index = {}
            return
        payload = load_json(self.path, default={"documents": {}, "checksum_index": {}})
        documents = payload.get("documents", {})
        checksum_index = payload.get("checksum_index", {})
        self.documents = {
            document_id: [Document.validate(version_record) for version_record in versions]
            for document_id, versions in documents.items()
            if isinstance(versions, list)
        }
        self.checksum_index = {
            checksum: tuple(value)
            for checksum, value in checksum_index.items()
            if isinstance(value, list)
        }

    def save(self) -> None:
        """Persist in-memory state to disk atomically.

        Raises:
            RagHubError: If the atomic write fails.

        """
        from raghub.errors import RagHubError
        from raghub.runtime import capture

        _, error = capture(
            atomic_write_json,
            self.path,
            {
                "documents": {
                    document_id: [version.dump(mode="json") for version in versions]
                    for document_id, versions in self.documents.items()
                },
                "checksum_index": {
                    checksum: list(value) for checksum, value in self.checksum_index.items()
                },
            },
        )
        if error is not None:
            raise RagHubError(str(error)) from error

    def save_version(self, document: Document) -> Document:
        """Persist a new or updated :class:`Document`."""
        with self.lock:
            versions = self.documents.setdefault(document.id, [])
            for index, existing in enumerate(versions):
                if existing.version == document.version:
                    # Replace-in-place: an out-of-order write for an
                    # existing version number should update, not append.
                    versions[index] = document
                    break
            else:
                # ``for/else`` runs when the loop completes without a
                # ``break`` — a brand-new version number.
                if versions and document.version > versions[-1].version:
                    versions[-1] = versions[-1].copy(
                        status=DocumentLifecycleStatus.Archived,
                        updated_at=datetime.now(UTC),
                    )
                versions.append(document)
            self.checksum_index[document.checksum] = (
                document.id,
                document.version,
            )
            self.save()
            return document

    def get_latest(self, document_id: str) -> Document | None:
        """Return the highest-versioned entry for ``document_id``."""
        with self.lock:
            versions = self.documents.get(document_id, [])
            if not versions:
                return None
            return max(versions, key=lambda v: v.version)

    def get_version(self, document_id: str, version: int) -> Document | None:
        """Return a specific historical version, or ``None``."""
        with self.lock:
            for version_record in self.documents.get(document_id, []):
                if version_record.version == version:
                    return version_record
            return None

    def by_checksum(self, checksum: str) -> Document | None:
        """Look up the document owning ``checksum``."""
        with self.lock:
            locator = self.checksum_index.get(checksum)
            if locator is None:
                return None
            return self.get_version(locator[0], locator[1])

    def list_accessible(self, companies: list[str]) -> list[Document]:
        """Return the latest version of every non-archived document."""
        with self.lock:
            result: list[Document] = []
            for versions in self.documents.values():
                latest = versions[-1]
                if (
                    latest.organization in companies
                    and latest.status != DocumentLifecycleStatus.Archived
                ):
                    result.append(latest)
            return result

    def archive(self, document_id: str) -> None:
        """Archive the latest version of ``document_id``. No-op if unknown."""
        with self.lock:
            latest = self.get_latest(document_id)
            if latest is None:
                return
            archived = latest.copy(
                status=DocumentLifecycleStatus.Archived,
                updated_at=datetime.now(UTC),
            )
            versions = self.documents[document_id]
            for index, current in enumerate(versions):
                if current.version == archived.version:
                    versions[index] = archived
                    break
            self.save()

    def dump(self) -> Snapshot:
        """Return an in-memory snapshot of the registry."""
        with self.lock:
            return Snapshot(self.documents, self.checksum_index)
