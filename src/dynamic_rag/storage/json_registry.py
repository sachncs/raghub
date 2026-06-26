"""JSON-backed document registry with version history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from dynamic_rag.exceptions import StorageError
from dynamic_rag.models import DocumentLifecycleStatus, DocumentVersion
from dynamic_rag.utils import atomic_write_json, load_json


@dataclass
class RegistrySnapshot:
    """Serializable registry payload."""

    documents: dict[str, list[DocumentVersion]]
    checksum_index: dict[str, tuple[str, int]]


class JsonDocumentRegistry:
    """Persistent registry for versioned documents."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = RLock()
        self._documents: dict[str, list[DocumentVersion]] = {}
        self._checksum_index: dict[str, tuple[str, int]] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = load_json(self.path, default={"documents": {}, "checksum_index": {}})
            documents = payload.get("documents", {})
            checksum_index = payload.get("checksum_index", {})
            self._documents = {
                document_id: [DocumentVersion.model_validate(item) for item in versions]
                for document_id, versions in documents.items()
                if isinstance(versions, list)
            }
            self._checksum_index = {
                checksum: tuple(value) for checksum, value in checksum_index.items() if isinstance(value, list)
            }
        except Exception:
            self._documents = {}
            self._checksum_index = {}

    def _save(self) -> None:
        try:
            atomic_write_json(
                self.path,
                {
                    "documents": {
                        document_id: [version.model_dump(mode="json") for version in versions]
                        for document_id, versions in self._documents.items()
                    },
                    "checksum_index": {checksum: list(value) for checksum, value in self._checksum_index.items()},
                },
            )
        except Exception as exc:  # pragma: no cover - persistence error path
            raise StorageError(str(exc)) from exc

    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        """Persist a new document version."""

        with self._lock:
            versions = self._documents.setdefault(document.document_id, [])
            for index, existing in enumerate(versions):
                if existing.version == document.version:
                    versions[index] = document
                    break
            else:
                if versions and document.version > versions[-1].version:
                    versions[-1].status = DocumentLifecycleStatus.ARCHIVED
                    versions[-1].updated_at = datetime.now(timezone.utc)
                versions.append(document)
            self._checksum_index[document.checksum] = (document.document_id, document.version)
            self._save()
            return document

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        """Return latest version for a document."""

        with self._lock:
            versions = self._documents.get(document_id, [])
            return versions[-1] if versions else None

    def get_version(self, document_id: str, version: int) -> DocumentVersion | None:
        """Return a specific version."""

        with self._lock:
            versions = self._documents.get(document_id, [])
            for item in versions:
                if item.version == version:
                    return item
            return None

    def get_by_checksum(self, checksum: str) -> DocumentVersion | None:
        """Return the document for a checksum."""

        with self._lock:
            locator = self._checksum_index.get(checksum)
            if locator is None:
                return None
            return self.get_version(locator[0], locator[1])

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        """List visible documents for a company filter."""

        with self._lock:
            result: list[DocumentVersion] = []
            for versions in self._documents.values():
                latest = versions[-1]
                if latest.organization in companies and latest.status != DocumentLifecycleStatus.ARCHIVED:
                    result.append(latest)
            return result

    def archive(self, document_id: str) -> None:
        """Archive the latest version."""

        with self._lock:
            latest = self.get_latest(document_id)
            if latest is None:
                return
            latest.status = DocumentLifecycleStatus.ARCHIVED
            latest.updated_at = datetime.now(timezone.utc)
            self._save()

    def dump(self) -> RegistrySnapshot:
        """Return the in-memory snapshot."""

        with self._lock:
            return RegistrySnapshot(self._documents, self._checksum_index)
