"""JSON-backed document registry with version history.

This module implements a simple on-disk registry that keeps every
document version on append, with a separate checksum index for fast
duplicate detection. It is intentionally minimal — a production
deployment should use the SQLite-backed
:class:`raghub.repositories.sqlite_document_repo.SqliteDocumentRepository`
instead; this module exists primarily as the legacy/JSON migration
source-of-truth and for tiny single-process deployments.

Concurrency:
    The store uses a :class:`threading.RLock` around all mutating and
    snapshot reads. Sharing an instance across threads is safe; sharing
    across processes requires external coordination.

Storage layout:

    documents = { "<document_id>": [DocumentVersion, ...], ... }
    checksum_index = { "<sha256>": ["<document_id>", <version>], ... }

The checksum index lets deduplication queries run in O(1) instead of
scanning every document's history.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from raghub.exceptions import StorageError
from raghub.models import DocumentLifecycleStatus, DocumentVersion
from raghub.utils import atomic_write_json, load_json


@dataclass
class RegistrySnapshot:
    """In-memory snapshot of the registry.

    Returned by :meth:`JsonDocumentRegistry.dump` for callers that want
    to inspect or serialise the entire registry without going through
    disk I/O.

    Attributes:
        documents: Mapping from document id to its version list.
        checksum_index: Mapping from SHA-256 checksum to
            ``(document_id, version)``.
    """

    documents: dict[str, list[DocumentVersion]]
    checksum_index: dict[str, tuple[str, int]]


class JsonDocumentRegistry:
    """Persistent registry for versioned documents.

    Versions are stored in append order; each new version that exceeds
    the previous latest's number automatically archives the prior
    latest (see :meth:`save_version`).
    """

    def __init__(self, path: Path) -> None:
        """Initialise the registry and load the existing JSON state.

        Args:
            path: Filesystem path to the JSON file. Created on first
                :meth:`save` if it does not exist.
        """
        self.path = path
        self.lock = RLock()
        self.documents: dict[str, list[DocumentVersion]] = {}
        self.checksum_index: dict[str, tuple[str, int]] = {}
        self.load()

    def load(self) -> None:
        """Hydrate in-memory state from disk.

        Tolerates a missing or malformed file by resetting to empty
        state; this is the behaviour we want for first-run startup.
        """
        try:
            payload = load_json(self.path, default={"documents": {}, "checksum_index": {}})
            documents = payload.get("documents", {})
            checksum_index = payload.get("checksum_index", {})
            # Defensive parsing: ignore entries whose shape is wrong.
            # Future schema changes should land here as drop-points so
            # we never crash on partially-migrated files.
            self.documents = {
                document_id: [DocumentVersion.model_validate(item) for item in versions]
                for document_id, versions in documents.items()
                if isinstance(versions, list)
            }
            self.checksum_index = {
                checksum: tuple(value) for checksum, value in checksum_index.items() if isinstance(value, list)
            }
        except Exception:
            # Bad JSON or schema mismatch — start fresh. This is
            # acceptable because the canonical store is the SQLite one.
            self.documents = {}
            self.checksum_index = {}

    def save(self) -> None:
        """Persist in-memory state to disk atomically.

        Raises:
            StorageError: If the atomic write fails for any reason
                (disk full, permission denied, etc.).
        """
        try:
            atomic_write_json(
                self.path,
                {
                    "documents": {
                        document_id: [version.model_dump(mode="json") for version in versions]
                        for document_id, versions in self.documents.items()
                    },
                    "checksum_index": {checksum: list(value) for checksum, value in self.checksum_index.items()},
                },
            )
        except Exception as exc:  # pragma: no cover - persistence error path
            raise StorageError(str(exc)) from exc

    def save_version(self, document: DocumentVersion) -> DocumentVersion:
        """Persist a new or updated :class:`DocumentVersion`.

        Algorithm:

        * If a version with the same number exists, replace it in place.
        * Otherwise append; if the new version is greater than the
          current latest, archive the current latest so the history
          shows a clean "superseded by" chain.
        * Always update the checksum index so future dedup queries hit
          the new version's id+number.

        Args:
            document: The version to persist.

        Returns:
            The same ``document`` argument, for call-site convenience.
        """
        with self.lock:
            versions = self.documents.setdefault(document.document_id, [])
            for index, existing in enumerate(versions):
                if existing.version == document.version:
                    # Replace-in-place: an out-of-order write for an
                    # existing version number should update, not append.
                    versions[index] = document
                    break
            else:
                # ``for/else`` runs when the loop completes without a
                # ``break`` — i.e. this is a brand-new version number.
                if versions and document.version > versions[-1].version:
                    # New latest supersedes the old; mark the previous
                    # latest as ``ARCHIVED`` so historical chains stay
                    # accurate even after the active version changes.
                    versions[-1].status = DocumentLifecycleStatus.ARCHIVED
                    versions[-1].updated_at = datetime.now(timezone.utc)
                versions.append(document)
            # Update the checksum index unconditionally so dedup lookups
            # always reflect the most recent write.
            self.checksum_index[document.checksum] = (document.document_id, document.version)
            self.save()
            return document

    def get_latest(self, document_id: str) -> DocumentVersion | None:
        """Return the newest version of ``document_id``.

        Args:
            document_id: The document id.

        Returns:
            The latest :class:`DocumentVersion`, or ``None`` if the
            document is unknown.
        """
        with self.lock:
            versions = self.documents.get(document_id, [])
            return versions[-1] if versions else None

    def get_version(self, document_id: str, version: int) -> DocumentVersion | None:
        """Return a specific historical version.

        Args:
            document_id: The document id.
            version: The version number to fetch.

        Returns:
            The matching :class:`DocumentVersion`, or ``None``.
        """
        with self.lock:
            versions = self.documents.get(document_id, [])
            for item in versions:
                if item.version == version:
                    return item
            return None

    def get_by_checksum(self, checksum: str) -> DocumentVersion | None:
        """Look up the document owning ``checksum``.

        Args:
            checksum: SHA-256 hex digest of the raw file bytes.

        Returns:
            The :class:`DocumentVersion` recorded for that checksum,
            or ``None``.
        """
        with self.lock:
            locator = self.checksum_index.get(checksum)
            if locator is None:
                return None
            return self.get_version(locator[0], locator[1])

    def list_accessible(self, companies: list[str]) -> list[DocumentVersion]:
        """Return the latest version of every non-archived document in ``companies``.

        Args:
            companies: Tenant allow-list; only documents whose
                ``organization`` is in this list are returned.

        Returns:
            A list of latest-version :class:`DocumentVersion` objects.
        """
        with self.lock:
            result: list[DocumentVersion] = []
            for versions in self.documents.values():
                latest = versions[-1]
                # Skip archived rows so historical chains don't pollute
                # tenant-facing listings.
                if latest.organization in companies and latest.status != DocumentLifecycleStatus.ARCHIVED:
                    result.append(latest)
            return result

    def archive(self, document_id: str) -> None:
        """Archive the latest version of ``document_id``.

        No-op if the document is unknown.

        Args:
            document_id: The document id to archive.
        """
        with self.lock:
            latest = self.get_latest(document_id)
            if latest is None:
                return
            latest.status = DocumentLifecycleStatus.ARCHIVED
            latest.updated_at = datetime.now(timezone.utc)
            self.save()

    def dump(self) -> RegistrySnapshot:
        """Return an in-memory snapshot of the registry.

        The snapshot shares structure with the live registry under the
        lock; do not mutate it after the lock is released.

        Returns:
            A :class:`RegistrySnapshot`.
        """
        with self.lock:
            return RegistrySnapshot(self.documents, self.checksum_index)