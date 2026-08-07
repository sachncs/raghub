"""Sync mixin for the RAG facade.

Holds the incremental-indexing entry points (:meth:`sync_index`,
:meth:`sync_one`, :meth:`remove_prior`) that reconcile a directory
against the manifest.

The mixin assumes the host class has already wired the
collaborators it needs:

- ``self.manifest`` :class:`Manifest` for prior checksums
- ``self.ingest`` and ``self.delete`` for the actual mutation
- ``self.settings`` for any per-facade policies
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from tqdm import tqdm

from raghub.config import Settings
from raghub.errors import IngestionError, RagHubError
from raghub.knowledge import Manifest, sha256_bytes
from raghub.models import Pipeline, deterministic_id
from raghub.types import JSONValue


class SyncHost(Protocol):
    """Protocol of the host methods :class:`SyncMixin` delegates to.

    The composing class (``RAG``) supplies these via other mixins
    (notably :class:`IngestMixin`). Protocols are zero-cost at runtime
    so this lives at module scope.
    """

    def ingest(
        self,
        source: str | Path | bytes,
        **options: JSONValue,
    ) -> Pipeline:
        """Host-provided ingest entry point."""

    def delete(self, document_id: str) -> None:
        """Host-provided delete entry point."""


class SyncMixin(SyncHost):
    """Mixin providing incremental-indexing (sync) entry points.

    Inherits from :class:`SyncHost` so mypy recognises the
    host-provided ``ingest`` and ``delete`` methods.
    """

    manifest: Manifest
    settings: Settings

    def sync_index(
        self,
        directory: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        user: Any | None = None,
        show_progress: bool = True,
    ) -> dict[str, list[str]]:
        """Reconcile ``directory`` against the manifest.

        Uses the manifest's ``bundle_id`` and the source URI
        independently: a changed file produces a new bundle id but
        the prior bundle id (still in the manifest under the same
        source URI) must be retired so a re-ingest does not double
        index or short-circuit on the wrong checksum.

        The summary is grouped into ``added``, ``modified``,
        ``unchanged``, and ``removed`` lists so the caller can
        report progress.

        Args:
            directory: Directory to walk.
            metadata: Optional per-file metadata.
            user: Optional :class:`User`.
            show_progress: When ``True`` (default), wrap the file loop
                in a tqdm progress bar.

        Returns:
            A summary dict with ``added``, ``modified``, ``unchanged``,
            and ``removed`` lists of source URIs.

        """
        directory = Path(directory)
        if not directory.is_dir():
            raise RagHubError(f"{directory} is not a directory")

        seen: set[str] = set()
        summary: dict[str, list[str]] = {
            "added": [],
            "modified": [],
            "unchanged": [],
            "removed": [],
        }

        files = sorted(p for p in directory.rglob("*") if p.is_file())
        iterator = tqdm(files, desc="Syncing index", disable=not show_progress, unit="file")
        for child in iterator:
            self.sync_one(child, metadata, user, seen, summary)

        for prior_uri in self.manifest.sources():
            if prior_uri in seen:
                continue
            if not prior_uri.startswith(str(directory.resolve())):
                continue
            self.remove_prior(prior_uri, summary)

        self.manifest.save()
        return summary

    def sync_one(
        self,
        child: Path,
        metadata: dict[str, Any] | None,
        user: Any | None,
        seen: set[str],
        summary: dict[str, list[str]],
    ) -> None:
        """Reconcile a single file against the manifest."""
        if not child.is_file():
            return
        uri = str(child.resolve())
        seen.add(uri)
        data = child.read_bytes()
        checksum = sha256_bytes(data)
        prior = self.manifest.get(uri)
        bundle_id = deterministic_id("bundle", uri, checksum)
        if prior is None:
            result = self.ingest(child, metadata=metadata, user=user)
            if isinstance(result, Pipeline) and getattr(result, "error", None) is not None:
                raise IngestionError(result.error or f"failed to ingest {uri}")
            self.manifest.record(uri, bundle_id=bundle_id, checksum=checksum)
            summary["added"].append(uri)
            return
        if prior.get("checksum") == checksum:
            summary["unchanged"].append(uri)
            return
        # Changed file: retire the prior bundle id before re-ingesting.
        prior_bundle_id = str(prior.get("bundle_id", ""))
        result = self.ingest(child, metadata=metadata, force=True, user=user)
        if isinstance(result, Pipeline) and getattr(result, "error", None) is not None:
            raise IngestionError(result.error or f"failed to ingest {uri}")
        if prior_bundle_id and prior_bundle_id != bundle_id:
            self.delete(prior_bundle_id)
        self.manifest.record(uri, bundle_id=bundle_id, checksum=checksum)
        summary["modified"].append(uri)

    def remove_prior(
        self,
        prior_uri: str,
        summary: dict[str, list[str]],
    ) -> None:
        """Drop a manifest entry that no longer has a file on disk."""
        prior_record = self.manifest[prior_uri]
        bundle_id = str(prior_record.get("bundle_id", ""))
        self.delete(bundle_id)
        self.manifest.remove(prior_uri)
        summary["removed"].append(prior_uri)
