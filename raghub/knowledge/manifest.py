"""Source manifest for incremental indexing.

Persists the SHA-256 checksum of every ingested source so
:class:`raghub.RAG.sync_index` can detect newly added, modified, and
removed files without re-ingesting the entire corpus.

The manifest is a JSON file keyed by ``source_uri`` and storing the
bundle id + checksum. When the file is missing the manifest is
rebuilt from the in-memory knowledge repository.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any


class SourceManifest:
    """Persistent index of source URIs and their checksums."""

    def __init__(self, path: Path | str) -> None:
        """Initialise the manifest at ``path``."""
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self.records = {str(k): v for k, v in payload.items() if isinstance(v, dict)}
        except json.JSONDecodeError:
            self.records = {}

    def save(self) -> None:
        """Persist the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(self, source_uri: str, *, bundle_id: str, checksum: str) -> None:
        """Record or update a source."""
        self.records[source_uri] = {"bundle_id": bundle_id, "checksum": checksum}

    def remove(self, source_uri: str) -> None:
        """Remove a source from the manifest."""
        self.records.pop(source_uri, None)

    def __contains__(self, source_uri: str) -> bool:
        """Check whether a source URI is tracked in the manifest.

        Args:
            source_uri: The source URI to look up.

        Returns:
            ``True`` if the URI has been recorded.
        """
        return source_uri in self.records

    def __getitem__(self, source_uri: str) -> dict[str, Any]:
        """Retrieve the record for a source URI.

        Args:
            source_uri: The source URI.

        Returns:
            The record dict (``bundle_id``, ``checksum``, …).

        Raises:
            KeyError: When the URI is not tracked.
        """
        return self.records[source_uri]

    def items(self) -> Iterable[tuple[str, dict[str, Any]]]:
        """Yield ``(source_uri, record)`` pairs."""
        return self.records.items()

    def sources(self) -> list[str]:
        """Return the list of known source URIs."""
        return list(self.records.keys())


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return sha256(data).hexdigest()


__all__ = ["SourceManifest", "sha256_bytes"]
