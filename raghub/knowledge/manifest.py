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
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


class SourceManifest:
    """Persistent index of source URIs and their checksums."""

    def __init__(self, path: Path | str) -> None:
        """Initialise the manifest at ``path``."""
        self._path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._records = {
                    str(k): v for k, v in payload.items() if isinstance(v, dict)
                }
        except json.JSONDecodeError:
            self._records = {}

    def save(self) -> None:
        """Persist the manifest to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._records, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(self, source_uri: str, *, bundle_id: str, checksum: str) -> None:
        """Record or update a source."""
        self._records[source_uri] = {"bundle_id": bundle_id, "checksum": checksum}

    def remove(self, source_uri: str) -> None:
        """Remove a source from the manifest."""
        self._records.pop(source_uri, None)

    def __contains__(self, source_uri: str) -> bool:
        return source_uri in self._records

    def __getitem__(self, source_uri: str) -> dict[str, Any]:
        return self._records[source_uri]

    def items(self) -> Iterable[tuple[str, dict[str, Any]]]:
        """Yield ``(source_uri, record)`` pairs."""
        return self._records.items()

    def sources(self) -> list[str]:
        """Return the list of known source URIs."""
        return list(self._records.keys())


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return sha256(data).hexdigest()


__all__ = ["SourceManifest", "sha256_bytes"]
