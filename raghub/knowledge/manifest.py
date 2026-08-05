"""Persistent source manifest and small content-hash helpers.

The manifest is the on-disk index of source URIs and their
bundle checksums; :func:`sha256_bytes` is the byte-level hash
helper used by the ingest pipeline and the OKF layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from raghub.coroutines import capture


class Manifest:
    """Persistent index of source URIs and their checksums.

    On disk format (v2):

    .. code-block:: json

        {"version": 2,
         "records": {"source_uri": {"bundle_id": "...",
                                       "checksum": "...",
                                       "chunk_ids": [...]}}}
    """

    CURRENT_VERSION = 2

    def __init__(self, path: Path | str) -> None:
        """Initialise the manifest at ``path``."""
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        self.version: int = self.CURRENT_VERSION
        self.load()

    def load(self) -> None:
        """Read the JSON manifest from disk into ``self.records``.

        Version-pins the on-disk format. v1 files (no ``version``)
        are migrated in-memory: bare record dicts are kept as-is.
        """
        if not self.path.exists():
            return
        text, text_error = capture(self.path.read_text, encoding="utf-8")
        if text_error is not None:
            self.records = {}
            return
        payload, json_error = capture(json.loads, text or "{}")
        if json_error is not None or not isinstance(payload, dict):
            self.records = {}
            return
        version = int(payload.get("version", 1)) if isinstance(payload.get("version"), int) else 1
        self.version = version
        records = payload.get("records", payload)
        if isinstance(records, dict):
            self.records = {str(k): v for k, v in records.items() if isinstance(v, dict)}
        else:
            self.records = {}

    def save(self) -> None:
        """Persist the manifest to disk.

        Always writes the current version (C2). Older v1 readers see
        the ``records`` key via the version-1 fallback path.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": self.CURRENT_VERSION, "records": self.records}
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def record(self, source_uri: str, *, bundle_id: str, checksum: str) -> None:
        """Record or update a source."""
        self.records[source_uri] = {"bundle_id": bundle_id, "checksum": checksum}

    def remove(self, source_uri: str) -> None:
        """Remove a source from the manifest."""
        self.records.pop(source_uri, None)

    def __contains__(self, source_uri: str) -> bool:
        """Check whether a source URI is tracked in the manifest."""
        return source_uri in self.records

    def __getitem__(self, source_uri: str) -> dict[str, Any]:
        """Retrieve the record for a source URI."""
        return self.records[source_uri]

    def get(self, source_uri: str) -> dict[str, Any] | None:
        """Return the record for ``source_uri`` or ``None``."""
        return self.records.get(source_uri)

    def items(self) -> Iterable[tuple[str, dict[str, Any]]]:
        """Yield ``(source_uri, record)`` pairs."""
        return self.records.items()

    def sources(self) -> list[str]:
        """Return the list of known source URIs."""
        return list(self.records.keys())


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of ``data``."""
    return sha256(data).hexdigest()
