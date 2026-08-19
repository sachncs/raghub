"""Backup / restore tooling.

Archive format v1:

* ``manifest.json`` — JSON document with ``format_version: 1``
  and a SHA-256 + size entry for every captured file.
* One entry per file (the captured SQLite / JSON / image bytes).
* Whole archive is wrapped in a deterministic ``tar`` and compressed
  with ``zstd``.

``ArchiveManifest`` is version-pinned per ``AGENTS.md`` R6; bumping
``format_version`` is a breaking change.

The local backend stores archives in
``Settings.archive.local_dir``; community backends (S3 / GCS /
Azure) can plug in via the entry-point
``group="raghub.archives"``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tarfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import zstandard
import tarfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raghub.constants import ENV_RAGHUB_ARCHIVE_SIGNING_KEY
from raghub.errors import RagHubError
from raghub.registry import Registry

__all__ = [
    "ArchiveCorruptionError",
    "ArchiveEntry",
    "ArchiveManifest",
    "ArchiveStore",
    "LocalArchiveStore",
    "create_snapshot",
    "restore_snapshot",
]

MANIFEST_FORMAT_VERSION = 1


class ArchiveCorruptionError(RagHubError, RuntimeError):
    """Raised when an archive's integrity check fails."""


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """One captured file inside an archive."""

    path: str
    sha256: str
    size_bytes: int
    # One of: registry, manifest, queue, feedback, sessions, image_store, embeddings, metadata
    kind: str


@dataclass(frozen=True, slots=True)
class ArchiveManifest:
    """The manifest describing one archive."""

    format_version: int
    app_version: str
    created_at: datetime
    tenant_ids: list[str]
    entries: list[ArchiveEntry]
    signature: str  # HMAC-SHA256 of the canonical JSON form

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (no signature yet)."""
        return {
            "format_version": self.format_version,
            "app_version": self.app_version,
            "created_at": self.created_at.isoformat(),
            "tenant_ids": list(self.tenant_ids),
            "entries": [
                {
                    "path": e.path,
                    "sha256": e.sha256,
                    "size_bytes": e.size_bytes,
                    "kind": e.kind,
                }
                for e in self.entries
            ],
        }

    def canonical_bytes(self) -> bytes:
        """Return the canonical byte form used for HMAC signing."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return payload.encode("utf-8")

    def expected_signature(self, key: bytes) -> str:
        """Return the HMAC-SHA256 of the canonical bytes under ``key``."""
        import hmac

        return hmac.new(key, self.canonical_bytes(), hashlib.sha256).hexdigest()

    def verify_signature(self, key: bytes) -> None:
        """Raise :class:`ArchiveCorruptionError` when the signature mismatches."""
        if not hmac_compare_digest(self.signature, self.expected_signature(key)):
            raise ArchiveCorruptionError("manifest signature mismatch")


def hmac_compare_digest(a: str, b: str) -> bool:
    """Constant-time comparison wrapper for :func:`hmac.compare_digest`."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def signing_key() -> bytes:
    """Return the HMAC key from env or raise :class:`MissingDepError`."""
    from raghub.errors import MissingDepError

    raw = os.getenv(ENV_RAGHUB_ARCHIVE_SIGNING_KEY)
    if not raw:
        raise MissingDepError(
            ENV_RAGHUB_ARCHIVE_SIGNING_KEY,
            "set RAGHUB_ARCHIVE_SIGNING_KEY to a 32+ byte secret.",
        )
    return raw.encode("utf-8")


class ArchiveStore(Registry):
    """Polymorphic base for archive storage backends.

    Concrete backends (LocalArchiveStore, …) register via
    ``@ArchiveStore.register`` and implement the four methods.
    """

    name: str = "archive_store"

    def put(self, key: str, data: bytes) -> None:
        """Write ``data`` under ``key``."""
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``."""
        raise NotImplementedError

    def list(self, prefix: str = "") -> list[str]:
        """Return all keys whose name begins with ``prefix``."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove ``key`` from the archive."""
        raise NotImplementedError


@ArchiveStore.register("local")
class LocalArchiveStore(ArchiveStore):
    """Stores archives on the local filesystem."""

    name = "local"

    def __init__(self, base_path: str | Path) -> None:
        """Persist archives under ``base_path``, creating the directory."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        """Write ``data`` to ``key``."""
        path = self.base_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes:
        """Return the bytes for ``key``."""
        return (self.base_path / key).read_bytes()

    def list(self, prefix: str = "") -> list[str]:
        """Return the keys under ``prefix``."""
        return [
            str(p.relative_to(self.base_path))
            for p in self.base_path.rglob(f"{prefix}*")
            if p.is_file()
        ]

    def delete(self, key: str) -> None:
        """Delete ``key``."""
        (self.base_path / key).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Bundle collectors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BundleComponents:
    """The set of files to include in a snapshot."""

    files: dict[str, bytes] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)
    tenant_ids: list[str] = field(default_factory=list)


def collect_registry_json(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the JSON registry if present."""
    path = data_dir / "registry.json"
    if not path.exists():
        return None
    return path.read_bytes(), "registry"


def collect_sessions_db(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the sessions SQLite DB if present."""
    path = data_dir / "sessions.db"
    if not path.exists():
        return None
    return path.read_bytes(), "sessions"


def collect_queue_db(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the queue SQLite DB if present."""
    path = data_dir / "ingestion_jobs.db"
    if not path.exists():
        return None
    return path.read_bytes(), "queue"


def collect_feedback_db(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the feedback SQLite DB if present."""
    path = data_dir / "feedback.db"
    if not path.exists():
        return None
    return path.read_bytes(), "feedback"


def collect_audit_db(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the audit SQLite DB if present."""
    path = data_dir / "audit.db"
    if not path.exists():
        return None
    return path.read_bytes(), "audit"


def collect_manifests(data_dir: Path) -> list[tuple[str, bytes, str]]:
    """Return every ``manifest.json`` under ``data_dir``."""
    out: list[tuple[str, bytes, str]] = []
    for path in data_dir.rglob("manifest.json"):
        rel = str(path.relative_to(data_dir))
        out.append((rel, path.read_bytes(), "manifest"))
    return out


def collect_image_store(data_dir: Path) -> list[tuple[str, bytes, str]]:
    """Return every file under the image store directory."""
    out: list[tuple[str, bytes, str]] = []
    image_dir = data_dir / "images"
    if not image_dir.is_dir():
        return out
    for path in image_dir.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(data_dir))
            out.append((rel, path.read_bytes(), "image_store"))
    return out


def collect_tenant_secrets_db(data_dir: Path) -> tuple[bytes, str] | None:
    """Return the tenant secrets SQLite DB if present."""
    path = data_dir / "tenant_secrets.db"
    if not path.exists():
        return None
    return path.read_bytes(), "metadata"


def create_snapshot(
    data_dir: str | Path,
    *,
    app_version: str = "0.7.8",
) -> tuple[ArchiveManifest, dict[str, bytes]]:
    """Capture every component under ``data_dir`` into an archive payload.

    Returns:
        ``(manifest, files)`` — the manifest lists every captured
        file with its SHA-256 + size; ``files`` maps ``path -> bytes``
        so the caller can wrap them in a tarball.

    """
    data_dir = Path(data_dir)
    files: dict[str, bytes] = {}
    kinds: dict[str, str] = {}
    tenant_ids: set[str] = set()

    collectors: list[Callable[[Path], tuple[bytes, str] | None]] = [
        collect_registry_json,
        collect_sessions_db,
        collect_queue_db,
        collect_feedback_db,
        collect_audit_db,
        collect_tenant_secrets_db,
    ]
    for collector in collectors:
        entry = collector(data_dir)
        if entry is None:
            continue
        contents, kind = entry
        name = collector.__name__.replace("collect_", "").replace("_db", ".db")
        files[name] = contents
        kinds[name] = kind

    for rel, contents, kind in collect_manifests(data_dir):
        files[rel] = contents
        kinds[rel] = kind

    for rel, contents, kind in collect_image_store(data_dir):
        files[rel] = contents
        kinds[rel] = kind

    entries = [
        ArchiveEntry(
            path=path,
            sha256=hashlib.sha256(contents).hexdigest(),
            size_bytes=len(contents),
            kind=kinds[path],
        )
        for path, contents in sorted(files.items())
    ]

    manifest = ArchiveManifest(
        format_version=MANIFEST_FORMAT_VERSION,
        app_version=app_version,
        created_at=datetime.now(UTC),
        tenant_ids=sorted(tenant_ids),
        entries=entries,
        signature="",  # filled below
    )
    manifest = ArchiveManifest(
        format_version=manifest.format_version,
        app_version=manifest.app_version,
        created_at=manifest.created_at,
        tenant_ids=manifest.tenant_ids,
        entries=manifest.entries,
        signature=manifest.expected_signature(signing_key()),
    )
    return manifest, files


def restore_snapshot(
    archive_path: str | Path,
    target_dir: str | Path,
    *,
    signing_key_value: bytes | None = None,
    include_embeddings: bool = False,
) -> None:
    """Restore an archive produced by :func:`create_snapshot`.

    Args:
        archive_path: Path to the ``.tar.zst`` archive.
        target_dir: Destination directory. Created if missing.
        signing_key_value: Optional override for the HMAC key
            (defaults to :func:`signing_key`).
        include_embeddings: When ``False``, the embeddings file is
            dropped on restore; the caller re-derives embeddings
            from source documents.

    Raises:
        ArchiveCorruptionError: When the signature or any per-file
            SHA-256 does not match.

    """
    archive_path = Path(archive_path)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw = archive_path.read_bytes()
    tar_bytes = zstd_decompress(raw)

    manifest_bytes = extract_member(tar_bytes, "manifest.json")
    manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
    if int(manifest_dict["format_version"]) != MANIFEST_FORMAT_VERSION:
        raise ArchiveCorruptionError(
            f"unsupported archive format_version: " f"{manifest_dict['format_version']!r}"
        )
    manifest = ArchiveManifest(
        format_version=int(manifest_dict["format_version"]),
        app_version=str(manifest_dict["app_version"]),
        created_at=datetime.fromisoformat(manifest_dict["created_at"]),
        tenant_ids=list(manifest_dict["tenant_ids"]),
        entries=[
            ArchiveEntry(
                path=e["path"],
                sha256=e["sha256"],
                size_bytes=int(e["size_bytes"]),
                kind=e["kind"],
            )
            for e in manifest_dict["entries"]
        ],
        signature=str(manifest_dict["signature"]),
    )

    key = signing_key_value if signing_key_value is not None else signing_key()
    manifest.verify_signature(key)

    for entry in manifest.entries:
        if not include_embeddings and entry.kind == "embeddings":
            continue
        data = extract_member(tar_bytes, entry.path)
        sha = hashlib.sha256(data).hexdigest()
        if sha != entry.sha256:
            raise ArchiveCorruptionError(
                f"sha256 mismatch for {entry.path}: expected {entry.sha256}, got {sha}"
            )
        target = target_dir / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def zstd_decompress(data: bytes) -> bytes:
    """Decompress ``data`` with zstd."""
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 32)


def zstd_compress(data: bytes) -> bytes:
    """Compress ``data`` with zstd."""
    return zstandard.ZstdCompressor().compress(data)


def extract_member(tar_bytes: bytes, member_path: str) -> bytes:
    """Return the bytes of ``member_path`` from a tar archive.

    Rejects absolute paths and ``..`` traversal.
    """
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tar:
        for member in tar.getmembers():
            if member.name != member_path:
                continue
            if member.name.startswith("/") or ".." in member.name.split("/"):
                raise ArchiveCorruptionError(f"path traversal detected in archive: {member.name!r}")
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ArchiveCorruptionError(f"member {member.name!r} is not a regular file")
            return extracted.read()
    raise ArchiveCorruptionError(f"member not found: {member_path!r}")


def make_tar(
    manifest: ArchiveManifest,
    files: dict[str, bytes],
) -> bytes:
    """Build a deterministic tar archive from ``manifest`` + ``files``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("manifest.json")
        manifest_bytes = json.dumps(
            manifest.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for entry in manifest.entries:
            data = files[entry.path]
            info = tarfile.TarInfo(entry.path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def write_archive(
    manifest: ArchiveManifest,
    files: dict[str, bytes],
    output_path: str | Path,
) -> None:
    """Build the archive, compress, and write to ``output_path``."""
    tar_bytes = make_tar(manifest, files)
    compressed = zstd_compress(tar_bytes)
    Path(output_path).write_bytes(compressed)


def verify_archive(archive_path: str | Path) -> None:
    """Verify ``archive_path`` integrity without restoring.

    Raises:
        ArchiveCorruptionError: When the signature or any per-file
            SHA-256 does not match.

    """
    archive_path = Path(archive_path)
    raw = archive_path.read_bytes()
    tar_bytes = zstd_decompress(raw)

    manifest_bytes = extract_member(tar_bytes, "manifest.json")
    manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
    if int(manifest_dict["format_version"]) != MANIFEST_FORMAT_VERSION:
        raise ArchiveCorruptionError(
            f"unsupported archive format_version: " f"{manifest_dict['format_version']!r}"
        )
    manifest = ArchiveManifest(
        format_version=int(manifest_dict["format_version"]),
        app_version=str(manifest_dict["app_version"]),
        created_at=datetime.fromisoformat(manifest_dict["created_at"]),
        tenant_ids=list(manifest_dict["tenant_ids"]),
        entries=[
            ArchiveEntry(
                path=e["path"],
                sha256=e["sha256"],
                size_bytes=int(e["size_bytes"]),
                kind=e["kind"],
            )
            for e in manifest_dict["entries"]
        ],
        signature=str(manifest_dict["signature"]),
    )
    manifest.verify_signature(signing_key())
    for entry in manifest.entries:
        data = extract_member(tar_bytes, entry.path)
        sha = hashlib.sha256(data).hexdigest()
        if sha != entry.sha256:
            raise ArchiveCorruptionError(
                f"sha256 mismatch for {entry.path}: expected {entry.sha256}, got {sha}"
            )


# ---------------------------------------------------------------------------
# SQLite migration helpers
# ---------------------------------------------------------------------------


def vacuum_sqlite(path: Path) -> None:
    """Run ``VACUUM`` on a SQLite file to compact before backup."""
    if not path.exists():
        return
    with sqlite3.connect(path) as conn, suppress(sqlite3.DatabaseError):
        conn.execute("VACUUM")


def collect_with_vacuum(
    data_dir: Path,
    *,
    sqlite_paths: tuple[Path, ...] = (),
) -> dict[str, bytes]:
    """VACUUM every SQLite file under ``data_dir`` then read its bytes."""
    for path in sqlite_paths:
        vacuum_sqlite(path)
    return {str(path.relative_to(data_dir)): path.read_bytes() for path in data_dir.rglob("*.db")}
