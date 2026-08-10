"""Domain package: ``raghub.archive``.

Re-exports the implementation in :mod:`raghub.archive._impl`.
"""

from __future__ import annotations

from raghub.archive._impl import (
    MANIFEST_FORMAT_VERSION,
    ArchiveCorruptionError,
    ArchiveEntry,
    ArchiveManifest,
    ArchiveStore,
    LocalArchiveStore,
    collect_with_vacuum,
    create_snapshot,
    extract_member,
    hmac_compare_digest,
    make_tar,
    restore_snapshot,
    signing_key,
    vacuum_sqlite,
    verify_archive,
    write_archive,
    zstd_compress,
    zstd_decompress,
)

__all__ = [
    "MANIFEST_FORMAT_VERSION",
    "ArchiveCorruptionError",
    "ArchiveEntry",
    "ArchiveManifest",
    "ArchiveStore",
    "LocalArchiveStore",
    "collect_with_vacuum",
    "create_snapshot",
    "extract_member",
    "hmac_compare_digest",
    "make_tar",
    "restore_snapshot",
    "signing_key",
    "vacuum_sqlite",
    "verify_archive",
    "write_archive",
    "zstd_compress",
    "zstd_decompress",
]
