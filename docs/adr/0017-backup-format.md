> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# ADR 0017 — Backup format

## Status

Accepted (v0.7.8)

## Context

Revex stores documents, chunks, vectors, and metadata in Postgres (or
SQLite). Operators need a portable, verifiable backup format that can be
restored across versions and environments.

## Decision

Backups use a **deterministic tar archive** with the following layout:

```
backup_<timestamp>_<version>.tar.zst
├── manifest.json          # format_version, checksums, HMAC
├── documents/
│   └── <doc_id>.json
├── chunks/
│   └── <chunk_id>.json
├── vectors/
│   └── <chunk_id>.npy
└── metadata/
    └── tenant.json
```

### manifest.json

```json
{
  "format_version": 1,
  "created_at": "2025-01-15T12:00:00Z",
  "raghub_version": "0.7.8",
  "checksums": {
    "documents/abc123.json": "sha256:…",
    "chunks/def456.json": "sha256:…"
  },
  "hmac": "HMAC-SHA256 over sorted checksums"
}
```

### Compression

- **zstd** (default, level 3) for speed/ratio balance.
- `REVEX_ARCHIVE_COMPRESSION` env var selects `zstd` | `gzip` | `none`.

### Signing

- When `REVEX_ARCHIVE_SIGNING_KEY` is set, the `hmac` field contains
  an HMAC-SHA256 over the lexicographically sorted `checksums` dict.
- Verification: `raghub archive verify <path>`.

## Consequences

- **Rotation required**: `REVEX_ARCHIVE_SIGNING_KEY` must be rotated
  periodically; old keys should be kept for verification only.
- **Deterministic layout**: Same data always produces the same tar
  structure, enabling diff-based backup comparison.
- **Size**: Vector `.npy` files dominate archive size; zstd keeps
  typical backups under 2× the raw vector data.

## Alternatives considered

- **Encrypted tar**: Deferred — encryption at rest is handled by the
  storage layer; archive-level encryption adds key-management burden.
- **Plain zip**: Rejected — no streaming support, worse compression
  ratios for numerical data.
- **pg_dump only**: Rejected — not portable across SQLite and Postgres.
