> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Backup & Restore

Revex production state lives entirely on the local filesystem. The
`revex backup` and `revex backup restore` sub-commands capture and
restore that state from a single HMAC-signed archive — no separate
database server is involved.

| State | Location | Backing store |
|---|---|---|
| SQLite registry (documents, chunks) | `RAG_REGISTRY_PATH` | `./data/registry.db` |
| SQLite session store (opaque session tokens) | `RAG_SESSIONS_PATH` | `./data/sessions.db` |
| SQLite vector store | `RAG_VECTORSTORE_PATH` (or `data/vecstore.sqlite`) | local SQLite file |
| Document upload blob cache | `RAG_DATA_DIR/images` | `./data/images` (optional) |

For production deployments that move the vector store to PostgreSQL,
the `revex migrate pgvector --dsn <dsn>` command initialises the
`PgVectorStore` schema and indexes on the target database; the
`revex backup` archive captures SQLite state and document manifests,
and the pgvector store is replicated through standard PostgreSQL
backup tooling (`pg_dump` / `pg_basebackup`).

Skipping any one of these surfaces is a partial backup and will
fail to restore end-to-end.

## One-shot backup

The CLI wraps the snapshot writer in `raghub.archive`:

```bash
revex backup create -o /var/backups/raghub-$(date -u +%Y%m%dT%H%M%SZ).tar.zst
```

This writes a single `tar.zst` archive containing every component
file plus a signed manifest. The manifest lists each entry with its
SHA-256 and `kind`; the manifest itself is signed with the
configured HMAC key (`RAG_TENANT_SIGNING_KEY`, or the key auto-loaded
from the secrets store).

Inspect an existing archive without restoring:

```bash
revex backup verify --input /var/backups/raghub-20260805T120000Z.tar.zst
```

`verify` checks both the HMAC signature and every per-file SHA-256;
it exits non-zero on the first mismatch.

Schedule the `create` command through your orchestrator of choice
(GitHub Actions schedule, Kubernetes `CronJob`, plain `cron`, etc.).
For PostgreSQL deployments, pair the schedule with a standard
`pg_dump` job.

## Restore

```bash
# 1. Stop any process holding the SQLite files open
#    (the FastAPI server, a background ingest worker, the CLI).
raghub run --host 0.0.0.0 --port 8000      # or stop it via systemd / k8s

# 2. Restore the archive into the data directory.
revex backup restore \
    --input /var/backups/raghub-20260805T120000Z.tar.zst \
    --target-dir /srv/revex/data

# 3. Restart the API.
raghub run --host 0.0.0.0 --port 8000
```

`restore_snapshot` in `raghub.archive` verifies the manifest
signature before any file is written, and reconstructs each entry
from its recorded SHA-256. The vector-store embeddings are restored
only when `--include-embeddings` is passed to the underlying
function; the CLI default omits them so the caller re-derives
embeddings from the source documents on the next ingest pass.

For PostgreSQL deployments, restore the `pg_dump` against the
target database *before* running `revex backup restore` against
the SQLite portion, so the document registry and the vector store
land at the same point in time.

## Retention

Keep the last 30 daily backups and the last 12 monthly backups.
Older archives can be deleted with the standard `rm` / object-store
lifecycle rules. The HMAC key used to sign the manifest must be the
same across `create` and `restore`; rotate it through
`RAG_TENANT_SIGNING_KEY` and re-issue archives out-of-band so older
backups remain verifiable under the previous key while new ones
sign under the new one.

## Cross-region

Push the `tar.zst` archive to object storage (S3/GCS/Azure Blob)
with a server-side encryption key. The `.env` file is **not**
included in the archive; secrets are managed out-of-band (see the
deployment guide).
