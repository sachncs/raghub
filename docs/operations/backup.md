# Backup & Restore

RAGHub production state lives in three places:

| State | Location | Backing store |
|---|---|---|
| SQLite registry (documents, chunks) | `RAG_REGISTRY_PATH` | `./data/registry.db` |
| SQLite session store (opaque session tokens) | `RAG_SESSIONS_PATH` | `./data/sessions.db` |
| Qdrant vectors | `QDRANT_URL` HTTP API | named volume `raghub_qdrant_data` |
| Document upload blob cache | `RAG_DATA_DIR/images` | `./data/images` (optional) |

In Compose, the API and UI share the named volume `raghub_data`; the
Qdrant service owns `raghub_qdrant_data` and `raghub_qdrant_snapshots`.
Back up every one of these. Skipping any one of them is a partial
backup and will fail to restore end-to-end.

## One-shot backup script

The same script backs up both the SQLite files (via plain file copy)
and the Qdrant collection (via the Qdrant HTTP snapshot API):

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="${BACKUP_ROOT:-./backups/$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_DIR="${RAG_DATA_DIR:-./data}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"

mkdir -p "$BACKUP_ROOT/sqlite"
cp -a "$DATA_DIR"/*.db "$BACKUP_ROOT/sqlite/" 2>/dev/null || true
cp -a "$DATA_DIR/images" "$BACKUP_ROOT/sqlite/" 2>/dev/null || true

# Qdrant snapshot. The collection name follows the AppSettings value;
# the default profile uses "raghub".
COLLECTION="${RAGHUB_QDRANT_COLLECTION:-raghub}"
curl -fsS -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots" \
    -o "$BACKUP_ROOT/qdrant-snapshot.json"

# Tarball for off-host storage.
tar -C "$BACKUP_ROOT" -czf "$BACKUP_ROOT.tar.gz" .
echo "Backup written to $BACKUP_ROOT.tar.gz"
```

Run as a daily cron, or schedule it through your orchestrator
(GitHub Actions schedule, Kubernetes CronJob, etc.).

## Restore

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKUP="$1"  # path to .tar.gz
RESTORE_DIR="${RESTORE_DIR:-./data}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
COLLECTION="${RAGHUB_QDRANT_COLLECTION:-raghub}"

# Stop the stack so the SQLite files are not locked.
docker compose -f docker-compose.yml --profile production down

mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP" -C /tmp/restore
cp -a /tmp/restore/sqlite/* "$RESTORE_DIR/"

# Qdrant: upload the snapshot.
curl -fsS -X POST \
    "$QDRANT_URL/collections/$COLLECTION/snapshots/upload?priority=snapshot" \
    -H "Content-Type: multipart/form-data" \
    -F "snapshot=@/tmp/restore/qdrant-snapshot.json"

docker compose -f docker-compose.yml --profile production up -d
```

## Retention

Keep the last 30 daily backups and the last 12 monthly backups. Older
snapshots can be deleted with the Qdrant snapshot API:

```bash
curl -fsS -X DELETE "$QDRANT_URL/collections/$COLLECTION/snapshots/<name>"
```

The `raghub_qdrant_snapshots` volume is itself a separate named
volume, so Qdrant's own snapshot rotation runs alongside the host
backup. Both should be kept.

## Cross-region

Push `$BACKUP_ROOT.tar.gz` to object storage (S3/GCS/Azure Blob) with
a server-side encryption key. The `.env` file is **not** backed up
here; secrets are managed out-of-band (see the deployment guide).
