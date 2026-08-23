# raghub — operations

Run raghub in production with a single daemon, no `.env` files,
and encrypted-on-disk workspaces.

## Layout

```
$RAGHUB_HOME/                         # default: ~/.raghub
├── registry.db                       # top-level WorkspaceRegistry
└── workspaces/
    └── wsp_<id>/
        ├── workspace.db              # encrypted, all data
        ├── workspace.db-shm          # WAL peer
        ├── workspace.db-wal          # WAL peer
        └── snapshots/                # LocalFileStorage root
```

## Environment

| variable | default | purpose |
| --- | --- | --- |
| `RAGHUB_WORKSPACE_HOME` | `~/.raghub` | registry + workspace root |
| `RAGHUB_API_PORT` | `3000` | Hono listener |
| `RAGHUB_API_BASE` | `http://localhost:3000` | web proxy target |
| `RAGHUB_JWT_SECRET` | dev-only fallback | HS256 signing key (≥32 bytes) |

The **only** place an LLM API key lives is `workspace_settings.llm`
inside the encrypted `workspace.db`. The browser never receives the
plaintext key; the API reads it once per request via the workspace
pool.

## First run

```bash
pnpm install
pnpm --filter @raghub/core build
pnpm --filter @raghub/api build
RAGHUB_JWT_SECRET="$(openssl rand -hex 32)" \
  RAGHUB_WORKSPACE_HOME=/var/lib/raghub \
  pnpm --filter @raghub/api start
```

In a second terminal:

```bash
RAGHUB_API_BASE=http://localhost:3000 \
  pnpm --filter @raghub/web dev
```

Open `http://localhost:3001/onboarding`. Five steps:

1. workspace name
2. admin email + password (bcrypt)
3. LLM provider/model/apiKey/baseUrl — encrypted at rest
4. workspace passphrase — required to unlock settings on every login
5. confirm

After login, the browser carries two cookies: `raghub_token` (JWT,
24h) and `raghub_passphrase` (workspace passphrase, 24h). Both are
cleared on logout.

## Migrations

`Workspace.open()` runs `runMigrations()` automatically. New SQL
landed in a release lands in `packages/core/src/migrations.ts` as
the next `0009_*` entry. The runner is idempotent — applied ids
land in `schema_migrations` and are skipped on re-open.

To inspect:

```sql
SELECT * FROM schema_migrations ORDER BY applied_at DESC;
```

## Backup

The encrypted workspace.db is the only state worth backing up.
SQLite is happy to back up while running (WAL mode):

```bash
sqlite3 /var/lib/raghub/workspaces/wsp_xxx/workspace.db ".backup /backup/wsp_xxx.db"
```

The backup is **still encrypted** — the passphrase is required to
open it on a fresh host. Store the passphrase in your password
manager, separate from the .db backup.

## Restore

Drop the backup in place:

```bash
systemctl stop raghub-api
cp /backup/wsp_xxx.db /var/lib/raghub/workspaces/wsp_xxx/workspace.db
systemctl start raghub-api
```

The registry at `$RAGHUB_HOME/registry.db` also stores the
workspace's on-disk path; if you move the .db file, re-register:

```sql
UPDATE workspace_directory SET path = '/new/path/workspace.db' WHERE workspace_id = 'wsp_xxx';
```

## Health checks

| endpoint | purpose | probe |
| --- | --- | --- |
| `GET /health` | liveness — process alive | every 10s |
| `GET /readyz` | readiness — registry reachable, pool not locked | every 30s |
| `GET /v1/diagnostics` | per-workspace state (auth required) | on demand |

`/health` returns `{ ok: true, version: '...' }` and is suitable for
a Kubernetes liveness probe.

## Logs

The daemon logs to stdout in JSON-ish plain text. Pipe to a
collector (`journalctl`, `vector`, `fluentbit`) and route by event
kind. Hook points (P-17) emit structured events:

```
[hook] beforeRetrieve role=vector
[hook] afterRetrieve role=vector hits=12
[hook] beforeLLM question_len=88
[hook] afterLLM answer_len=512
```

## Rate limits

The middleware applies per-IP (60/min) and per-workspace
(600/min) sliding windows in memory. Restart the daemon to reset.
Production deployments behind a reverse proxy should add a Redis-
backed bucket at the proxy layer.

## Troubleshooting

### "invalid passphrase" on every login

The user typed the wrong workspace passphrase. The browser never
sees the key, so there's no client-side debug path. Reset:

1. Stop the daemon.
2. Move the .db aside: `mv workspace.db workspace.db.broken`.
3. Re-onboard with a new passphrase — the old data is in
   `workspace.db.broken` and **cannot be decrypted** without the
   original passphrase.

### Slow query

Inspect `query_stats`:

```sql
SELECT * FROM audit_event WHERE kind = 'ingest.failure' ORDER BY created_at DESC LIMIT 20;
```

### Migration fails

The migration runner wraps each statement in its own transaction
but re-throws on failure. The workspace remains open with whatever
schema it had before. Inspect:

```sql
SELECT * FROM schema_migrations ORDER BY applied_at DESC LIMIT 5;
```

A migration that fails to apply leaves `schema_migrations` empty for
that id — the next `Workspace.open()` will re-attempt it.

### Stale pool handles

The workspace pool holds up to 64 handles. On overflow the oldest
LRU is closed and reopened on demand. If a passphrase change is
required mid-session (rotation), restart the daemon.

## Security checklist

- [ ] `RAGHUB_JWT_SECRET` is a fresh 32+ byte secret per deployment
- [ ] `workspace.db` files are stored on a filesystem with `chmod 600` permissions
- [ ] TLS terminated at the reverse proxy (nginx, Caddy, ALB)
- [ ] `Set-Cookie: ...; HttpOnly; Secure; SameSite=Strict` at the proxy (TODO P-07 harden)
- [ ] Audit log exported off-host (vector, splunk, loki)
- [ ] Backups stored in a separate security domain from the passphrase

## Single-process vs. multi-process

The default daemon is single-process. SQLite WAL allows multiple
readers + one writer per database, so horizontal scaling means
deploying **multiple stateless API replicas** behind a load
balancer, each pointing at the same shared `$RAGHUB_HOME`. The
in-memory rate-limit bucket and workspace pool become per-replica
state — that's fine for a single-workspace deployment but degrades
in a multi-tenant SaaS scenario (out of scope per the local-first
product brief).