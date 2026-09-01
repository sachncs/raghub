# Workspace model

A workspace is the top-level multi-tenant boundary. It is one SQLite file
(`workspace.db`) plus an optional `snapshots/` directory and any file blobs
managed by `FsLocalFileStorage`. The active implementation uses
**RowLevel isolation only**: a tenant is a logical row with a stable
`workspace_id` that every store query filters on.

## Opening a workspace

`@revex/core` exposes two openers:

### `openWorkspace`

```ts
import { openWorkspace } from '@revex/core';

const handle = await openWorkspace({ path: './workspace.db' });
// handle: { path, db, id, close() }
```

Opens in plaintext mode (dev/single-tenant). Runs migrations and seeds a
default `workspace` row.

### `openEncryptedWorkspace`

```ts
import { openEncryptedWorkspace } from '@revex/core';

const handle = await openEncryptedWorkspace({
  path: './workspace.db',
  passphrase: 'hunter2-...',
});
```

Derives a 32-byte AES-256-GCM key from the passphrase via scrypt
(N=2¹⁵, r=8, p=1). All `workspace_settings` rows are encrypted at rest. The
`EncryptedField` wrapper stores a verifier ciphertext so a wrong passphrase
fails fast before any data is touched.

## Storage layout

| Path | Purpose |
|---|---|
| `$REVEX_WORKSPACE_HOME/registry.db` | Workspace registry (multi-tenant). |
| `$REVEX_WORKSPACE_HOME/registry.json` | CLI-readable registry snapshot. |
| `$REVEX_WORKSPACE_HOME/files/` | File blobs via `FsLocalFileStorage`. |
| `$REVEX_WORKSPACE_HOME/wsp_<name>/workspace.db` | per-workspace SQLite file. |

`REVEX_WORKSPACE_HOME` defaults to `~/.revex`.

## Stores

All `Sqlite*Store` classes share the `Database` handle from
`Workspace.open(path)`:

- `SqliteUserStore` — users.
- `SqliteDocumentStore` — document metadata + lifecycle.
- `SqliteDocumentPrincipalStore` — per-document ACL.
- `SqliteWorkspaceMemberStore` — workspace membership + roles.
- `SqliteRoleStore` / `SqliteGroupStore` — named roles & groups.
- `SqliteSessionStore` / `SqliteConversationStore` — sessions + turns.
- `SqliteJobQueue` — background ingestion jobs.
- `SqliteAuditEventStore` — audit log.
- `SqliteFeedbackStore` — per-turn feedback.
- `SqliteWorkspaceMemoryStore` — per-session memory facts.
- `SqliteVecStore` — vector + FTS5 keyword store.

## Workspace registry

`openFileWorkspaceRegistry({ registryPath }, openConnection)` returns a
`WorkspaceRegistry` with `register`, `resolve`, `unregister`, and `list`.
The API uses it plus `WorkspacePool` (caches up to 64 unlocked handles with a
30s lock) to resolve workspaces per request.

## Tenancy helpers

`@revex/core` exposes async-local-storage context helpers:

- `currentWorkspace()` / `requireWorkspace()`
- `runWithTenant(ctx, fn)` / `runWithWorkspaceAsync(workspaceId, fn)`
- `tenantContext`

## Migration

`runMigrations(db)` applies `MIGRATIONS` (baseline schema, workspace
directory, ...). `lastAppliedId(db)` reports the current migration level.