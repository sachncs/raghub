# raghub — workspace model

A **workspace** is the unit of isolation in raghub. Every workspace has
exactly one `workspace.db` file. Documents, groups, roles, memory
facts, sessions, ingestion jobs, the audit log, and any encrypted
configuration live in that single SQLite file.

## Storage layout

```
~/revex/
  wsp_<id>/
    workspace.db          # single SQLite file (better-sqlite3 v12)
    snapshots/            # LocalFileStorage root for session snapshots
```

## Opening a workspace

```ts
import { openEncryptedWorkspace } from '@revex/core';

const handle = await openEncryptedWorkspace({
  path: `${dir}/workspace.db`,
  passphrase: 'correct horse battery staple',
});

handle.db;            // shared better-sqlite3 Database handle
handle.settings;      // WorkspaceSettingsStore (encrypted at rest)
handle.encryption;    // 'plaintext' | 'passphrase-aes-256-gcm'
handle.close();       // idempotent
```

When `passphrase` is provided on first open, scrypt (N=2¹⁵, r=8, p=1)
derives a 32-byte key, the `workspace_keycheck` table is seeded with a
verifier ciphertext, and `workspace_settings` rows are stored as
AES-256-GCM `{ nonce, ciphertext }` blobs. Subsequent opens with the
wrong passphrase throw `ConfigurationError` (mapped to HTTP 500 in
the API layer, surfaced to the browser as "invalid passphrase" 401
during login).

When `passphrase` is omitted, the workspace runs in **plaintext
mode** — `workspace_settings` rows hold raw JSON. Use this only for
local development or disposable workspaces.

## Schema (DDL highlights)

```sql
CREATE TABLE workspace (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  plan TEXT NOT NULL DEFAULT 'free',
  created_at INTEGER NOT NULL
);

CREATE TABLE workspace_member (
  user_id TEXT PRIMARY KEY,
  role TEXT NOT NULL CHECK (role IN ('owner','admin','member','viewer')),
  joined_at INTEGER NOT NULL
);

CREATE TABLE workspace_settings (
  key TEXT PRIMARY KEY,
  ciphertext BLOB,        -- AES-GCM(nonce || ciphertext || tag) when encrypted
  -- plaintext JSON is allowed when encryption is disabled
);

CREATE TABLE workspace_keycheck (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  kdf_n INTEGER NOT NULL,
  kdf_r INTEGER NOT NULL,
  kdf_p INTEGER NOT NULL,
  salt BLOB NOT NULL,
  verifier_nonce BLOB NOT NULL,
  verifier_ciphertext BLOB NOT NULL
);

CREATE TABLE document_principal (
  document_id TEXT NOT NULL,
  principal_type TEXT NOT NULL CHECK (principal_type IN ('user','role','group')),
  principal_id TEXT NOT NULL,
  permission TEXT NOT NULL CHECK (permission IN ('read','admin')),
  granted_by TEXT NOT NULL,
  granted_at INTEGER NOT NULL,
  PRIMARY KEY (document_id, principal_type, principal_id, permission)
);
```

Other tables in the same file: `documents`, `chunks`, `vec_chunks`,
`fts_chunks`, `role`, `role_member`, `workspace_group`,
`workspace_group_member`, `sessions`, `turns`, `memory_fact`,
`ingestion_jobs`, `graph_entities`, `graph_edges`, `trace_corpus`,
`audit_event`, and `users`.

Every `Sqlite*Store` takes its `Database` handle from the
`WorkspaceHandle`. There is no second SQLite file in the runtime —
the in-process daemon opens exactly one database per workspace.

## Encryption parameters

| parameter | value |
| --- | --- |
| KDF | scrypt |
| N | 2¹⁵ = 32768 |
| r | 8 |
| p | 1 |
| salt | 16 bytes, random per workspace |
| key length | 32 bytes |
| cipher | AES-256-GCM |
| nonce | 12 bytes, random per write |
| tag | 16 bytes (default for GCM) |

The verifier ciphertext is `AES-GCM("raghub", key)`. Possessing the
key lets you decrypt the verifier; failing that produces an
authentication tag mismatch and a `ConfigurationError`.

## Locking policy

- **Server-side unlock only.** The browser never sees the
  passphrase-derived key and never receives raw `workspace_settings`
  values. The Next.js API forwards the passphrase on every request
  through a server-only handler.
- **JWT-only after login.** Once `/v1/auth/login` accepts the
  passphrase, the server returns a JWT and forgets the passphrase.
  Subsequent requests use the bearer token alone; the passphrase is
  re-required only on a fresh login.
- **Passphrase loss is unrecoverable.** Re-onboarding requires a
  fresh workspace.