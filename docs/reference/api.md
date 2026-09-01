# API reference

`@revex/api` is a Hono HTTP server. `boot()` returns a `BootResult` with the
app, registry, pool, embedder, jwt, hasher, bound stores, and file storage.
`start()` boots and listens on `REVEX_API_PORT` (default `3000`).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `REVEX_WORKSPACE_HOME` | `${HOME}/.revex` | Root for workspace data. |
| `REVEX_WORKSPACE_DIR` | — | Direct parent of `workspace.db` files (overrides home-derived path). |
| `REVEX_API_PORT` | `3000` | HTTP listen port. |
| `REVEX_VERSION` | `0.1.0` | Version surfaced in `/health`. |
| `REVEX_JWT_SECRET` | dev fallback | HS256 signing secret; **required** when `NODE_ENV=production`. |
| `REVEX_EMBEDDER_API_KEY` | — | Use `OpenAIEmbedder`; else `FeatureHashingEmbedder`. |
| `OPENAI_API_KEY` | — | Fallback for `REVEX_EMBEDDER_API_KEY`. |
| `REVEX_EMBEDDER_MODEL` | `text-embedding-3-large` | Embedding model. |
| `REVEX_VECTOR_EMBEDDING_DIM` | `3072` | Vector dimension. |
| `REVEX_PASSPHRASE_VAULT` | `memory` | `memory` (dev) or `kms` (prod). |
| `REVEX_WORKER_ROLE` | `leader` | `leader` / `follower` / `disabled`. |
| `REVEX_RESET_STUCK_JOBS` | — | `1` re-queues stale `running` jobs on boot. |
| `REVEX_QUOTAS_DOC` | — | Advisory document soft limit. |
| `REVEX_QUOTAS_CHUNK` | — | Advisory chunk soft limit. |
| `REVEX_LLM_PROVIDER` / `_MODEL` / `_API_KEY` / `_BASE_URL` / `_TEMPERATURE` | — | LLM config (used by core `Settings`). |
| `REVEX_API_BASE` | `http://localhost:3000` | CLI default base URL. |

## Middleware

Applied in order by `createApp`.

### `securityHeadersMiddleware`

- CORS to `allowOrigins` (default `http://localhost:3001`,
  `http://127.0.0.1:3001`).
- Always sends `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `X-Frame-Options: DENY`.
- `OPTIONS` → 204.

### `errorMiddleware`

Maps `RevexError.code` → HTTP status (see [error table](#error-codes)). All
errors render:

```json
{ "error": { "code": "...", "message": "...", "details": { ... } } }
```

### `rateLimitMiddleware`

- Per-IP: 60 req/min. Per-workspace: 600 req/min (from JWT `workspace_id`).
- Exceeded → `429` with `Retry-After`.
- Bypassed paths: `/health`, `/readyz`.

### `jwtAuthMiddleware`

Verifies `Authorization: Bearer <jwt>`, reads `revex_workspace_key` cookie.
Sets `c.var.claims` (`JwtClaims`) and `c.var.passphrase`. Helpers `getClaims(c)`
and `getPassphrase(c)`.

## Routes

> JWT = requires `Authorization: Bearer`. PP = requires `revex_workspace_key`
> passphrase cookie. Role = additional admin/owner/member gate.

### Open endpoints

#### `GET /health`

```json
{ "ok": true }
```

#### `GET /readyz`

```json
{ "ok": true, "workspaces": 1, "poolSize": 1 }
```

### Auth

#### `POST /v1/auth/register`

Body: `email`, `password` (≥8), `workspaceName`, `passphrase` (≥8),
`llm: { provider, model, apiKey?, baseUrl? }`. Creates workspace + admin user,
returns `{ token, user, workspace }`. Errors: 400 missing / password too short /
passphrase too short.

#### `POST /v1/auth/login`

Body: `email`, `password`, `passphrase`. Returns `{ token, user }`. Errors:
400 missing, 401 invalid credentials / invalid passphrase, 404 workspace not
found.

#### `POST /v1/auth/logout` — JWT

Clears `revex_session` + `revex_workspace_key` cookies, removes the session
record. Returns `{ ok: true }`.

#### `POST /v1/auth/password` — JWT, PP

Body: `currentPassword`, `newPassword` (≥8). Returns `{ ok: true }`.

#### `POST /v1/auth/reset` — JWT, PP, admin/owner

Body: `userId`, `newPassword` (≥8). Returns `{ ok: true }`.

### Me

#### `GET /v1/me` — JWT

```json
{ "user": { "id": "...", "workspaceId": "...", "email": "...",
            "isAdmin": true, "role": "admin", "displayName": "..." } }
```

#### `PATCH /v1/me/strategy` — JWT

Body: arbitrary strategy override object. Returns `{ ok, strategy }`.

#### `GET /v1/me/history` — JWT

Returns `{ turns: [] }` (placeholder).

### Query

#### `POST /v1/query` — JWT

Body: `question` (required), `session_id?`, `mode?` (`graph | swarm |
workflow | deep_research`).

```json
{ "answer": "...", "citations": [], "hits": [ { "id","score","text" } ], "mode": "graph" }
```

#### `POST /v1/query/stream` — JWT

Same body. SSE stream of `PlannerEvent`s:
`id: step`, `event: kind`, `data: JSON(payload)`.

#### `POST /v1/agent/run` — JWT

Body: `question`, `sessionId?`, `strategy?`. Returns `{ answer, citations, mode }`.

### Settings

#### `GET /v1/settings/llm` — JWT, PP

```json
{ "llm": { "provider": "...", "model": "...", "apiKey": "••••••••", "baseUrl": "..." } }
```

`apiKey` always redacted.

#### `PUT /v1/settings/llm` — JWT, PP

Body: `provider` (`openai | minimax | litellm | anthropic | bedrock`),
`model`, `apiKey?`, `baseUrl?`, `temperature?`. Returns `{ ok: true }`.

### Documents

#### `GET /v1/documents` — JWT, PP

```json
{ "documents": [ { "id","workspaceId","ownerId","filename","mimeType",
                   "hash","byteSize","status","metadata","createdAt" } ] }
```

#### `POST /v1/documents` — JWT, PP (multipart)

Field `file` (File), optional `collection_id`, other fields become metadata.
Returns `202 { documentId, hash, status, byteSize, alreadyExisted }` (200 if
duplicate hash).

#### `POST /v1/documents/ingest-stream` — JWT, PP (multipart)

Same payload; SSE-streams `IngestEvent` phases (`done`, `failed`, ...).

#### `DELETE /v1/documents/:id` — JWT, PP, owner

Returns `{ ok: true }`. Errors: 404 not found, 403 not the owner.

### Document ACL

All JWT + PP, owner or admin.

#### `GET /v1/documents/:id/principals`

```json
{ "principals": [ { "documentId","principalType","principalId","permission",
                    "grantedBy","grantedAt" } ] }
```

#### `POST /v1/documents/:id/principals`

Body: `principalType` (`user | role | group`), `principalId`, `permission`
(`read | admin`).

#### `DELETE /v1/documents/:id/principals`

Body: `principalType`, `principalId`, `permission`.

### Workspaces

#### `GET /v1/workspaces/members` — JWT, PP

```json
{ "members": [ { "userId": "...", "role": "...", "joinedAt": "..." } ] }
```

#### `POST /v1/workspaces/members` — JWT, PP, admin/owner

Body: `email`, `role` (`owner | admin | member | viewer`), `displayName?`.
Returns `201 { member }`.

#### `PATCH /v1/workspaces/members/:userId` — JWT, PP, admin/owner

Body: `role`. Returns `{ member }`.

#### `DELETE /v1/workspaces/members/:userId` — JWT, PP, admin/owner

Returns `{ ok: true }`.

### Feedback

#### `POST /v1/feedback` — JWT, PP

Body: `turnId`, `rating` (`up | down | neutral`), `comment?`. Returns
`{ id, status: "recorded" }`.

#### `GET /v1/feedback` — JWT, PP, member

Returns `{ feedback: [] }`.

#### `GET /v1/feedback/aggregate` — JWT, PP, member

Returns `{ up, down, neutral }`.

#### `GET /v1/feedback/:id` — JWT, PP, member

Single feedback record.

#### `DELETE /v1/feedback/:id` — JWT, PP, member

Returns `{ deleted: boolean }`.

### Audit & stats

#### `GET /v1/audit` — JWT, PP, admin/owner

Query: `limit` (clamped 1–500, default 100), `kind`. Returns `{ events: [] }`.

#### `GET /v1/audit/kinds` — JWT, PP, admin/owner

Returns `{ kinds: [] }`.

#### `GET /v1/stats` — JWT, PP

```json
{ "documents": { "total": 0, "byStatus": {} }, "audit": { "events": 0 },
  "feedback": {} }
```

#### `GET /v1/admin/stats` — JWT, PP

Memory stats: `documentCount`, `chunkCount`, `totalTokens`, `embeddingBytes`,
`bytesOnDisk`, `lastIngestedAt`, `statusCounts`, `sources`, `capacity`.

#### `POST /v1/admin/vacuum` — JWT, PP

```json
{ "workspaceId": "...", "integrityBefore": "...", "integrityAfter": "...",
  "vacuumedAt": 0 }
```

### Tenants

#### `GET /v1/tenants` — JWT, PP

Returns `{ tenants: [] }` (registry entries).

#### `GET /v1/tenants/:id` — JWT, PP

Returns `{ tenant }`. Errors: 404 not found.

### Webhooks

All JWT + PP, admin/owner. (Delivery is not yet implemented; the surface is
placeholder.)

#### `POST /v1/webhooks`

Body: `url`, `events[]`, `secret?`. Returns `201 { id, url, events, secret, createdAt }`.

#### `GET /v1/webhooks`

Returns `{ webhooks: [] }`.

#### `DELETE /v1/webhooks/:id`

Returns `{ deleted: true }`.

### Operational

#### `GET /v1/diagnostics` — JWT

```json
{ "ok": true, "workspace": { "id","encryption","registeredAt" } | null,
  "poolSize": 0, "uptimeSec": 0 }
```

## Error codes {#error-codes}

| Code | Status |
|---|---|
| `auth_error` | 401 |
| `authorization_error` | 403 |
| `configuration_error` | 500 (503 for `StoreUnavailableError`) |
| `generation_error` | 502 |
| `ingestion_error` | 400 |
| `missing_dependency` | 500 |
| `pipeline_error` | 500 |
| `retrieval_error` | 502 |
| `vector_store_error` | 500 |
| `verification_error` | 400 |
| `revex_error` | 500 |
| `rate_limit` | 429 |

## Route → auth matrix

| Method | Path | JWT | PP | Role |
|---|---|---|---|---|
| GET | `/health` | | | |
| GET | `/readyz` | | | |
| POST | `/v1/auth/register` | | | |
| POST | `/v1/auth/login` | | | |
| POST | `/v1/auth/logout` | ✓ | | |
| POST | `/v1/auth/password` | ✓ | ✓ | |
| POST | `/v1/auth/reset` | ✓ | ✓ | admin/owner |
| GET | `/v1/me` | ✓ | | |
| PATCH | `/v1/me/strategy` | ✓ | | |
| GET | `/v1/me/history` | ✓ | | |
| POST | `/v1/query` | ✓ | | |
| POST | `/v1/query/stream` | ✓ | | |
| POST | `/v1/agent/run` | ✓ | | |
| GET | `/v1/settings/llm` | ✓ | ✓ | |
| PUT | `/v1/settings/llm` | ✓ | ✓ | |
| GET | `/v1/documents` | ✓ | ✓ | |
| POST | `/v1/documents` | ✓ | ✓ | |
| POST | `/v1/documents/ingest-stream` | ✓ | ✓ | |
| DELETE | `/v1/documents/:id` | ✓ | ✓ | owner |
| GET/POST/DELETE | `/v1/documents/:id/principals` | ✓ | ✓ | owner/admin |
| GET | `/v1/workspaces/members` | ✓ | ✓ | |
| POST | `/v1/workspaces/members` | ✓ | ✓ | admin/owner |
| PATCH/DELETE | `/v1/workspaces/members/:userId` | ✓ | ✓ | admin/owner |
| POST | `/v1/feedback` | ✓ | ✓ | |
| GET | `/v1/feedback` | ✓ | ✓ | member |
| GET | `/v1/feedback/aggregate` | ✓ | ✓ | member |
| GET/DELETE | `/v1/feedback/:id` | ✓ | ✓ | member |
| GET | `/v1/audit` | ✓ | ✓ | admin/owner |
| GET | `/v1/audit/kinds` | ✓ | ✓ | admin/owner |
| GET | `/v1/stats` | ✓ | ✓ | |
| GET | `/v1/admin/stats` | ✓ | ✓ | |
| POST | `/v1/admin/vacuum` | ✓ | ✓ | |
| GET | `/v1/tenants` | ✓ | ✓ | |
| GET | `/v1/tenants/:id` | ✓ | ✓ | |
| POST/GET/DELETE | `/v1/webhooks` | ✓ | ✓ | admin/owner |
| GET | `/v1/diagnostics` | ✓ | | |