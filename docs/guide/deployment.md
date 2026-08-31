> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Deployment

Revex ships as a Python package on PyPI. The deployment model is
"install the wheel, configure the environment, run the process" —
no container images, no Compose stack, no orchestration manifests
are shipped from the Revex repository. Operators can run the
process directly, under `systemd`, inside a virtualenv on a bare
host, or as a workload in any application platform that can host
a Python entry point.

## Quick reference

```bash
# 1. Configure secrets (REQUIRED before any production run).
cp .env.example .env
$EDITOR .env                    # set JWT_SECRET, LLM key, etc.
openssl rand -base64 48         # generate JWT_SECRET

# 2. Install the package.
pip install "raghub[api,structured,langfuse,pdf]"

# 3. Initialise the data directory and SQLite stores.
raghub init -o revex.yaml

# 4. Run the FastAPI server.
raghub run --host 0.0.0.0 --port 8000
# ...or, equivalently:
python -c "from raghub.api import App; from raghub.config import Settings; \
import uvicorn; uvicorn.run(lambda: App.create(Settings.load()), factory=True, host='0.0.0.0', port=8000)"
```

Verify with:

```bash
curl -fsS http://127.0.0.1:8000/health
raghub health
```

The same process runs both the legacy `RagApplication` surface
(`/auth/login`, `/documents/upload`, `/query`, etc.) and the new
`RAG` facade behind the same app factory. Either is reachable from
the installed `raghub` package — there is no separate "UI" service
to start.

## Persistence

Three files live under `RAG_DATA_DIR` (default `./data`):

| File | Purpose |
|---|---|
| `registry.db` | Document registry, chunks, embeddings |
| `sessions.db` | Opaque session tokens (signed with `JWT_SECRET`) |
| `images/` | Optional content-addressed upload blob cache |

Backups are produced with `revex backup create -o <archive>` and
restored with `revex backup restore --input <archive> --target-dir
<dir>`. See [`operations/backup.md`](../operations/backup.md) for
the full procedure, retention policy, and cross-region guidance.

For PostgreSQL deployments, run `revex migrate pgvector --dsn
<dsn>` once to create the `PgVectorStore` schema and indexes, then
point the application at the database with `RAG_VECTORSTORE_DSN`.
The `pg_dump` / `pg_basebackup` job is the canonical PostgreSQL
backup; the SQLite files continue to be backed up through
`revex backup create`.

## Hardening (production defaults)

The application is hardened by configuration, not by container
manifests. The production profile (`config/production.yaml` plus
`RAG_PROFILE=production`) sets:

* Fail-closed `CORS_ORIGINS` — the server refuses to start with a
  wildcard origin and `allow_credentials=True`.
* Non-zero `JWT_SECRET` required — the process aborts at startup
  if the secret is the placeholder.
* Oversize uploads rejected at the edge with `413` before the body
  is buffered.
* Demo-user seeding suppressed automatically in production (and
  whenever `CORS_ORIGINS` is the default `*`).
* `RAG_ALLOW_PASSWORDLESS=false` by default; explicit opt-in is
  required to skip password hashing.

Process-level isolation (filesystem permissions, capability
restrictions, read-only root filesystem, `tmpfs` mounts, PID-1
process supervision) is the responsibility of the surrounding
platform — `systemd` unit hardening directives, Kubernetes
`securityContext`, or whatever the operator's environment
provides. Revex ships only a wheel, so the unit of isolation is
the install path and the surrounding process, not an image.

## Health and readiness

* `GET /health` — liveness probe, no auth.
* `GET /v1/health` — service-level health summary.
* `raghub health` — CLI equivalent.

The application starts up lazily and serves `/health` as soon as
FastAPI binds the socket, before the document registry is
populated. Use `/v1/health` (or `raghub health`) as the readiness
probe instead, because it reports the state of the configured
vector store, embedder, and LLM provider.

## Configuration profiles

`Revex` reads `config/<profile>.yaml` (and the optional matching
`.toml`) where `profile` comes from `RAG_PROFILE`. The shipped
profiles:

| Profile | File | Purpose |
|---|---|---|
| `development` | `config/development.yaml` | Local dev (offline defaults) |
| `staging`     | `config/staging.yaml`     | Pre-production |
| `production`  | `config/production.yaml`  | Production (fail-closed) |

Override per deployment with `RAG_PROFILE=…` in `.env` or the
process environment.

## Key environment variables

| Variable | Description |
|---|---|
| `RAG_PROFILE` | Configuration profile name |
| `RAG_DATA_DIR` | Root for registry, sessions, manifest, ingestion ledger |
| `RAG_REGISTRY_PATH` | Document registry path |
| `RAG_SESSIONS_PATH` | Session store path |
| `RAG_VECTORSTORE_PATH` | SQLite vector store path |
| `RAG_VECTORSTORE_DSN` | PostgreSQL DSN for the pgvector vector store (when set, overrides the SQLite path) |
| `RAG_REGISTRY_DSN` | Optional Postgres DSN for the registry store |
| `RAG_SESSIONS_DSN` | Optional Postgres DSN for the session store |
| `RAG_CHUNK_SIZE_WORDS` | Override chunk size |
| `RAG_CHUNK_OVERLAP_WORDS` | Override chunk overlap |
| `RAG_TOP_K` | Default retrieval top-k |
| `RAG_EMBEDDING_DIM` | Embedding dimensionality |
| `RAG_EMBEDDING_MODEL` | Embedding model id |
| `RAG_LLM_MODEL` | LLM model id |
| `RAG_LOG_LEVEL` | Log level |
| `JWT_SECRET` | Opaque session-token signing secret (≥ 32 bytes in production). 0.4.0 no longer issues JWTs; this secret signs the UUID session tokens minted by `SqliteSessionStore`. |
| `RAG_LLM_API_KEY` | Preferred unified LLM credential (any provider) |
| `OPENAI_API_KEY` | OpenAI credential (fallback) |
| `ANTHROPIC_API_KEY` | Anthropic credential (fallback) |
| `GROQ_API_KEY` | Groq credential (fallback) |
| `LITELLM_API_KEY` | Generic LiteLLM credential (fallback) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse self-hosted endpoint |
| `CORS_ORIGINS` | FastAPI CORS allow-list (comma-separated). Must be a non-wildcard list — the server refuses to start with `*` because browsers reject wildcard+credentials. |

## Production checklist

* `JWT_SECRET` is a unique value of at least 32 bytes
  (`openssl rand -base64 48`). It signs the opaque session tokens
  minted by `SqliteSessionStore`; the legacy JWT path was deleted
  in 0.4.0.
* At least one LLM credential is exported (`NVIDIA_API_KEY`,
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `LITELLM_API_KEY`).
* `RAG_PROFILE=production` is set in `.env`.
* `CORS_ORIGINS` is an explicit comma-separated list (no `*`). The
  server fails fast at startup if a wildcard origin is configured
  alongside `allow_credentials=True`.
* Demo-user seeding is suppressed automatically in production
  (and whenever `CORS_ORIGINS` is the default `*`). Operators
  must set `REVEX_USERS` or bootstrap accounts before the first
  start.
* `RAG_ALLOW_PASSWORDLESS=false` is set in `.env`.
* `RAG_DATA_DIR` lives on durable storage with daily backups
  (see [`operations/backup.md`](../operations/backup.md)).
* `/v1/health` (or `raghub health`) reports `status: ok` and a
  populated `vector_store` entry before the first user request
  is routed.
* The process supervisor restarts on exit (`Restart=on-failure`
  under systemd, `restartPolicy: OnFailure` under Kubernetes).

## One canonical ingestion path

Documents enter the system through the FastAPI surface. The flows
are:

| Path | Endpoint | Notes |
|---|---|---|
| Synchronous | `POST /v1/documents/upload` | Returns the document id and status |
| Batch        | `POST /v1/documents/ingest/batch` | One failure does not abort the others |
| Async        | `POST /v1/ingest/async` | Submits to the background pool, returns `{job_id}` |

The CLI equivalents (`revex ingest <path>`) wrap the HTTP
surface. Anything that mutates the document registry must go
through `RagApplication.upload_document` (the same entry point the
API exposes).

## Operational references

The operations handbook covers the day-to-day concerns of running a
deployed instance:

- [`operations/runbook.md`](../operations/runbook.md) — first-line
  triage for failing services; covers health, logs, restarts, and
  the canonical reset path.
- [`operations/backup.md`](../operations/backup.md) — `raghub
  backup create` / `revex backup restore`, retention, and
  cross-region storage.
- [`operations/monitoring.md`](../operations/monitoring.md) —
  Prometheus metrics, Langfuse spans, and structured logging on a
  long-running process.
- [`operations/scaling.md`](../operations/scaling.md) —
  vertical / horizontal scaling, the API and vector store.
- [`operations/runbook.md`](../operations/runbook.md) — incident
  triage.

## Notes

* The `RAG` facade is designed for embedding in your own service.
  Wiring it in FastAPI is a thin shim around its sync and async
  methods; no auth or storage is added by the facade.
* The FastAPI app at `raghub.api.App.create(config)` (Uvicorn
  `--factory`) remains the canonical multi-tenant HTTP surface
  until a v2 is shipped.
* `python -m build` produces both an `sdist` and a `wheel`; the
  wheel is installable with `pip install <wheel>` and pulls its
  runtime dependencies from the metadata declared in
  `pyproject.toml`. There is no separate runtime requirements
  file — the wheel is self-describing.
