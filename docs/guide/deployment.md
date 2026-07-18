# Deployment

RAGHub ships with two Compose files and a single multi-stage
`Dockerfile`. The production target is
`docker-compose.yml`; the explicit development override is
`docker-compose.dev.yml`.

## Quick reference

```bash
# 1. Configure secrets (REQUIRED before any production run).
cp .env.example .env
$EDITOR .env                    # set JWT_SECRET, LLM key, etc.
openssl rand -base64 48          # generate JWT_SECRET

# 2. Build the images.
docker compose -f docker-compose.yml build

# 3. Start the production stack (api + ui + qdrant).
docker compose -f docker-compose.yml --profile production up -d

# 4. Verify health.
docker compose -f docker-compose.yml --profile production ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:6333/healthz
```

For local development, use the explicit dev override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml \
    --profile dev up
```

The Makefile wraps both as `compose-up` / `compose-dev`. The dev
override mounts the source tree and runs uvicorn with `--reload`;
the base compose file does neither.

## Why two compose files

`docker-compose.override.yml` is auto-merged by Compose, which
makes it invisible to anyone reading the production manifest. We
renamed the override to `docker-compose.dev.yml` so:

* `docker compose -f docker-compose.yml up` always starts the
  hardened production stack.
* The dev overrides (source mounts, `--reload`, relaxed security)
  are an explicit, opt-in decision.
* The README, runbook, and deployment docs all show the same
  command.

## Container images

The `Dockerfile` is a multi-stage build:

* **builder** — compiles the wheel and emits a
  `requirements-runtime.txt` from the project metadata.
* **runtime** — installs the wheel plus only the runtime
  dependencies, drops privileges to the unprivileged `raghub`
  user, and runs as PID 1 under tini (`init: true`).

A `SERVICE=api|ui` build arg selects the in-image healthcheck
(`/health` for the API, `/_stcore/health` for the Streamlit UI)
and the container `CMD`. Production compose passes `SERVICE=api`
and `SERVICE=ui` respectively. Both images are built from the
same Dockerfile; there is no separate `Dockerfile.api` /
`Dockerfile.ui`.

The image is `python:3.12-slim-bookworm` with a pinned patch tag
(no digest). The `PIP_NO_CACHE_DIR=1` and
`PIP_DISABLE_PIP_VERSION_CHECK=1` env vars are set globally.

## Persistence

Three named volumes are managed by Compose:

| Volume | Service | Backing |
|---|---|---|
| `raghub_data`           | api, ui | SQLite registry, sessions, image cache |
| `raghub_qdrant_data`    | qdrant  | Vector index |
| `raghub_qdrant_snapshots` | qdrant | Local snapshot history |

`docker compose down` keeps the volumes; `down -v` removes them.
See [`operations/backup.md`](../operations/backup.md) for the
backup / restore procedure.

## Hardening (production defaults)

* `read_only: true` on the API and UI containers.
* `tmpfs` mounts for `/tmp` and `/run`.
* `cap_drop: [ALL]` plus a minimal `cap_add` allow-list.
* `security_opt: [no-new-privileges:true]`.
* `init: true` (tini as PID 1).
* `restart: unless-stopped`.
* JSON-file log driver with `max-size: 10m`, `max-file: 5`.
* `deploy.resources` limits on every service.
* `env_file` is `required: true` — the stack fails closed if
  `.env` is missing.

## Health and readiness

* `GET /health` (API) — liveness probe, no auth.
* `GET /v1/health` (API) — service-level health summary.
* `GET /_stcore/health` (UI) — Streamlit's internal probe.
* `GET /healthz` (Qdrant) — Qdrant's own probe.

Every service has a `healthcheck` block in compose, and dependents
use `condition: service_healthy` to gate startup. The UI waits on
the API, the API waits on Qdrant.

## Configuration profiles

`RAGHub` reads `config/<profile>.yaml` (and the optional matching
`.toml`) where `profile` comes from `RAG_PROFILE`. The shipped
profiles:

| Profile | File | Purpose |
|---|---|---|
| `development` | `config/development.yaml` | Local dev (offline defaults) |
| `staging`     | `config/staging.yaml`     | Pre-production |
| `production`  | `config/production.yaml`  | Production (fail-closed) |

The compose file pins `RAG_PROFILE=production` in the API and UI
service environment. Override per deployment with `RAG_PROFILE=…`
in `.env`.

## Key environment variables

| Variable | Description |
|---|---|
| `RAG_PROFILE` | Configuration profile name |
| `RAG_DATA_DIR` | Root for registry, sessions, manifest, ingestion ledger |
| `RAG_REGISTRY_PATH` | Document registry path |
| `RAG_SESSIONS_PATH` | Session store path |
| `RAG_ZVEC_DIR` | Legacy ZVec vector store directory |
| `RAG_CHUNK_SIZE_WORDS` | Override chunk size |
| `RAG_CHUNK_OVERLAP_WORDS` | Override chunk overlap |
| `RAG_TOP_K` | Default retrieval top-k |
| `RAG_EMBEDDING_DIM` | Embedding dimensionality |
| `RAG_EMBEDDING_MODEL` | Embedding model id |
| `RAG_LLM_MODEL` | LLM model id |
| `RAG_LOG_LEVEL` | Log level |
| `JWT_SECRET` | Opaque session-token signing secret (≥ 32 bytes in production). 0.4.0 no longer issues JWTs; this secret signs the UUID session tokens minted by `SqliteSessionStore`. |
| `NVIDIA_API_KEY` | NVIDIA provider credential |
| `OPENAI_API_KEY` | OpenAI credential |
| `ANTHROPIC_API_KEY` | Anthropic credential |
| `GROQ_API_KEY` | Groq credential |
| `LITELLM_API_KEY` | Generic LiteLLM credential |
| `QDRANT_URL` | Qdrant endpoint (set automatically by compose) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse self-hosted endpoint |
| `CORS_ORIGINS` | FastAPI CORS allow-list (comma-separated). Must be a non-wildcard list — the server refuses to start with `*` because browsers reject wildcard+credentials. |
| `RAGHUB_USERS` | Streamlit UI demo-user JSON override |
| `RAGHUB_API_URL` | URL the UI uses to call the API (compose sets this) |

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
  must set `RAGHUB_USERS` or bootstrap accounts before the first
  start.
* `RAG_ALLOW_PASSWORDLESS=false` is set in `.env`.
* `CORS_ORIGINS` is the real frontend origin, not the wildcard.
* The Qdrant volume (`raghub_qdrant_data`) is on durable storage.
* `docker compose -f docker-compose.yml --profile production ps`
  reports every service as `healthy`.
* The `raghub_data` volume is included in the daily backup (see
  [`operations/backup.md`](../operations/backup.md)).

## One canonical ingestion path

Documents enter the system through the FastAPI surface, not the
Streamlit UI. The two flows are:

| Path | Endpoint | Notes |
|---|---|---|
| Synchronous | `POST /v1/documents/upload` | Returns the document id and status |
| Batch        | `POST /v1/documents/ingest/batch` | One failure does not abort the others |
| Async        | `POST /v1/ingest/async` | Submits to the background pool, returns `{job_id}` |

The CLI equivalents (`raghub ingest <path>`) and the Streamlit
"Upload" widget both wrap the HTTP surface. Anything that mutates
the document registry must go through `DynamicRagApplication.
upload_document` (the same entry point the API exposes).

## Demo users (Streamlit UI)

The Streamlit UI pre-seeds five demo users:

| Email | Companies | Admin |
|---|---|---|
| `alice@acme.com` | Apple | No |
| `bob@acme.com` | Microsoft | No |
| `charlie@acme.com` | Amazon, Tesla | No |
| `diana@acme.com` | Google | No |
| `admin@acme.com` | (all) | Yes |

Default password: `password`. Override via the `RAGHUB_USERS`
environment variable (a JSON object of email →
`{password, companies, is_admin}`).

## Notes

* The `RAG` facade is designed for embedding in your own service.
  Wiring it in FastAPI is a thin shim around its sync and async
  methods; no auth or storage is added by the facade.
* The legacy FastAPI app at `raghub.api.app:get_app` (Uvicorn
  `--factory`) remains the canonical multi-tenant HTTP surface
  until a v2 is shipped.
* The Dockerfile builds a wheel and installs it with
  `--no-deps`; the runtime requirements are pinned in the
  builder-emitted `requirements-runtime.txt`.
