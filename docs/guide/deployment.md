# Deployment

## Local development

```bash
pip install -e ".[api,ui,dev]"

# Start the legacy FastAPI server (auth-enabled, port 8000)
uvicorn raghub.api.app:app --reload

# Or start the Streamlit UI (RAG facade, port 8501)
streamlit run streamlit_app.py
```

The FastAPI server is bound to the legacy `DynamicRagApplication`
(see [`reference/api.md`](../reference/api.md)). The Streamlit UI
consumes the new `raghub.RAG` facade and pre-seeds demo users.

## Configuration profiles

`RAGHub` reads `config/<profile>.yaml` (and the optional matching
`.toml`) where `profile` comes from `RAG_PROFILE` (defaults to
`development`). Shipped profiles:

| Profile | File | Purpose |
|---|---|---|
| `development` | `config/development.yaml` | Local dev (offline defaults, `allow_passwordless_login: false`, `require_zvec: false`) |
| `staging`     | `config/staging.yaml`     | Pre-production (NVIDIA models, passwordless off) |
| `production`  | `config/production.yaml`  | Production (NVIDIA models, passwordless **must** be false, `require_zvec: true`) |

Override via `RAG_PROFILE` env var or pass `RAG.from_config(path)`.

## Container images

A `Dockerfile` and a `docker-compose.yml` ship at the repo root.

```bash
docker compose up --build
```

The compose file defines two services:

- `api` — `uvicorn raghub.api.app:app --host 0.0.0.0 --port 8000`
- `ui`  — `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501`

Each service mounts `./data` and `./config` so the registry,
sessions, ingestion ledger, and configuration profile persist
across container restarts.

The default image healthcheck is `python -m raghub.cli health`;
`HEALTHCHECK` is configured for both services.

## Key environment variables

| Variable | Description |
|---|---|
| `RAG_PROFILE` | Configuration profile name |
| `RAG_ENV` | Sets `settings.environment` (default = `RAG_PROFILE`) |
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
| `JWT_SECRET` | HS256 signing secret (≥ 32 bytes in production) |
| `NVIDIA_API_KEY` | NVIDIA provider credential |
| `OPENAI_API_KEY` | OpenAI credential |
| `ANTHROPIC_API_KEY` | Anthropic credential |
| `GROQ_API_KEY` | Groq credential |
| `LITELLM_API_KEY` | Generic LiteLLM credential |
| `QDRANT_URL` | Qdrant endpoint (enable Qdrant vector store) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse self-hosted endpoint |
| `CORS_ORIGINS` | FastAPI CORS allow-list (comma-separated) |
| `RAGHUB_USERS` | Streamlit UI demo-user JSON override |
| `RAGHUB_FINANCEBENCH_CACHE` | Local FinanceBench cache directory |

## Production checklist

- Set `JWT_SECRET` to a strong, unique value of **at least 32
  bytes** (PyJWT rejects shorter keys for HS256; the production
  profile enforces this at startup).
- Set `NVIDIA_API_KEY` (or the equivalent `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY`) so the `RAG` facade wires the real LLM
  instead of `HeuristicLLMProvider`.
- Set `QDRANT_URL` so the Qdrant vector store is enabled.
- Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for
  telemetry; otherwise the facade runs with `NoOpTelemetry`.
- Confirm `config/production.yaml`'s `allow_passwordless_login:
  false` is in effect (the loader raises `RuntimeError`
  otherwise).
- Persist `./data` to a durable volume; the SQLite ingestion
  ledger, manifest, registry, and sessions live there.
- Front the API with a reverse proxy (nginx, Caddy, Traefik,
  etc.) for TLS termination.
- Set `CORS_ORIGINS` to the actual frontend origin rather than
  the wildcard default.
- Pin the spec libraries (`marker-pdf`, `chonkie`, `litellm`,
  `instructor`, `qdrant-client`, `langfuse`) — they move fast.

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

## Health and readiness

- `raghub health` (CLI) calls `RAG.health()` and prints a JSON
  summary of every collaborator.
- `GET /health` (FastAPI) is the liveness probe; no auth.

## Notes

- The `RAG` facade is designed for embedding in your own service.
  Wiring it in FastAPI is a thin shim around its sync and async
  methods; no auth or storage is added by the facade.
- The legacy FastAPI app at `raghub.api.app:app` remains the
  canonical multi-tenant HTTP surface until a v2 is shipped.
