# Configuration Reference

`AppSettings` is the runtime configuration snapshot consumed by both
the [`raghub.RAG`](../..//raghub/api/rag.py) facade and the legacy
`DynamicRagApplication`. It is a Python `@dataclass(slots=True)`;
each field has a corresponding `RAG_*` environment variable and a
matching key in `config/<profile>.yaml` / `.toml`.

## Configuration profiles

Profiles are loaded from `config/`:

| Profile | File (default) | Purpose |
|---|---|---|
| `development` | `config/development.yaml` | Local dev (default) |
| `staging`     | `config/staging.yaml`     | Pre-production |
| `production`  | `config/production.yaml`  | Production deployment |

Select via `RAG_PROFILE` environment variable or pass `profile=` to
`load_settings()`. YAML is loaded first; if a matching `.toml`
profile exists it is merged on top.

## Settings keys

The full list of `AppSettings` fields:

| Field | Type | Default | Env Variable | Description |
|---|---|---|---|---|
| `environment` | `str` | `"development"` | `RAG_ENV` | Profile name |
| `data_dir` | `Path` | `./data` | `RAG_DATA_DIR` | Root for derived state |
| `registry_path` | `Path` | `./data/registry.db` | `RAG_REGISTRY_PATH` | SQLite document registry |
| `sessions_path` | `Path` | `./data/sessions.db` | `RAG_SESSIONS_PATH` | SQLite session store |
| `zvec_dir` | `Path` | `./data/zvec` | `RAG_ZVEC_DIR` | Legacy ZVec vector store |
| `chunk_size_words` | `int` | `800` | `RAG_CHUNK_SIZE_WORDS` | Default chunk size |
| `chunk_overlap_words` | `int` | `100` | `RAG_CHUNK_OVERLAP_WORDS` | Default overlap |
| `top_k` | `int` | `5` | `RAG_TOP_K` | Default retrieval top-k |
| `embedding_dim` | `int` | `384` | `RAG_EMBEDDING_DIM` | Embedding dimensionality |
| `session_timeout_seconds` | `int` | `3600` | `RAG_SESSION_TIMEOUT_SECONDS` | Session TTL |
| `max_upload_bytes` | `int` | `20971520` (20 MiB) | `RAG_MAX_UPLOAD_BYTES` | Max upload size |
| `embedding_model` | `str` | `"hashing-bge"` | `RAG_EMBEDDING_MODEL` | Embedding model id |
| `llm_model` | `str` | `"heuristic-llm"` | `RAG_LLM_MODEL` | LLM model id (LiteLLM namespace) |
| `retrieval_mode` | `str` | `"sync"` | `RAG_RETRIEVAL_MODE` | `sync` or `background` |
| `log_level` | `str` | `"INFO"` | `RAG_LOG_LEVEL` | Minimum log level |
| `worker_backend` | `str` | `"threadpool"` | `RAG_WORKER_BACKEND` | `threadpool` or `asyncio` |
| `require_zvec` | `bool` | `False` | `RAG_REQUIRE_ZVEC` | Fail startup if ZVec is unavailable (legacy surface only) |
| `jwt_secret` | `str` | `""` | `JWT_SECRET` | HS256 signing secret |
| `nvidia_api_key` | `str` | `""` | `NVIDIA_API_KEY` | NVIDIA API key |
| `allow_passwordless_login` | `bool` | `True` (dev) / `False` (prod) | `RAG_ALLOW_PASSWORDLESS` | Dev-only convenience for passwordless sessions |
| `profile_path` | `Path \| None` | `None` | — | Profile file that was loaded |
| `extra` | `dict` | `{}` | — | Free-form forward-compatible config |

### Configuration precedence

Highest wins:

1. `RAG(settings=AppSettings.override(chunk_size_words=400))` or
   any constructor argument.
2. Environment variables (see above).
3. TOML profile (`config/<profile>.toml`) — overrides YAML.
4. YAML profile (`config/<profile>.yaml`).
5. Built-in defaults.

### `AppSettings.override(**changes)`

Returns a new `AppSettings` with the given fields changed
(immutable `dataclasses.replace` semantics). The receiver is not
mutated:

```python
from raghub.config.settings import load_settings

settings = load_settings(profile="development")
small_settings = settings.override(chunk_size_words=400, top_k=8)
```

## Environment variables

Beyond `RAG_*`, the facade reacts to these direct variables:

| Variable | Effect |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA provider credentials (consumed by `default_llm` / `default_embedder`) |
| `OPENAI_API_KEY` | OpenAI provider credentials |
| `ANTHROPIC_API_KEY` | Anthropic provider credentials |
| `GROQ_API_KEY` | Groq provider credentials |
| `LITELLM_API_KEY` | LiteLLM provider credentials (generic) |
| `QDRANT_URL` | Switches the default vector store to Qdrant |
| `LANGFUSE_PUBLIC_KEY` | Enables Langfuse telemetry |
| `LANGFUSE_SECRET_KEY` | Enables Langfuse telemetry |
| `LANGFUSE_HOST` | Optional Langfuse self-hosted endpoint |
| `CORS_ORIGINS` | FastAPI surface only; comma-separated allow-list |
| `RAGHUB_USERS` | Streamlit UI demo-user JSON override |
| `RAGHUB_FINANCEBENCH_CACHE` | Local FinanceBench cache directory |

## Production invariants (`environment == "production"`)

`load_settings` enforces three rules when
`settings.environment == "production"`:

1. `JWT_SECRET` must be set.
2. `JWT_SECRET` must be ≥ 32 bytes when UTF-8 encoded (PyJWT's
   `InsecureKeyLengthWarning` is fatal in CI).
3. `allow_passwordless_login` must be `false`.

Violating any of these raises `RuntimeError` at startup. The
production profile in `config/production.yaml` already sets
`allow_passwordless_login: false` and `require_zvec: true`.

## Defaults that are *dev-only stubs*

The `embedding_model: hashing-bge` and `llm_model: heuristic-llm`
defaults are offline, deterministic, and **not** recommended for
production. Replace them in your config:

```yaml
embedding_model: nvidia/embed-qa-4
llm_model: nvidia/llama-3.3-nemotron-super-49b-v1.5
```

…paired with `NVIDIA_API_KEY=$NVIDIA_API_KEY` in the environment.

## Sample config

Generated by `raghub init -o raghub.yaml`:

```yaml
environment: development
data_dir: ./data
chunk_size_words: 800
chunk_overlap_words: 100
embedding_dim: 384
embedding_model: hashing-bge
llm_model: heuristic-llm
retrieval_mode: sync
log_level: INFO
worker_backend: threadpool
jwt_secret: change-me
nvidia_api_key: ""
allow_passwordless_login: true
```

## Notes

- `RAG.from_config(path)` accepts either a `.yaml` or `.toml`
  config file. TOML takes precedence over YAML when both are
  present and both paths match.
- `AppSettings.ensure_dirs()` is called by `load_settings` and by
  `RAG.from_config`, so the directories are created lazily before
  the first I/O.
- Free-form keys (anything not in the dataclass field list) are
  preserved on `settings.extra` for forward compatibility.
