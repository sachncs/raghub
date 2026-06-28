# Configuration Reference

## Configuration Profiles

Profiles are YAML files in `config/`:

| Profile | File | Use Case |
|---------|------|----------|
| development | `config/development.yaml` | Local dev (default) |
| staging | `config/staging.yaml` | Pre-production |
| production | `config/production.yaml` | Production deployment |

Select profile via `RAG_PROFILE` environment variable or `--profile` CLI flag.

## Key Settings

| Key | Default | Description | Env Variable |
|-----|---------|-------------|--------------|
| `environment` | `development` | Runtime environment | `RAG_ENV` |
| `data_dir` | `./data` | Data directory | `RAG_DATA_DIR` |
| `registry_path` | `./data/registry.json` | Document registry path | `RAG_REGISTRY_PATH` |
| `sessions_path` | `./data/sessions.json` | Session store path | `RAG_SESSIONS_PATH` |
| `zvec_dir` | `./data/zvec` | ZVec vector store directory | `RAG_ZVEC_DIR` |
| `chunk_size_words` | `800` | Document chunk size | `RAG_CHUNK_SIZE_WORDS` |
| `chunk_overlap_words` | `100` | Chunk overlap | `RAG_CHUNK_OVERLAP_WORDS` |
| `top_k` | `5` | Top-k retrieval results | `RAG_TOP_K` |
| `embedding_dim` | `384` | Embedding dimension | `RAG_EMBEDDING_DIM` |
| `session_timeout_seconds` | `3600` | Session TTL | `RAG_SESSION_TIMEOUT_SECONDS` |
| `max_upload_bytes` | `20971520` | Max upload size (20MB) | `RAG_MAX_UPLOAD_BYTES` |
| `embedding_model` | `hashing-bge` | Embedding model name | `RAG_EMBEDDING_MODEL` |
| `llm_model` | `heuristic-llm` | LLM model name | `RAG_LLM_MODEL` |
| `retrieval_mode` | `sync` | Retrieval mode | `RAG_RETRIEVAL_MODE` |
| `log_level` | `INFO` | Logging level | `RAG_LOG_LEVEL` |
| `worker_backend` | `threadpool` | Async worker backend | `RAG_WORKER_BACKEND` |
| `require_zvec` | `false` | Require ZVec vector store | `RAG_REQUIRE_ZVEC` |
| `jwt_secret` | `""` | JWT signing secret | `JWT_SECRET` |
| `nvidia_api_key` | `""` | NVIDIA API key | `NVIDIA_API_KEY` |
| `allow_passwordless_login` | `true` | Allow login without password | `RAG_ALLOW_PASSWORDLESS` |

## Production Requirements

In production (`environment: production`) the following are enforced:

- `jwt_secret` must be set (via env var or config)
- `allow_passwordless_login` must be `false`

## Notes

- Environment variables override YAML values
- The `embedding_model` `hashing-bge` is a dev-only stub; use `all-MiniLM-L6-v2` or `nvidia/embed-qa-4` for real embeddings
- The `llm_model` `heuristic-llm` is a dev-only stub; use `nvidia/llama-3.3-nemotron-super-49b-v1.5` for real LLM
