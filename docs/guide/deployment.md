# Deployment

## Local Development

```bash
pip install -e ".[api,ui,dev]"
uvicorn raghub.api.app:app --reload
streamlit run streamlit_app.py
```

## Docker

Build and run with the provided `Dockerfile` and `docker-compose.yml`:

```bash
docker-compose up --build
```

## Configuration Profiles

Profiles are loaded via the `RAG_PROFILE` environment variable:

- `config/development.yaml` (default)
- `config/staging.yaml`
- `config/production.yaml`

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | API key for NVIDIA LLM and embeddings |
| `JWT_SECRET` | JWT signing secret (required in production) |
| `RAG_PROFILE` | Configuration profile name |
| `RAG_DATA_DIR` | Data directory path |
| `RAG_EMBEDDING_MODEL` | Embedding model override |
| `RAG_LLM_MODEL` | LLM model override |

## Production Checklist

- Set `JWT_SECRET` to a strong, unique value
- Set `NVIDIA_API_KEY`
- Use `config/production.yaml` (passwordless login disabled by default)
- Configure `docker-compose.yml` with proper volume mounts for persistence
- Set up reverse proxy (nginx, Caddy) with TLS termination
