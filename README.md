# raghub

Production-grade multi-user RAG platform with ZVec vector store, multi-format document support, NVIDIA LLM provider, SQLite persistence, JWT auth, RBAC, and observability.

## Quick Start

```bash
./setup.sh
source .venv/bin/activate
uvicorn raghub.api.app:app --reload
```

```bash
# Login (creates a session token)
python -m raghub login admin@example.com secret
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "secret"}'

# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@report.pdf" \
  -F "company=Acme"
```

## Architecture

```
raghub/
  api/              FastAPI server, CLI, Streamlit UI
  auth/             JWT authentication + RBAC authorization
  conversation/     Session-scoped conversation manager
  core/             DI container, application factory
  documents/        Parsers, chunkers, lifecycle, validation
  embeddings/       NVIDIA / SentenceTransformers providers
  ingestion/        Document ingestion pipeline
  llm/              NVIDIA LLM provider
  observability/    Prometheus, OpenTelemetry, structlog
  retrieval/        Vector + hybrid retrieval pipeline
  services/         Application service facade (ServiceMixin)
  storage/          SQLite persistence, image store
  vectorstore/      ZVec / InMemory vector store adapters
  config/           YAML configuration settings
```

## Key Decisions

| Decision | Choice |
|----------|--------|
| Vector store | ZVec (prod), InMemoryVectorStore (dev) |
| Embeddings | NVIDIA NV-Embed-QA, SentenceTransformers |
| LLM | NVIDIA Llama 3.3 Nemotron Super 49B |
| Auth | JWT (PyJWT) + bcrypt password hashing |
| Persistence | SQLite via aiosqlite + DatabaseManager |
| Domain pattern | Active Record (build, update, remove) |
| Package structure | Flattened (no src/) |

## Documentation

- [Getting Started](docs/guide/getting-started.md)
- [Development Guide](docs/guide/development.md)
- [Deployment Guide](docs/guide/deployment.md)
- [API Reference](docs/reference/api.md)
- [Configuration Reference](docs/reference/configuration.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Design Decisions](docs/architecture/decisions.md)
- [Monitoring](docs/operations/monitoring.md)
- [Future Extensions](docs/future.md)

## License

MIT
