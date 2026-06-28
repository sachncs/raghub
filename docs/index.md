# Dynamic RAG Framework

Multi-user retrieval augmented generation platform with runtime PDF ingestion, metadata-aware retrieval, session-scoped conversational memory, and pluggable adapters.

```text
raghub/
  api/            FastAPI server, CLI, Streamlit UI
  auth/           JWT authentication, RBAC authorization
  config/         AppSettings, YAML configuration loading
  conversation/   Session-scoped conversation manager
  core/           DI container, application factory
  documents/      Parsers, chunkers, validation
  domain/         Active Record models (Document, Chunk, Session)
  embeddings/     NVIDIA NV-Embed-QA, SentenceTransformers
  exceptions/     Domain exception classes
  ingestion/      Document ingestion pipeline
  interfaces/     Abstract base classes (providers, storage)
  llm/            NVIDIA LLM provider
  models/         Pydantic DTOs (API, domain)
  observability/  Prometheus, OpenTelemetry, structlog
  prompts/        Prompt template builder
  repositories/   SQLite repository implementations, UnitOfWork
  retrieval/      Vector + hybrid retrieval pipeline
  services/       Application service facade (ServiceMixin)
  storage/        SQLite persistence, image store
  vectorstore/    ZVec / InMemory vector store adapters
```

## Documentation

- [Getting Started](guide/getting-started.md)
- [Development Guide](guide/development.md)
- [Deployment Guide](guide/deployment.md)
- [API Reference](reference/api.md)
- [Configuration Reference](reference/configuration.md)
- [Architecture Overview](architecture/overview.md)
- [Design Decisions](architecture/decisions.md)
- [Monitoring & Observability](operations/monitoring.md)
- [Future Extensions](future.md)
