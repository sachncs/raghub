# Architecture Overview

## Domain Layer

Domain models follow the **Active Record** pattern. Each domain object wraps a Pydantic DTO and provides `build`, `update`, and `remove` behavior through injected repositories.

```
Document  -> DocumentRecord (DTO) -> SqliteDocumentRepository
Chunk     -> ChunkRecord (DTO)     -> SqliteChunkRepository -> BaseVectorStore
Session   -> SessionRecord (DTO)   -> SqliteSessionRepository
```

### UnitOfWork

The `UnitOfWork` aggregates all three repositories and is passed to domain objects for persistence context.

```
UnitOfWork
  +-- DocumentRepository -> SqliteDocumentRepository
  +-- ChunkRepository    -> SqliteChunkRepository -> BaseVectorStore
  +-- SessionRepository  -> SqliteSessionRepository
```

## Service Layer

`DynamicRagApplication` is the use-case facade. It delegates to:

- `DocumentService` for document upload and lifecycle
- `QueryService` for question answering
- `AuthService` for login, logout, user resolution
- `HealthService` for health checks

All services inherit from `ServiceMixin` which provides `log()` and `emit_metric()`.

## Data Flow

### Document Ingestion

```
Upload -> ParserRegistry -> ChunkingPlan -> EmbeddingProvider -> VectorStore
         \-> ImageStore                                     \-> SqliteDocumentRepository
```

### Query

```
Question -> EmbeddingProvider -> VectorStore.search -> LLM.generate -> Response
             \-> RBAC filter     \-> IdentityReranker
```

## Storage

All SQLite stores use `DatabaseManager` which enforces:
- WAL journal mode (concurrent reads)
- Foreign key enforcement
- Connection sharing across repositories

## Vector Stores

| Store | Description |
|-------|-------------|
| ZVec | Production vector store with SQLite-backed vectors |
| InMemoryVectorStore | Ephemeral store for testing |

## Security

- JWT bearer tokens for API authentication (PyJWT + bcrypt)
- RBAC with company-level access control
- No passwordless login path in production
- Rate limiting via custom `RateLimiterMiddleware` (token bucket)
- CORS middleware configuration

## Observability

- Prometheus metrics (histograms for latency, counters for operations)
- OpenTelemetry tracing (FastAPIInstrumentor)
- structlog for structured logging
