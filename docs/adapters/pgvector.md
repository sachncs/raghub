# PgVector Store

The pgvector store uses PostgreSQL with the `pgvector` extension for
production-grade vector search.

## Registration

```python
from raghub.stores.pgvector import PgVectorStore
```

Registered as `@Store.register("pgvector")`.

## Usage

```python
from raghub.stores.pgvector import PgVectorStore

store = PgVectorStore(dsn="postgresql://localhost/raghub", embedding_dim=1536)
await store.initialize()
await store.insert(chunks, vectors)
results = await store.search(vector=query_vector, top_k=5)
```

## Features

- HNSW or IVFFlat indexing (configurable via `index_type`).
- `hybrid_search()` combines dense search with Postgres FTS via RRF fusion.
- `optimize()` runs `VACUUM ANALYZE` and rebuilds HNSW indexes.
- Row-level security hooks via `set_session()`.

## Requirements

- `asyncpg>=0.29` (core dependency).
- PostgreSQL 15+ with `vector` extension enabled.
