# SQLite Vector Store

The SQLite vector store persists embeddings in a local SQLite database.
It is the default for single-node deployments.

## Registration

```python
from raghub.stores.vector_sqlite import SqliteStore
```

Registered as `@Store.register("sqlite")`.

## Usage

```python
from raghub.stores.vector_sqlite import SqliteStore

store = SqliteStore(db_path="raghub.db")
store.create_collection()
store.insert(chunks, vectors)
results = store.search(vector=query_vector, top_k=5)
```

## Features

- `hybrid_search()` combines dense vectors with BM25 keyword scoring.
- `optimize()` runs `VACUUM` and rebuilds FTS indexes.
- `delete_version()` supports document versioning via metadata.
