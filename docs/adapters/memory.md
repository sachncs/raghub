> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Memory Vector Store

The in-memory vector store is the default for development and testing.
It requires no external dependencies and stores everything in a Python list.

## Registration

```python
from raghub.stores.vector_memory import MemoryStore
```

Registered as `@Store.register("memory")`.

## Usage

```python
from raghub.stores.vector_memory import MemoryStore

store = MemoryStore()
store.create_collection()
store.insert(chunks, vectors)
results = store.search(vector=query_vector, top_k=5)
```

## Limitations

- Data is lost on process restart.
- Not suitable for production workloads.
- Hybrid search uses in-memory BM25 via `rank_bm25`.
