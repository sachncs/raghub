# Troubleshooting

## `ModuleNotFoundError: No module named 'aiosqlite'`

The legacy storage path requires `aiosqlite`. Install it via
`pip install aiosqlite>=0.22` or use the new `RAG` facade which
falls back to the in-process `InMemoryVectorStore` and
`InMemoryKnowledgeRepository` when aiosqlite is unavailable.

## `ConfigurationError: litellm is not installed`

The RAG facade tries LiteLLM first; when LiteLLM is missing it falls
back to the offline `HeuristicLLMProvider` automatically. If you
explicitly passed `LiteLLMProvider(...)` to the constructor, install
LiteLLM: `pip install litellm`.

## `ConfigurationError: marker-pdf is not installed`

The RAG facade tries Marker first; when Marker is missing it falls
back to `PlainTextConverter`. Install Marker for PDF support:
`pip install marker-pdf`.

## `ConfigurationError: chonkie is not installed`

The RAG facade tries Chonkie first; when Chonkie is missing it falls
back to `WordWindowChunker`. Install Chonkie for token-aware
chunking: `pip install chonkie`.

## `ConfigurationError: qdrant-client is not installed`

The RAG facade tries Qdrant first; when Qdrant is missing it falls
back to `InMemoryVectorStore`. Install Qdrant for production:
`pip install qdrant-client`.

## `ConfigurationError: instructor is not installed`

The structured-output provider is optional. Install Instructor for
typed Pydantic outputs: `pip install instructor`.

## `RuntimeError: asyncio.run() cannot be called from a running event loop`

You are calling a sync `RAG.ingest/evaluate` from inside an event
loop (e.g. Jupyter, FastAPI handler). Use the async variants
(`RAG.aingest`, `RAG.aquery`, `RAG.astream`) instead.

## Langfuse traces do not appear

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the
environment. When the env vars are missing the facade falls back
to `NoOpTelemetry`. Verify with `rag.health()["telemetry"]`.

## Qdrant returns no results

Qdrant requires an explicit collection create step. Call
`RAG.initialize()` once after construction; the facade delegates
to `vector_store.create_collection()`.

## Incremental indexing is skipping updates

`RAG.ingest(...)` short-circuits when the source's SHA-256 hasn't
changed. Pass `force=True` to force re-indexing, or call
`RAG.sync_index(directory)` to reconcile a whole directory.

## `RAG.evaluate()` raises `EvaluationError`

The `evaluator.evaluate(...)` call wrapped by the facade is the
FinanceBench evaluator. When `examples=None` it tries to download
the dataset; the download may fail in restricted environments.
Pass an explicit `examples=[...]` to bypass the download.
