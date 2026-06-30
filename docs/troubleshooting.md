# Troubleshooting

Common failure modes, organised by symptom. The framework uses
graceful degradation for missing optional dependencies — every
default that depends on an external library falls back to an
in-process deterministic implementation when the library or key
is missing.

## `ImportError` or `ConfigurationError: litellm is not installed`

The `RAG` facade tries LiteLLM first; when LiteLLM is missing or
no LLM API key is configured, it falls back to the offline
`HeuristicLLMProvider`. If you explicitly passed
`LiteLLMProvider(...)` to the constructor, install it:

```bash
pip install litellm
```

## `ConfigurationError: marker-pdf is not installed`

The facade tries Marker first; when Marker is missing it falls
back to `PlainTextConverter`. Install Marker for PDF support:

```bash
pip install marker-pdf
```

## `ConfigurationError: chonkie is not installed`

The default chunker uses Chonkie when available; it falls back to
`WordWindowChunker` otherwise. Install Chonkie for token-aware
chunking:

```bash
pip install chonkie
```

## `ConfigurationError: qdrant-client is not installed`

The facade tries Qdrant first; when `qdrant-client` is missing or
`QDRANT_URL` is not set, it falls back to `InMemoryVectorStore`.
Install Qdrant for production:

```bash
pip install qdrant-client
# In your environment:
export QDRANT_URL=...
```

## `ConfigurationError: instructor is not installed`

Structured output is optional. The `RAG.query(..., response_model=...)`
path returns `Response.structured = None` when Instructor is
missing or no LLM API key is configured. Install Instructor for
typed Pydantic outputs:

```bash
pip install instructor
export OPENAI_API_KEY=...      # Instructor requires an LLM credential
```

## `RuntimeError: asyncio.run() cannot be called from a running event loop`

You are calling a sync method (`RAG.ingest`, `RAG.query`,
`RAG.evaluate`, `RAG.delete`, `RAG.health`, `RAG.sync_index`)
from inside an event loop (e.g. Jupyter, a FastAPI handler,
`asyncio.run` already active).

Use the async variants instead — they are non-blocking and
cooperate with the running loop:

```python
await rag.aingest(b"...")
response = await rag.aquery("...", user=alice)
async for piece in rag.astream("...", user=alice):
    ...
```

## Langfuse traces do not appear

Set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in the
environment. When the env vars are missing (or the
`langfuse` package is not installed) the facade falls back to
`NoOpTelemetry` and every method is a no-op. Verify with:

```python
rag.health()["telemetry"]
# 'LangfuseTelemetryProvider' or 'NoOpTelemetry' or 'RedactingTelemetry'
```

Optional: set `LANGFUSE_HOST` to a self-hosted endpoint.

## Qdrant returns no results on first run

Qdrant requires an explicit collection-create step. Call once:

```python
rag.initialize()
```

…the facade delegates to `vector_store.create_collection()`.
After that, retrieval should return results as expected.

## Incremental indexing is skipping updates

`RAG.ingest(...)` short-circuits when the source's SHA-256 hasn't
changed (`knowledge_repo.get(bundle_id)` returns the prior
checksum). To force re-embedding:

```python
rag.ingest(b"...", force=True)
```

…or reconcile an entire directory in one go:

```python
rag.sync_index("./documents", user=alice)
# {"added": [...], "modified": [...], "unchanged": [...], "removed": [...]}
```

## `RAG.evaluate()` raises `EvaluationError`

`RAG.evaluate(benchmark="financebench")` uses
`FinanceBenchEvaluator`, which downloads the dataset from
HuggingFace when no `examples=` is supplied. The download may fail
in restricted environments or where the network is offline.

Bypass the download by passing explicit examples:

```python
from pathlib import Path
examples = FinanceBenchEvaluator(dataset_path=Path("./local.jsonl"))
rag.evaluate(benchmark="financebench", examples=examples)
```

…or place a JSONL file at
`${RAGHUB_FINANCEBENCH_CACHE:-~/.cache/raghub/financebench}/financebench.jsonl`.

## `RAG.query` returns no chunks for a user

When the caller supplies a `UserPrincipal` with an empty
`allowed_companies` (and is not admin), the retrieval layer
filters to `{"company": []}` which matches nothing. The user
receives an empty result set, not a 403. Give the user at least
one company, or set `is_admin=True`.

## `JWT_SECRET` rejected in production

`load_settings` enforces that `JWT_SECRET` is set and is at
least 32 UTF-8 bytes long when `environment == "production"`.

```bash
export JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
echo -n "$JWT_SECRET" | wc -c   # must be >= 32
```

PyJWT's `InsecureKeyLengthWarning` is treated as fatal in CI.

## FastAPI auth errors after upgrade

The legacy `DynamicRagApplication` requires `JWT_SECRET` when
`allow_passwordless_login=False`. Set both before booting the
service:

```bash
export JWT_SECRET=...
# Set in config/<profile>.yaml (or via env):
# allow_passwordless_login: false
```

## `load_settings` raises `RuntimeError: Passwordless login is forbidden in production`

Either flip the profile (`environment: staging`) or set
`allow_passwordless_login: false` in the YAML profile. The loader
rejects `allow_passwordless_login: true` in production.

## Telemetry attributes still contain secrets

The default `RedactingTelemetry` scrubs kwarg keys matching
`password|secret|api_key|token|jwt|authorization` (case
insensitive, recursive into nested dicts). If you pass your own
`telemetry=` to the facade, you bypass the redaction layer.
Either re-wrap with `RedactingTelemetry(...)` or scrub at the
sink.

## Multi-user isolation isn't visible from the UI

The Streamlit UI signs the caller in as a demo user; the
session's `UserPrincipal` is what the facade's RBAC sees. Sign
in with a non-admin user (e.g. `alice@acme.com`) and confirm
the companies box shows `Apple`. All retrieved chunks should
have `chunk.company == "Apple"`.

## Streaming returns empty

`RAG.astream` requires the underlying LLM to expose an `astream`
method. The default `HeuristicLLMProvider` does not stream by
default — switch to a real `LiteLLMProvider` for true streaming:

```python
from raghub.llm.litellm import LiteLLMProvider

rag = RAG()
rag.llm = LiteLLMProvider(model="nvidia/llama-3.3-nemotron-super-49b-v1.5")
```

The facade's `QueryPipeline.stream` will then route through
`LiteLLMProvider.astream` with
`stream_options={"include_usage": True}`.

## Performance regressions after a config change

Check that `chunk_size_words`, `top_k`, `embedding_dim`, and the
embedding model haven't been accidentally overridden. Run the
benchmark to baseline:

```bash
python -m bench.benchmark --documents 100 --queries 200 --concurrency 8
```

The report is at `bench/report.json`. Compare latency p50 / p95
and queries-per-second across runs.
