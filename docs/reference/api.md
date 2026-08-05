# API Reference

RAGHub exposes two parallel surfaces:

1. **`raghub.RAG`** — the recommended Python facade. Typed Pydantic
   models in, typed Pydantic models out.
2. **FastAPI** (`uvicorn raghub.api:AppFactory.create_app --factory`)
   — the HTTP surface, bound to the RAG facade, with bearer-token
   auth.

---

## Python facade: `raghub.RAG`

```python
from raghub import RAG

rag = RAG()                   # default components
# or
rag = RAG.from_config("raghub.yaml")
# or
rag = RAG(
    settings=...,
    converter=...,
    chunker=...,
    embedder=...,
    llm=...,
    vector_store=...,
    generator=...,
    reranker=...,
    structured=...,
    telemetry=...,
    knowledge_repo=...,
    registry=...,
    background_service=...,
    manifest=...,
)
```

### Construction

| API | Description |
|---|---|
| `RAG(*, settings=None, ...components)` | Build with explicit collaborators. |
| `RAG.from_config(path: str \| Path)` | Load YAML/TOML profile, then construct. |

### Lifecycle

| API | Description |
|---|---|
| `rag.initialize()` | Call `vector_store.create_collection()` and `knowledge_repo.initialize()` when present. Idempotent. |
| `rag.shutdown()` | Close telemetry, vector store, knowledge repo, embedder, LLM, generator, background service. Errors are swallowed so the rest of the shutdown still completes. |

### Ingestion

| API | Description |
|---|---|
| `rag.ingest(source, *, source_uri=None, mime_type="text/plain", metadata=None, force=False, user=None)` | Sync ingest of file / directory / bytes. Directories are walked recursively. |
| `rag.aingest(...)` | Async equivalent. |
| `rag.delete(document_id)` | Delete by `bundle_id` or `source_uri`. Removes chunks, knowledge bundle, and source-manifest entry. |
| `rag.ingest_async(source, ...)` | Submit a background ingestion job. Returns a `job_id`. |
| `rag.job_status(job_id)` | Returns the job status (or `None` when `background_service` was never used). |
| `rag.sync_index(directory, *, metadata=None, user=None)` | Reconcile `directory` against the SHA-256 manifest. Returns `{"added": [...], "modified": [...], "unchanged": [...], "removed": [...]}`. |

### Query

| API | Description |
|---|---|
| `rag.query(question, *, user=None, session_id=None, top_k=5, metadata_filter=None, response_model=None)` | Synchronous. Blocks on the async path; safe to call outside an event loop. |
| `rag.aquery(...)` | Async equivalent. |
| `rag.astream(question, ...)` | `async for chunk in rag.astream(...): ...` — real token stream through `QueryPipeline.stream` → `DefaultGenerator.astream` → `LiteLLM.astream`. |

### Diagnostics and conversation

| API | Description |
|---|---|
| `rag.health()` | Returns a dict summarising every collaborator (see below). |
| `rag.conversation_history(session_id, *, user=None, limit=50)` | Returns the most recent turns for a session scoped by the user's `user_id`/`email`. |
| `rag.clear_conversation(session_id, *, user=None)` | Clear the conversation history for a session. |

### Response shape

`RAG.query` / `RAG.aquery` return a `CanonicalResponse` (`Response`):

```python
print(response)
# answer:        str
# citations:     list[Citation]
# source_chunks: list[SearchResult]
# structured:    dict[str, Any] | None   (set when response_model=... was used)
# metadata:      dict
```

### Health shape

`RAG.health()` returns:

```python
{
    "status":       "ok",
    "vector_store": "<class name>",
    "embedder":     "<class name>",
    "llm":          "<class name>",
    "chunker":      "<class name>",
    "converter":    "<class name>",
    "telemetry":    "<class name>",
    "structured":   "<class name>" | None,
    "reranker":     "<class name>",
}
```

---

## FastAPI surface

`uvicorn raghub.api:AppFactory.create_app --factory` mounts the
following endpoints. All endpoints except `/health` require
`Authorization: Bearer <session_token>`.

### `GET /health`

Service liveness probe. Delegates to `RagApplication.health()`.

### `POST /auth/login`

```json
{"email": "alice@acme.com", "password": "password"}
```

Returns:

```json
{
  "session_token": "...",
  "user_email": "alice@acme.com",
  "allowed_companies": ["Apple"]
}
```

### `POST /auth/logout`

Invalidate the current session. Returns `{"status": "logged_out"}`.

### `POST /documents/upload`

Multipart form with `file` (binary) and optional `company`
(string override). Returns `202` with the new document metadata:

```json
{
  "document_id": "...",
  "version": "...",
  "status": "...",
  "company": "...",
  "filename": "..."
}
```

### `GET /documents`

List documents visible to the calling user. Returns
`{"documents": [<DocumentRecord>, ...]}`.

### `GET /documents/{document_id}/status`

Return the latest `DocumentRecord` for the given id.

### `DELETE /documents/{document_id}`

Delete a document and all of its chunks. **Admin-only.** Returns `204`.

### `POST /query`

```json
{"question": "What was the total revenue in 2023?"}
```

Returns a typed `QueryResponse` (answer, citations, source chunks).

### `GET /session/history`

Returns `{"history": [<ConversationTurn>, ...]}` (oldest first).

### `DELETE /session/history`

Clear conversation history. Returns `204`.

### `POST /ingest/async`

Multipart form with `file`. Returns `{"job_id": "..."}`; the job is
processed by the `Batch` mounted at
`app.state.background_ingestion`.

### `/admin/*`

Admin-only routes mounted from `raghub.api.admin` (e.g. user CRUD,
storage audit).

### `GET /metrics`

Prometheus scrape endpoint (when the optional
`prometheus_client` instrumentation is enabled).

### `POST /v1/feedback`

Submit user feedback for a specific chunk. Records a positive or
negative rating with an optional note.

```json
{
  "chunk_id": "chunk-123",
  "rating": "positive",
  "note": "Highly relevant"
}
```

Returns `201` with the feedback id:

```json
{
  "feedback_id": "...",
  "user_id": "alice@acme.com",
  "chunk_id": "chunk-123",
  "rating": "positive",
  "tenant_id": "acme-corp",
  "created_at": "2024-01-01T00:00:00Z"
}
```

The `user_id` and `tenant_id` are resolved from the bearer token; the
submitted values are ignored.

### `GET /v1/feedback/{feedback_id}`

Retrieve a single feedback record by id. Returns `404` if not found
or not visible to the calling tenant.

### `DELETE /v1/feedback/{feedback_id}`

Delete a feedback record. The feedback must be owned by the calling
tenant. Returns `204`.

### `GET /v1/feedback/aggregate`

Aggregate feedback for the calling tenant. Returns:

```json
{
  "tenant_id": "acme-corp",
  "positives": 42,
  "negatives": 3,
  "by_chunk": {
    "chunk-123": {"positive": 5, "negative": 0},
    "chunk-456": {"positive": 0, "negative": 2}
  }
}
```

---

## Rate limiting

The HTTP surface applies a token-bucket rate limit per bearer token.
The limiter is configured via `RAGHUB_RATELIMIT_PER_MINUTE` (default
120) and `RAGHUB_RATELIMIT_BURST` (default 20).

Every response includes:

| Header | Description |
|---|---|
| `X-RateLimit-Limit` | The maximum number of requests permitted per minute. |
| `X-RateLimit-Remaining` | The number of requests remaining in the current window. |
| `X-RateLimit-Reset` | Unix timestamp when the window resets. |

When the limit is exceeded, the server returns `429 Too Many Requests`
with a `Retry-After` header (in seconds) and a JSON body:

```json
{"error": "rate_limit_exceeded", "retry_after": 5}
```

---

## Queue CLI sub-commands

The CLI surface (under `raghub` / `python -m raghub.cli`) exposes the
persistent queue as a standalone sub-app:

```bash
# Submit a job manually
raghub queue submit --kind ingest --payload '{"source": "..."}'

# Start workers (foreground)
raghub queue run --workers 4

# Inspect queue state
raghub queue stats --json
raghub queue list --status pending --limit 100

# Drain dead-letter queue
raghub queue purge --status dead
```

All sub-commands exit with code 0 on success and a non-zero code on
failure.

---

## Models

All public models live in `raghub.models`:

```python
from raghub.models import (
    User,             # the principal carrying allowed_companies
    Chunk, Document,           # canonical models
    Bundle,           # OKF representation
    Citation, SearchResult,
    PipelineCtx, PipelineResult,
    Result,
    ConversationTurn,
)
```

`User`:

```python
User(
    user_id: str,
    email: str,
    allowed_companies: list[str] = [],
    allowed_groups: list[str] = [],
    is_admin: bool = False,
    created_at: datetime | None = None,
)
```
