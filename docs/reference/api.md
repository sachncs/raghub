# API Reference

RAGHub exposes two parallel surfaces:

1. **`raghub.RAG`** — the recommended Python facade. Typed Pydantic
   models in, typed Pydantic models out.
2. **FastAPI** (`uvicorn raghub.api.app:app`) — the legacy HTTP
   surface, bound to `DynamicRagApplication`, with JWT bearer
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
| `rag.astream(question, ...)` | `async for chunk in rag.astream(...): ...` — real token stream through `QueryPipeline.stream` → `DefaultGenerator.astream` → `LiteLLMProvider.astream`. |

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

## FastAPI surface (legacy)

`uvicorn raghub.api.app:app --host 0.0.0.0 --port 8000` mounts the
following endpoints. All endpoints except `/health` require
`Authorization: Bearer <session_token>`.

### `GET /health`

Service liveness probe. Delegates to `DynamicRagApplication.health()`.

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
processed by the `BackgroundIngestionService` mounted at
`app.state.background_ingestion`.

### `/admin/*`

Admin-only routes mounted from `raghub.api.admin` (e.g. user CRUD,
storage audit).

### `GET /metrics`

Prometheus scrape endpoint (when the optional
`prometheus_client` instrumentation is enabled).

---

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

Pre-seeds five demo users (see
[`guide/deployment.md`](../guide/deployment.md)). The UI uses
`st.chat_message` + `st.chat_input`, citation rendering per turn,
per-user document upload scoped to the user's company, and
persistent sign-in via `st.session_state`.

---

## Models

All public models live in `raghub.models`:

```python
from raghub.models import (
    UserPrincipal,             # the principal carrying allowed_companies
    Chunk, Document,           # canonical models
    KnowledgeBundle,           # OKF representation
    Citation, SearchResult,
    PipelineContext, PipelineResult,
    EvaluationResult,
    ConversationTurn,
)
```

`UserPrincipal`:

```python
UserPrincipal(
    user_id: str,
    email: str,
    allowed_companies: list[str] = [],
    allowed_groups: list[str] = [],
    is_admin: bool = False,
    created_at: datetime | None = None,
)
```
