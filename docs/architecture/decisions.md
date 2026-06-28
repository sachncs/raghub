# Design Decisions

## Architecture

| Decision | Choice |
|----------|--------|
| Vector store | ZVec (production), InMemoryVectorStore (dev/test) |
| Embeddings | NVIDIA NV-Embed-QA, SentenceTransformers (all-MiniLM-L6-v2) |
| LLM | NVIDIA Llama 3.3 Nemotron Super 49B |
| Auth | JWT (PyJWT) + bcrypt password hashing |
| Persistence | SQLite via aiosqlite with DatabaseManager |
| Domain pattern | Active Record (build, update, remove) |
| Package structure | Flattened (no src/) |

## Key Decisions

### Fully Async Stack
All I/O (database, HTTP, embeddings, LLM) uses `asyncio`. No `to_sync()` wrappers. Database operations use `aiosqlite`.

### Shared DatabaseManager
All SQLite repositories (`SqliteDocumentRepository`, `SqliteChunkRepository`, `SqliteSessionRepository`) accept an optional `db_manager`. When provided, they share a single connection; otherwise they create one per call. `DatabaseManager` enables WAL mode and foreign keys on connect.

### No Passwordless Login
The passwordless login path was removed. `AuthLoginRequest` requires `password: str`. Production config has `allow_passwordless_login: false`.

### No Underscore-Prefixed Names
All "private" methods and attributes were renamed to not use underscore prefixes (e.g., `_conn` → `conn`, `_run_job` → `run_job`). The codebase enforces this convention.

### ServiceMixin
`ServiceMixin` in `services/__init__.py` provides shared `log()` and `emit_metric()` for all four service classes, eliminating duplicate implementations.

### Retry with Exponential Backoff
NVIDIA API calls use inline retry wrappers (`_retry` in `llm/nvidia.py`, `_retry_embed` in `embeddings/nvidia.py`) with exponential backoff rather than external dependencies like tenacity.

### Factory Routing
`build_embedding_provider` routes to NVIDIA only when the model name contains `"nvidia"`. `build_llm_provider` applies the same convention.

### FinanceBench PDFs
Evaluation downloads PDFs from GitHub raw URLs (`pdfs/` directory), not SEC EDGAR.

### Admin Company Access
`allowed_company_filter()` returns `""` (match-all) for `is_admin=True` users, bypassing company-level filtering.
