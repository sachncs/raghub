# Changelog

## [Unreleased]

### Changed
- Full async conversion — all I/O uses asyncio (aiosqlite, httpx)
- Consolidated all SQLite stores under shared `DatabaseManager` (WAL mode, FK enforcement)
- Removed passwordless login path — `AuthLoginRequest` requires `password`
- Renamed all underscore-prefixed "private" methods/attributes across codebase
- Consolidated `SqliteDocumentRegistry` into `SqliteDocumentRepository`
- All service classes now inherit from `ServiceMixin` (shared `log()`/`emit_metric()`)
- Restructured documentation into subdirectories

### Fixed
- NvidiaLLMProvider extends BaseLLMProvider with 120s timeout
- NvidiaEmbeddingProvider / SentenceTransformerEmbeddingProvider implement `embed_text()`/`embed_texts()`
- Factory routing only routes to NVIDIA when model name contains "nvidia"
- All 78+ non-FinanceBench tests pass
- 0 ruff errors, 0 mypy errors in `raghub/`
- Coverage improved to 76%

### Added
- `tests/test_retrieval.py` — 14 tests for RetrievalPipeline, FacetedSearchEngine
- `tests/test_financebench.py` — FinanceBench evaluation test suite
- `evaluate_financebench.py` — standalone eval script (downloads data+PDFs from GitHub)
- Retry wrappers with exponential backoff for NVIDIA API calls
- `ServiceMixin` in `services/__init__.py`

### Removed
- Dead modules: `raghub/chunking.py`, `raghub/cache/lru.py`, `raghub/memory/session_memory.py`
- Stale skeleton in `src/raghub/`
- `app/` directory and `tests/app/`
- `EmailDirectory`, `InMemoryAuthenticator`, `TokenSessionManager`, `CompanyAuthorizationService`
- Duplicate `detect_mime_type` in `document_service.py`
- Old flat documentation files (consolidated into `docs/` subdirectories)
