# RAGHub — Complete Remediation Plan

> Everything here gets done. No skips.

---

## Phase 0 — Blockers (fix first, everything depends on these)

### 0.1 Restore a working offline LLM default
- [ ] Add `HeuristicProvider` class in `generation.py` — returns canned/enumerated answer, no API key needed
- [ ] Change `default_llm()` to return `HeuristicProvider(...)` when no API key is present instead of raising
- [ ] Update `RAG.__init__` to succeed without env vars
- [ ] Remove or update the dead `llm_model = "heuristic-llm"` default in `config.py`
- [ ] Verify: `pip install -e . && python -c "from raghub import RAG; RAG()"` succeeds

### 0.2 Break the circular import that kills `raghub --help`
- [ ] Move `from raghub.auth import RBACAuthorizationService, SqliteUserStore` inside the function that needs them in `helper/services.py`
- [ ] Verify: `raghub --help` exits 0
- [ ] Add CI smoke test: `raghub --help` must exit 0

### 0.3 Split dependencies into optional extras
- [ ] Define extras in `pyproject.toml`:
  - `[pdf]`: marker-pdf
  - `[graph]`: python-igraph, leidenalg, scikit-learn
  - `[rerank]`: cohere, ragatouille, rank-bm25
  - `[web]`: duckduckgo-search
  - `[docs]`: Pillow, openpyxl, python-docx, python-pptx, beautifulsoup4
  - `[auth]`: bcrypt, aiosqlite
  - `[otel]`: opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi
  - `[langfuse]`: langfuse
  - `[tiktoken]`: tiktoken
  - `[all]`: everything
- [ ] Add lazy import wrappers with `OptionalDependencyMissing` for each optional dep
- [ ] Keep only core deps required: pydantic, numpy, PyYAML, loguru, tqdm, typer, litellm, instructor, pypdf, chonkie, prometheus-client, datasets, python-dateutil, httpx, aiohttp
- [ ] Verify `pip install raghub` installs only core deps

### 0.4 Fix config profile path
- [ ] `config.py`: Replace `Path.cwd() / "config"` with something installable
  - Option A: `importlib.resources.files("raghub") / "config"`
  - Option B: Accept config path as parameter, default to env var `RAG_CONFIG_DIR`
  - Option C: Use XDG config dir (`~/.config/raghub/`)
- [ ] Update `load_profile_payload()` to not assume CWD

### 0.5 Implement or remove RAGHUB_STORE_BACKEND
- [ ] Either: add code to `Settings` that reads `RAGHUB_STORE_BACKEND` and switches store backend
- [ ] Or: remove the env var from README entirely

### 0.6 Handle .env auto-load from pydantic-settings
- [ ] Check if `Settings` inherits from `pydantic.BaseSettings` (auto-loads `.env`)
- [ ] If yes, disable `.env` auto-load in production or document behavior
- [ ] Add `.env` to `.gitignore`

---

## Phase 1 — Blockers continued: data verification

### 1.1 Make insert() return row count
- [ ] `BaseVectorStore.insert()` signature: `-> int`
- [ ] `InMemoryVectorStore.insert()`: `return len(chunks)`
- [ ] `SqliteVectorStore.insert()`: return `cursor.rowcount` after commit
- [ ] Same for `upsert()`
- [ ] Update callers in `pipeline.py`, `rag.py` to check `count >= len(chunks)`

### 1.2 Validate vector dimensions on insert
- [ ] `SqliteVectorStore.__init__` stores `self.embedding_dim`
- [ ] Both `InMemoryVectorStore` and `SqliteVectorStore`: `assert all(len(v) == self.embedding_dim for v in vectors)` at start of `insert()`/`upsert()`
- [ ] Raise `VectorStoreError` on mismatch (not bare `AssertionError`)

### 1.3 Populate ChunkRecord.hash
- [ ] `pipeline.py`: compute `sha256(chunk.text.encode()).hexdigest()` per chunk
- [ ] Pass `hash` to `ChunkRecord` constructor at creation point
- [ ] Remove default `""` from `models.py:270` (make it required)

### 1.4 Catch litellm exceptions properly
- [ ] Add except clauses for: `litellm.AuthenticationError`, `litellm.RateLimitError`, `litellm.APITimeoutError`, `litellm.ServiceUnavailableError`
- [ ] Wrap each in `LLMError` with preserved message
- [ ] Add `try/except` around response shape extraction (`choices[0].message.content`) to catch `KeyError`/`TypeError` → `LLMError`

### 1.5 Add retry with exponential backoff
- [ ] Write a simple `@retry` decorator (no new dependency): `max_attempts=3, base_delay=0.5, backoff=2.0`
- [ ] Apply to vector store writes in `pipeline.py`
- [ ] Apply to LLM completion calls in `llm.py`
- [ ] Retry on: `VectorStoreError`, `LLMError`, transient net errors

### 1.6 Add pre/post conditions to RAG facade
- [ ] `rag.py` `query()`: guard `if not question.strip(): raise ValidationError(...)`
- [ ] `rag.py` `ingest()`: verify result by checking store after insert
- [ ] `rag.py` `ingest_directory_concurrent`: replace `contextlib.suppress(Exception)` with explicit error handling that tracks failed files

### 1.7 Add model validators
- [ ] `SearchResult`: `@model_validator(mode='after')` verifying `chunk_id == chunk.chunk_id`
- [ ] `Response`: verify every citation's `chunk_id` appears in `source_chunks`
- [ ] `PipelineResult`: `@model_validator` requiring `error` when `success=False`

### 1.8 Enable SQLite foreign keys and constraint enforcement
- [ ] `PRAGMA foreign_keys = ON` in `SqliteVectorStore.__init__`
- [ ] Add `NOT NULL` constraints to essential columns
- [ ] `INSERT OR REPLACE` → `INSERT OR IGNORE` + explicit dedup check

### 1.9 Harden config env-var loading
- [ ] Wrap `int()`/`float()` coercions in `try/except ValueError` with clear error message
- [ ] Validate Literal values against allowed set before `cast()`
- [ ] Log warning when production settings use built-in defaults
- [ ] Unify API key env var name: pick `RAG_LLM_API_KEY` and use consistently in `has_llm_api_key()`, error messages, defaults

---

## Phase 2 — Naming (rename everything, single pass)

### 2.1 Shorten all class names > 25 chars to single words
Target list (rename + update all callers in one atomic step):

| Old | New |
|-----|-----|
| `ResumableBackgroundIngestionService` | `Resumable` |
| `InstructorStructuredOutputProvider` | `Instructor` |
| `BackgroundIngestionService` | `Batch` |
| `InMemoryKnowledgeRepository` | `MemoryRepo` |
| `InMemoryVectorStore` | `MemoryStore` |
| `InMemoryConversationStore` | `MemoryConversations` |
| `InMemoryQueue` | `MemoryQueue` |
| `SqliteVectorStore` | `SqliteStore` |
| `SqliteUserStore` | `SqliteUsers` |
| `SqliteChunkRepository` | `ChunkStore` |
| `SqliteDocumentRepository` | `DocStore` |
| `SqliteSessionRepository` | `SessionStore` |
| `LiteLLMProvider` | `LiteLLM` |
| `LiteLLMEmbeddingProvider` | `LiteLLMEmbedder` |
| `HashingEmbeddingProvider` | `Hasher` |
| `BaseEmbeddingProvider` | `Embedder` |
| `BaseLLMProvider` | `Generator` |
| `BaseVectorStore` | `Store` |
| `BaseTool` | `Tool` |
| `BaseConverter` | `Converter` |
| `BaseChunker` | `Chunker` |
| `DocumentIngestionService` | `Ingestor` |
| `RBACAuthorizationService` | `Authz` |
| `DocumentLifecycleManager` | `Lifecycle` |
| `DocumentConverter` | `Converter` |
| `ChonkieChunker` | `Chonkie` |
| `WordWindowChunker` | `WordChunker` |
| `MarkdownSection` | `Section` |
| `PlainTextConverter` | `TextConverter` |
| `MarkerConverter` | `Marker` |
| `MarkerPdfConverter` | `PdfConverter` |
| `RaptorIndex` | `Raptor` |
| `GraphRagIndex` | `GraphIndex` |
| `SourceManifest` | `Manifest` |
| `AgenticQueryPipeline` | `AgentPipeline` |
| `UserPrincipal` | `User` |
| `PipelineContext` | `PipelineCtx` |
| `PluginRegistry` | `PluginRegistry` (already single word) |
| `BackgroundWorker` | `Worker` |
| `OptionalDependencyMissing` | `MissingDep` |
| `StructuredOutputProvider` | `Structured` |
| `ConversationManager` | `Conversations` |
| `SlidingWindowManager` | `SlidingWindow` |
| `TokenBucket` | `TokenBucket` (OK, well-known term) |
| `RateLimiterMiddleware` | `RateLimiter` |
| `LlmJudge` | `LlmJudge` (OK) |
| `Cohere` | `Cohere` (OK) |
| `Cascade` | `Cascade` (OK) |
| `Identity` | `Identity` (OK) |
| `Compose` | `Compose` (OK) |
| `Fusion` | `Fusion` (OK) |
| `Hyde` | `Hyde` (OK) |
| `MultiQuery` | `MultiQuery` (OK) |
| `StepBack` | `StepBack` (OK) |
| `Decompose` | `Decompose` (OK) |
| `Colbert` | `Colbert` (OK) |
| `RerankerFactory` | `RerankerFactory` (OK) |
| `Retrieval` | `Retrieval` (OK) |
| `DefaultGenerator` | `DefaultGen` |

### 2.2 Shorten all function and method names > 25 chars

| Old | New |
|-----|-----|
| `record_rerank_latency_provider` | `rerank_latency` |
| `validate_cors_for_credentials` | `validate_cors` |
| `authentication_error_handler` | `auth_error` |
| `authorization_error_handler` | `authz_error` |
| `assert_production_invariants` | `production_check` |
| `chunks_from_knowledge_bundle` | `get_chunks` |
| `markdown_to_document_blocks` | `md_to_blocks` |
| `ingest_directory_concurrent` | `ingest_dir` |
| `normalise_litellm_response` | `normalise_response` |
| `extract_text_from_content` | `extract_text` |
| `select_converter_for_path` | `pick_converter` |
| `marker_converter_instance` | `get_marker` |
| `load_long_context_config` | `load_longcontext` |
| `load_query_transforms_config` | `load_transforms` |
| `load_simple_env_payload` | `load_env` |
| `load_profile_payload` | `load_profile` |
| `load_agent_config` | `load_agent` |
| `load_web_search_config` | `load_web` |
| `load_reranker_config` | `load_reranker` |
| `load_hybrid_config` | `load_hybrid` |
| `numeric_within_tolerance` | `within_tolerance` |
| `marker_text_from_rendered` | `rendered_text` |
| `marker_text_from_pdf_bytes` | `pdf_to_text` |
| `build_embedding_provider` | `build_embedder` |
| `build_vector_store` | `build_store` |
| `build_llm_provider` | `build_llm` |
| `build_context_prompt` | `context_prompt` |
| `list_all_records_helper` | `list_records` |
| `document_by_id_helper` | `get_doc` |
| `upload_record_helper` | `upload_record` |
| `parse_seed_users_json` | `parse_users` |
| `raise_missing_document` | `missing_doc` |

### 2.3 Shorten all constant names > 25 chars

| Old | New |
|-----|-----|
| `SUMMARISE_COMMUNITY_PROMPT` | `COMMUNITY_PROMPT` |
| `MULTI_QUERY_SYSTEM_PROMPT` | `MULTI_QUERY` |
| `STEP_BACK_SYSTEM_PROMPT` | `STEP_BACK` |
| `HYDE_SYSTEM_PROMPT` | `HYDE` |
| `DECOMPOSE_SYSTEM_PROMPT` | `DECOMPOSE` |
| `CONTEXT_SYSTEM_PROMPT` | `CONTEXT` |
| `MIME_TYPES_BY_EXTENSION` | `MIME_TYPES` |
| `EQUATION_BLOCK_RE` | `EQUATION_RE` |
| `MARKER_AVAILABLE` | `MARKER` (if boolean) |
| `ALL_LICENSE_WIKI_LINKS` | `ALL_LINKS` |

### 2.4 Rename modules to short single-word names

| Old | New |
|-----|-----|
| `vectorstore.py` | `store.py` (or keep as-is, already short) |
| `embeddings.py` | `embedder.py` |
| `repositories.py` | `repos.py` |
| `exceptions.py` | `errors.py` |
| `observability.py` | `telemetry.py` |
| `generation.py` | `gen.py` |
| `conversation.py` | `conv.py` |
| `ingestion.py` | `ingest.py` |
| `documents.py` | `docs.py` |
| `helper/evaluation.py` | `eval.py` |

### 2.5 Make remaining private `_STOPWORDS` public
- [ ] `helper/evaluation.py`: `_STOPWORDS` → `STOPWORDS`

### 2.6 Fix builtin name collisions
- [ ] `ingestion.py:834`: `all` → `all_ids`
- [ ] `models.py:270`: `hash` → `checksum`
- [ ] `observability.py:100`, `pipeline.py:205`: `set` → `registry_set` / `filter_set`
- [ ] `helper/sse.py:18`: `Sse.format` → `Sse.format_event`

### 2.7 Remove import aliases with `_` prefix
- [ ] `llm.py:196`: `import httpx as _httpx` → inline `import httpx` inside the function
- [ ] `rag.py:322,917`: `import json as _json` → inline imports
- [ ] `rag.py:323`: `from pathlib import Path as _P` → inline import
- [ ] `rag.py:325`: `from raghub.config import Settings as _S` → inline import
- [ ] `helper/services.py:799`: `import importlib as _importlib` → inline import
- [ ] `helper/evaluation.py:658`: `import csv as _csv` → inline import
- [ ] `helper/evaluation.py:681`: `import ast as _ast` → inline import
- [ ] `rag.py:145,191,214,243,255,258`: `from ... import ... as _X` → inline imports

---

## Phase 3 — Package Structure

### 3.1 Add `__all__` to every module
- [ ] `rag.py`: `__all__ = ["RAG", "Settings", "RagHubError"]`
- [ ] `config.py`: `__all__` = config classes only, not `env_bool`, `csv_to_transforms`, `read_toml_file`
- [ ] `models.py`: `__all__` = data models only (Chunk, ChunkRecord, DocumentRecord, ConversationTurn, SearchResult, Response, etc.), not protocols or helpers
- [ ] `exceptions.py`: `__all__` = new spec categories + `RagHubError`, not legacy aliases (keep legacy for import compat but remove from `__all__`)
- [ ] `agent.py`, `pipeline.py`, `llm.py`, `embedder.py`, `store.py`, etc: each gets `__all__` limiting to public surface
- [ ] Verify `from raghub import *` yields a clean small set

### 3.2 Eliminate re-export wrapper files
- [ ] `retrieval.py`: delete it, update all imports to `from raghub.helper.retrieval import ...`
- [ ] `services.py`: delete it, update all imports to `from raghub.helper.services import ...`
- [ ] `storage.py`: delete it, update all imports to `from raghub.helper.storage import ...`
- [ ] `tools.py`: delete it, update all imports to `from raghub.helper.tools import ...`
- [ ] `evaluation.py`: keep only the Typer CLI app code, remove the re-export layer
- [ ] `documents.py`: keep only actual code, remove the re-export layer

### 3.3 Update `__init__.py`
- [ ] Replace `def __getattr__` with explicit `from raghub.rag import RAG`
- [ ] Remove docstring claims about `services.Facade` and `core.build_application` if those aren't top-level exports
- [ ] Add additional top-level exports that users need: `Settings`, `RagHubError`, key config types

### 3.4 Move helper/ files into themed subpackages
- [ ] `helpers/retrieval.py` → `raghub/retrieval/` package (with `__init__.py`)
- [ ] `helpers/services.py` → `raghub/services/` package
- [ ] `helpers/storage.py` → `raghub/stores/` package
- [ ] `helpers/tools.py` → `raghub/tools/` package
- [ ] `helpers/evaluation.py` → `raghub/eval/` package
- [ ] `helpers/documents.py` → `raghub/docs/` package
- [ ] Keep shallow helpers (auth.py, cli.py, rate_limit.py, response.py, search.py, sse.py) at top level
- [ ] Delete `helper/` directory when done

---

## Phase 4 — Tests (restore + rewrite)

### 4.1 Restore conftest.py
- [ ] Restore from `865e194:tests/conftest.py`
- [ ] Remove `RAG_ZVEC_DIR` env var (zvec is gone)
- [ ] Update `JWT_SECRET` to match current requirement

### 4.2 Restore well-written test files from 865e194
- [ ] Recover these files from commit `865e194`:
  - `tests/test_vectorstore_memory.py` (60 tests, good coverage)
  - `tests/test_embeddings.py` (determinism, dimensions)
  - `tests/test_llm.py` (error wrapping, streaming, message construction)
  - `tests/test_pipeline.py` (e2e ingest-then-query, RBAC, caching)
  - `tests/test_rag_facade.py` (20+ tests on RAG class)
  - `tests/test_config_validation.py` (env_bool, CSV transforms, production guard)
  - `tests/test_ingestion.py` (dedup, retry, pipeline failure)
  - `tests/test_services.py` (upload RBAC, list scoping, delete atomicity)
  - `tests/test_exceptions.py` (hierarchy verification)
  - `tests/test_hypothesis_properties.py` (property-based tests)
  - `tests/test_production_readiness.py` (admin redaction, CORS, health)
  - `tests/test_end_to_end.py` (multi-user RBAC, streaming, workload)
  - `tests/test_storage_database.py` (connection lifecycle, durability)
- [ ] Fix import paths in each (renamed classes, changed module layout)
- [ ] Verify each file passes: `pytest -x tests/<file>.py`

### 4.3 Write SqliteVectorStore tests (new)
- [ ] `tests/test_sqlite_vectorstore.py`:
  - Insert + search 10 chunks
  - Verify dimension mismatch raises `VectorStoreError`
  - Verify delete removes chunks
  - Verify `health()` returns expected keys
  - Test `INSERT OR IGNORE` dedup
  - Test `PRAGMA foreign_keys` enforcement
  - Test concurrent access (2 threads)

### 4.4 Write config loading from YAML test (new)
- [ ] `tests/test_config_loading.py`:
  - Write a real YAML config file to `tmp_path`
  - Load it with `Settings.load()`
  - Verify all fields parsed correctly
  - Test with missing optional fields (defaults applied)
  - Test with invalid env var values (error raised)

### 4.5 Write integration test with real SQLite data flow (new)
- [ ] `tests/test_integration_data_flow.py`:
  - Create `SqliteStore` with real SQLite file
  - Create `RAG` with `HeuristicProvider` + `SqliteStore`
  - `rag.ingest()` a small document (plain text)
  - `rag.query()` and verify the response includes the ingested text
  - Verify `insert()` returned `count > 0`
  - Verify vector dimension guard fires on mismatch
  - Verify `ChunkRecord.hash` is non-empty
  - Verify search returns correct chunks

### 4.6 Write new tests for remaining gaps
- [ ] `tests/test_heuristic_llm.py`: Test HeuristicProvider generates answers, handles edge inputs
- [ ] `tests/test_retry.py`: Test retry decorator with transient failures
- [ ] `tests/test_model_validators.py`: Test SearchResult validators, Response cross-reference, PipelineResult constraints

### 4.7 Remove stale pyproject.toml test references
- [ ] Delete or update `[tool.ruff.lint.per-file-ignores]` entries for files that no longer exist
- [ ] Update `--cov-fail-under` to match restored coverage

---

## Phase 5 — Documentation

### 5.1 Fix README
- [ ] Update "No API keys required" → accurately describe that HeuristicProvider works offline, real LLM needs API key
- [ ] Remove Qdrant/zvec references entirely
- [ ] Remove `RAGHUB_STORE_BACKEND` and `QDRANT_*` env vars from table
- [ ] Fix quick start to show working code:
  ```python
  from raghub import RAG
  rag = RAG()
  rag.ingest("Hello world")
  result = rag.query("What is this?")
  print(result)
  ```
- [ ] Fix `await` example: wrap in `async def main(): ... asyncio.run(main())`
- [ ] Fix project structure diagram: `raghub/raghub/` not `raghub/src/raghub/`
- [ ] Fix "minimal environment" advice: use `pip install raghub` not `pip install -e ".[dev]"`
- [ ] Add env var docs for unified API key name `RAG_LLM_API_KEY`
- [ ] Document optional extras: `[pdf]`, `[graph]`, `[rerank]`, etc.

### 5.2 Replace .env with .env.example
- [ ] Rename `.env` → `.env.example`
- [ ] Remove real-looking API key string, replace with `sk-your-key-here`
- [ ] Add `.env` to `.gitignore`
- [ ] Provide commented-out defaults in `.env.example`

---

## Phase 6 — Polish

### 6.1 Unify API key env-var names
- [ ] `config.py`: change default/fallback to `RAG_LLM_API_KEY`
- [ ] `llm.py` `has_llm_api_key()`: check `RAG_LLM_API_KEY`
- [ ] Update all error messages to reference `RAG_LLM_API_KEY`
- [ ] Update `.env.example` to use `RAG_LLM_API_KEY`
- [ ] Update README to document `RAG_LLM_API_KEY`

### 6.2 Hide legacy exception aliases from public exports
- [ ] `exceptions.py`: keep legacy classes (`AuthenticationError`, `AuthorizationError`, etc.) for import compat
- [ ] Remove from `__all__` so `from raghub.exceptions import *` doesn't include them
- [ ] Add deprecation warning in legacy class `__init__` or via subclass: `warnings.warn("Use ConfigurationError instead", DeprecationWarning)`

### 6.3 Remove unused `_` import aliases from earlier cleanup
- [ ] `rag.py`: remove stale `_json`, `_P`, `_S`, `_MarkerConverter`, `_LiteLLMEmbeddingProvider`, `_LiteLLMProvider`, `_Instructor`, `_LangfuseTelemetryProvider`, `_NoOpTelemetry` import aliases (all now handled via lazy inline imports)

---

## Execution Order Summary

| Phase | What | Files touched | Est. time |
|-------|------|---------------|-----------|
| **0** | Blockers (crash fixes, deps, config paths, env) | ~10 | 2 days |
| **1** | Data verification (insert returns, dims, checksums, error handling, retry, validators, FK, config hardening) | ~12 | 3 days |
| **2** | Naming (rename everything — classes, functions, modules, constants, aliases) | ~50 | 3 days |
| **3** | Structure (`__all__`, re-export wrappers, subpackages, `__init__.py`) | ~35 | 2 days |
| **4** | Tests (restore, rewrite, new SqliteStore tests, integration, property tests) | ~90 | 4 days |
| **5** | Documentation (README, .env.example) | 2 | 1 day |
| **6** | Polish (API key names unified, legacy exceptions, stale alias cleanup) | ~5 | 0.5 day |
| **Total** | | ~90 unique files | ~15.5 days |
