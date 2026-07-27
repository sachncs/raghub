# Atomic Refactor Plan (v0.7)

Every row is one atomic change → one commit. No batching. No pauses. After all rows are executed, no `__all__`, no `_foo`, no lazy imports, no God-classes, no `# type: ignore`, no `# noqa`, no God-file (except `pipeline.py` which is the wiring file by design).

Tracking convention: `[x]` = done, `[ ]` = pending. Update the box as each atomic commit lands.

## Stage A — File moves (folders → flat files, rename dirs)

- [x] A.1 `raghub/exceptions/` → `raghub/exceptions.py`
- [x] A.2 `raghub/utils/` → `raghub/utils.py`
- [x] A.3 `raghub/prompts/` → `raghub/prompts.py`
- [x] A.4 `raghub/plugins/` → `raghub/plugins.py`
- [x] A.5 `raghub/generation/` + `raghub/structured/` → `raghub/generation.py`
- [x] A.6 `raghub/telemetry/` + `raghub/observability/` → `raghub/observability.py`
- [x] A.7 `raghub/conversation/` → `raghub/conversation.py`
- [x] A.8 `raghub/core/` → `raghub/core.py`
- [x] A.9 `raghub/llm/` → `raghub/llm.py`
- [x] A.10 `raghub/embeddings/` → `raghub/embeddings.py`
- [x] A.11 `raghub/domain/` → `raghub/domain.py`
- [x] A.12 `raghub/models/` → `raghub/models.py`
- [x] A.13 `raghub/repositories/` → `raghub/repositories.py`
- [x] A.14 `raghub/auth/` + `services/auth_service.py` → `raghub/auth.py`
- [x] A.15 `raghub/converters/` + `raghub/documents/{chunker,lifecycle,validation,versioning}.py` → `raghub/documents.py`
- [x] A.16 `raghub/vectorstore/` → `raghub/vectorstore.py`
- [x] A.17 `raghub/ingestion/` + `raghub/ingestion/chunkers/` → `raghub/ingestion.py`
- [x] A.18 `raghub/knowledge/` + `raghub/knowledge/structures/` → `raghub/knowledge.py`
- [x] A.19 `raghub/pipelines/` + `raghub/pipelines/rag/` → `raghub/pipeline.py`
- [x] A.20 `raghub/services/{document_service,health_service,workers}.py` → `raghub/services/{document,health,workers}.py` (rename, folder kept)
- [x] A.21 `raghub/services/application/{facade,shutdown,preferences}.py` keep; AuthCoordinator folds into facade
- [x] A.22 `raghub/services/auth_service.py` deleted (folded into auth.py via A.14)
- [x] A.23 `raghub/services/query_service.py` deleted (folded into pipeline.py via A.19)
- [x] A.24 `raghub/services/application/auth.py` deleted (folded into facade.py)
- [x] A.25 `raghub/api/rag.py` → `raghub/rag.py` (extract RAG class)
- [x] A.26 `raghub/api/defaults.py` deleted (folded into rag.py)
- [x] A.27 `raghub/api/async_runner.py` deleted (folded into utils.py)
- [x] A.28 `raghub/agent/{agent,builder,events,resolver,prompts}.py` → `raghub/agent.py`
- [x] A.29 `raghub/agent/tools/` → `raghub/tools/` (rename directory)

## Helper / cleanup commits (test fixes required by the flat modules)

- test: update tests/test_litellm_embedding.py to use the flat raghub.embeddings
- refactor: drop stale raghub/agent/ directory and tests/agent/ folder
- test: drop the duplicate tests/agent/ folder (moved to tests/tools/)

Folders that KEEP (justified by file count + distinct concepts):
- `raghub/cli/` (8 commands)
- `raghub/tools/` (10 agent tools — was agent/tools/)
- `raghub/interfaces/` (17 protocols)
- `raghub/api/` (9 routes after rag.py extracted)
- `raghub/storage/` (8 SQLite files)
- `raghub/evaluation/` (5 evaluator files)
- `raghub/documents/parsers/` (9 parsers)
- `raghub/retrieval/` (6 files) + `raghub/retrieval/rerankers/` (5) + `raghub/retrieval/transforms/` (6)
- `raghub/services/` (3 services) + `raghub/services/application/` (3 app coordinators)

## Stage B — Semi-private name conversions (`_foo` → public or `__foo`)

- [x] B.1 `raghub/config.py`: `_env_bool` → `env_bool`, `_csv_to_transforms` → `csv_to_transforms`, `_TRUTHY` → `TRUTHY`, `_TRANSFORM_NAMES` → `TRANSFORM_NAMES`
- [x] B.2 `raghub/telemetry/langfuse.py`: docstring `_safe` → `try_call`
- [x] B.3 `raghub/agent/agent.py`: any `_safe_call` references
- [x] B.4 `raghub/agent/tools/*.py`: any `_foo` references
- [x] B.5 `raghub/knowledge/structures/graphrag.py`: `_running_loop_present` → `running_loop_present`, `_run_in_thread` → `run_in_thread`
- [x] B.6 `raghub/vectorstore/zvec.py`: confirm `native_filter` (was `_native_filter`)
- [x] B.7 Sweep: `grep -rn "^\s*\bself\._[a-z]\|^\s*def _[a-z]\|^\s*class _[A-Z]" raghub/ --include="*.py"` → 0 results
- [x] B.8 Any remaining `_foo` that needs true privacy → convert to `__foo` (name-mangled)

## Stage C — Drop `__getattr__` cycle-breakers + lazy imports

- [x] C.1 `raghub/__init__.py` — drop `__getattr__`
- [x] C.2 `raghub/agent/__init__.py` — drop `__getattr__`
- [x] C.3 `raghub/storage/__init__.py` — drop `__getattr__`
- [x] C.4 `raghub/ingestion/__init__.py` — delete (now flat file)
- [x] C.5 `raghub/pipelines/__init__.py` — delete (now flat file)
- [x] C.6 `raghub/pipelines/rag/__init__.py` — delete
- [x] C.7 `raghub/observability/tracing.py` — drop lazy `tracing_exporters` import
- [x] C.8 `raghub/retrieval/rerankers/__init__.py` — drop lazy reranker imports
- [x] C.9 `raghub/cli/__init__.py` — keep (just re-exports `app`, `main`)
- [x] C.10 Sweep: `grep -rn "def __getattr__" raghub/ --include="*.py"` → 0 results
- [x] C.11 Sweep: `grep -rEn "^\s+(from raghub\.[a-z]+|import raghub\.[a-z]+)" raghub/ --include="*.py"` → 0 lazy imports inside functions

## Stage D — Module-level singletons → class instances

- [x] D.1 `raghub/api/app.py`: `app_singleton` → `AppFactory.create_app()` class method
- [x] D.2 `raghub/observability/metrics.py`: `MetricsRegistry.instance` → instance held by `RAG`
- [x] D.3 Sweep: `grep -rEn "^[A-Z_]+\s*=\s*(\[\]|{})" raghub/ --include="*.py"` → 0 results
- [x] D.4 Sweep: any module-level mutable globals → instance state

## Stage E — Type annotations

- [x] E.1 Add `from __future__ import annotations` to every file (if not present)
- [x] E.2 Sweep: every public method has typed parameters + return type

## Stage F — Docstrings (Google style)

- [x] F.1 Every module has a module docstring
- [x] F.2 Every public class has a class docstring (one-liner + extended)
- [x] F.3 Every public method has a method docstring (Args/Returns/Raises)
- [x] F.4 Sweep: `interrogate -vv raghub/` → ≥95%

## Stage G — Import order (stdlib / third-party / local)

- [x] G.1 Sweep every file's imports: standard library → third-party → local
- [x] G.2 Remove any `from module import *`
- [x] G.3 No lazy imports in function bodies

## Stage H — Naming convention sweep

- [x] H.1 Classes: `CapWords`
- [x] H.2 Functions/variables: `snake_case`
- [x] H.3 Constants: `SCREAMING_SNAKE_CASE`
- [x] H.4 Modules: `snake_case`

## Stage I — try/except boundary audit

Only 12 sites should retain try/except:
- [x] I.1 `raghub/api/app.py` lifespan
- [x] I.2 `raghub/api/app.py` route handlers
- [x] I.3 `raghub/api/admin.py` route handlers
- [x] I.4 `raghub/api/preferences.py` route handlers
- [x] I.5 `raghub/pipeline.py` `IngestPipeline.run`, `QueryPipeline.run`
- [x] I.6 `raghub/pipeline.py` `AgenticQueryPipeline`
- [x] I.7 `raghub/services/workers.py` worker loop
- [x] I.8 `raghub/cli/main.py` CLI entry
- [x] I.9 `raghub/observability/metrics.py` `record_rerank_latency` (label validation)
- [x] I.10 `raghub/utils/retry.py` retry counter
- [x] I.11 `raghub/utils/execution.py` `DurationTimer`
- [x] I.12 `raghub/agent/agent.py` agent loop error isolation

Verify: `grep -rn "^\s\+try:" raghub/ --include="*.py"` → exactly 12 sites

## Stage J — Suppressions

- [x] J.1 Sweep: `grep -rn "# type: ignore\|# noqa\|# pylint:" raghub/ --include="*.py"` → 0 results
- [x] J.2 Fix any leftover suppression by proper typing

## Stage K — Long function audit

- [x] K.1 `raghub/pipeline.py` ~1500 lines: ACCEPTED (wiring file)
- [x] K.2 `raghub/observability.py` ~940 lines: ACCEPTED
- [x] K.3 `raghub/vectorstore.py` ~980 lines: ACCEPTED
- [x] K.4 `raghub/agent.py` ~700 lines: ACCEPTED
- [x] K.5 Sweep: no function > 80 lines

## Stage L — Comments

- [x] L.1 Strip redundant comments that just repeat code
- [x] L.2 Keep WHY comments
- [x] L.3 No TODO/FIXME/XXX/HACK: `grep -rn "# TODO\|# FIXME\|XXX\|HACK" raghub/ --include="*.py"` → 0 results

## Stage M — Drop `__all__` everywhere

- [x] M.1 Remove every `__all__ = [...]` from every module
- [x] M.2 Sweep: `grep -rn "^__all__\s*=" raghub/ --include="*.py"` → 0 results

## Stage N — Test rewrite (qualitative, breaks-the-impl)

- [x] N.1 Rewrite `tests/test_rag_facade.py` with qualitative tests that exercise edge cases
- [x] N.2 Rewrite `tests/test_pipelines_rag.py` with end-to-end pipeline tests
- [x] N.3 Rewrite `tests/test_services.py` with realistic service interaction tests
- [x] N.4 Rewrite `tests/test_conversation.py` with state-machine tests
- [x] N.5 Rewrite `tests/test_knowledge.py` with knowledge bundle integration tests
- [x] N.6 Rewrite `tests/test_vectorstore_*.py` with concurrency, error-path tests
- [x] N.7 Rewrite `tests/test_storage*.py` with migration and corruption tests
- [x] N.8 Rewrite `tests/test_ingestion.py` with full-pipeline tests
- [x] N.9 Rewrite `tests/test_evaluation.py` with benchmark integration
- [x] N.10 Rewrite `tests/test_legacy_services.py`, `tests/test_dynamic_application.py`
- [x] N.11 Rewrite `tests/test_config_validation.py` with edge-case env handling
- [x] N.12 Rewrite `tests/test_auth_*`, `tests/test_documents_*`, `tests/test_cli*`, `tests/test_api*`, `tests/test_agent*`
- [x] N.13 Move `tests/agent/tools/` → `tests/tools/`
- [x] N.14 Delete redundant test files (`tests/test_cli.py`, `tests/test_cli_commands.py`, `tests/test_cli_rate_limiter.py`, `tests/test_renamed_helpers.py`, `tests/test_schemas_and_module_smoke.py`)

## Stage P — Final deletions

- [x] P.1 Delete `raghub/exceptions/` (after A.1)
- [x] P.2 Delete `raghub/utils/` (after A.2)
- [x] P.3 Delete `raghub/structured/` (after A.5)
- [x] P.4 Delete `raghub/telemetry/` (after A.6)
- [x] P.5 Delete `raghub/generation/` (after A.5)
- [x] P.6 Delete `raghub/prompts/` (after A.3)
- [x] P.7 Delete `raghub/plugins/` (after A.4)
- [x] P.8 Delete `raghub/conversation/` (after A.7)
- [x] P.9 Delete `raghub/core/` (after A.8)
- [x] P.10 Delete `raghub/llm/` (after A.9)
- [x] P.11 Delete `raghub/embeddings/` (after A.10)
- [x] P.12 Delete `raghub/domain/` (after A.11)
- [x] P.13 Delete `raghub/models/` (after A.12)
- [x] P.14 Delete `raghub/repositories/` (after A.13)
- [x] P.15 Delete `raghub/auth/` + `services/auth_service.py` (after A.14)
- [x] P.16 Delete `raghub/converters/` + `raghub/documents/` top 4 files (after A.15)
- [x] P.17 Delete `raghub/vectorstore/` (after A.16)
- [x] P.18 Delete `raghub/ingestion/` + `raghub/ingestion/chunkers/` (after A.17)
- [x] P.19 Delete `raghub/knowledge/` + `raghub/knowledge/structures/` (after A.18)
- [x] P.20 Delete `raghub/pipelines/` + `raghub/pipelines/rag/` (after A.19)
- [x] P.21 Delete `services/auth_service.py`, `services/query_service.py`, `services/application/auth.py`
- [x] P.22 Delete `raghub/api/rag.py`, `raghub/api/defaults.py`, `raghub/api/async_runner.py` (folded into rag.py and utils.py)
- [x] P.23 Delete `raghub/agent/` top + `raghub/agent/prompts.py` (after A.28); rename `agent/tools/` to `tools/`

## Stage Q — Final grep verification

- [x] Q.1 `grep -rn "^\s*\bself\._[a-z]" raghub/ --include="*.py"` → 0
- [x] Q.2 `grep -rn "^\s*def _[a-z]" raghub/ --include="*.py"` → 0
- [x] Q.3 `grep -rn "# type: ignore\|# noqa\|# pylint:" raghub/ --include="*.py"` → 0
- [x] Q.4 `grep -rn "def __getattr__" raghub/ --include="*.py"` → 0
- [x] Q.5 `grep -rn "^__all__\s*=" raghub/ --include="*.py"` → 0
- [x] Q.6 `grep -rn "^\s\+try:" raghub/ --include="*.py"` → exactly 12 (Stage I)
- [x] Q.7 `grep -rn "^_\(app\|module\|[a-z]\+\)\s*=" raghub/ --include="*.py"` → 0
- [x] Q.8 `grep -rEn "^\s+(from raghub\.[a-z]+|import raghub\.[a-z]+)" raghub/ --include="*.py"` → 0 (no lazy imports inside functions)
- [x] Q.9 `ruff check raghub/ tests/` → 0 errors
- [x] Q.10 `mypy --strict raghub/` → 0 errors
- [x] Q.11 `pytest -q --no-cov tests/` → all pass
- [x] Q.12 `pytest --cov=raghub --cov-fail-under=85` → passes

## Final state — top-level layout

```
raghub/
├── __init__.py
├── config.py              # Settings
├── exceptions.py
├── utils.py               # retry + DurationTimer + maybe_await
├── auth.py                # SqliteUserStore + RBAC + AuthService
├── core.py                # Container + RBAC service + doc state
├── conversation.py
├── repositories.py
├── domain.py
├── models.py
├── embeddings.py
├── llm.py
├── generation.py          # DefaultGenerator + InstructorStructuredOutputProvider
├── knowledge.py           # manifest + OKF + repository + KnowledgeIndex + RaptorIndex + GraphRagIndex
├── vectorstore.py
├── observability.py       # logging + metrics + redact + noop + tracing + langfuse
├── ingestion.py
├── documents.py           # chunker + lifecycle + validation + versioning + converters
├── pipeline.py            # cache + agentic + Ingest + Query + ConversationRouter + PipelineResultBuilder
├── rag.py                 # public RAG facade (absorbs defaults + async_runner)
├── agent.py
├── plugins.py
├── prompts.py
├── cli/                   # 8 commands
├── tools/                 # 10 agent tools (was agent/tools/)
├── interfaces/            # 17 protocols
├── api/                   # 9 routes (after rag.py extracted)
├── storage/               # 8 SQLite files
├── evaluation/            # 5 evaluator files
├── documents/parsers/     # 9 parsers
├── retrieval/             # 6 files (pipeline, colbert, fusion, context, reranker, search)
│   ├── rerankers/         # 5 files
│   └── transforms/        # 6 files
└── services/              # 3 services (document, health, workers)
    └── application/       # 3 app coordinators (facade, shutdown, preferences)
```

22 flat files + 12 folders = 34 top-level paths (down from 50).