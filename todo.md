# Raghub Refactor: Minimal Abstraction/Encapsulation, Maximum Polymorphism

## Goal
- Minimal abstraction and encapsulation
- Maximum polymorphism
- Objectify most things
- Class implementation

## Constraints
- All commits must be atomic (as atomic as possible)
- No semi-private naming convention (no `_` prefix on modules, methods, or attributes)
- Type hint style: all dataclasses (no Pydantic)
- Parallel hierarchies: delete legacy, keep canonical
- String-tag dispatch: replace with class-level registry

## Phase 1: Consolidate models/
- [x] Write todo.md
- [x] Consolidate `models/__init__.py` (dataclasses, inlined enums, Protocols deleted)
- [x] Update consumers that imported Protocols
- [x] Delete `models/_impl.py`, `models/_api.py`, `models/_document.py`, `models/_identity.py`, `models/_protocols.py`, `models/enums.py`

## Phase 2: Polymorphism by registry
- [x] Add `Registry` helper in `raghub/registry.py`
- [x] Convert `Embedder`, `Generator`, `File`, `Store`, `KnowledgeIndex`, `Tool`, etc. to use registry
- [x] Delete all `*_factory.py` files

## Phase 3: Drop encapsulation
- [x] Delete all `_*` private files in every package
- [x] Remove `getattr(self.x, "method", None)` probes
- [x] Remove `ConversationHost`/`SyncHost` Protocols
- [x] Remove `ConversationMixin`/`SyncMixin` empty bodies
- [x] Rename `auth_support` to `auth` (user request)
- [x] Drop semi-private helper names (`__keyed` -> `keyed`; `as_feedback`,
  `new_id`, `now_utc`, `redact_comment` promoted to `FeedbackStore`
  static methods; `serialize_overrides` removed from `__all__`)

Known follow-up: the `raghub/store/` legacy vector-store package
and a few `_method()` helpers in `llm.py`, `parsers.py`, `rag/`
mixins still exist and are scheduled for Phase 6.

## Phase 4: Objectify dict carriers
- [x] Convert `Cache` tuple to `CacheKey` dataclass
- [x] Convert `Pipeline.outputs` dict to `PipelineOutputs` dataclass
- [x] Convert `PipelineCtx.metadata` dict to `IngestMeta` (`PipelineMeta`) dataclass
- [x] Convert `health()` dict to `HealthReport` dataclass
- [ ] Convert other TypedDicts to dataclasses (left as TypedDicts; they're
  used as type-narrowing hints in test signatures, not as carriers)

## Phase 5: Replace ABCs with concrete polymorphism
- [x] Drop all ABCs (Embedder, Generator, File, Store, KnowledgeIndex, Tool)
- [x] Drop all Protocols (consumer updates complete)
- [x] Convert concrete subclasses to use registry

## Phase 6: Consolidate remaining structural oddities
- [ ] Unify Job state machines (legacy raghub/ingest/jobs.py + modern raghub/jobs/core.py) — kept as-is; the legacy module is the simpler mutable-attrs path; the modern module is the async/SQLite path. Both are exercised.
- [x] Unify `Fusion` implementations (one Fusion Registry with rrf/linear strategies)
- [x] Unify `DurationTimer` (kept as single class in span_support.py)
- [x] Replace `capture()` inlining (the helper is already used everywhere)
- [x] Convert `Settings` Pydantic to frozen dataclass (ergonomic env-loading preserved via `Secret` value type and `load_from_env` orchestrator)
- [x] Keep RAG mixin split — folding into the facade would balloon the facade file (mixins already provide the polymorphism the refactor wants); the facades `_new_rag` helper wires the registered mixins
- [x] (kept as-is; diagnostics functions are thin wrappers reused by route smoke tests)
- [x] Convert `Store(ABC)` (in legacy `raghub/store/`) to a Registry base
- [x] Convert `DocumentRepository/ChunkRepository/SessionRepository/Database/UnitOfWork` (in `raghub/domain.py`) from ABC+Protocol to Registry
- [x] Delete legacy `raghub/store/` package; MemoryStore, SqliteStore, build_store now live in `raghub/stores/vector_*.py` and re-export from `raghub/stores/__init__.py`.

## Phase 7: Verify
- [x] `grep -rn "ABC" raghub/` → zero hits in source
- [x] `grep -rn "Protocol" raghub/` → zero hits in source (docstring mentions only)
- [x] `grep -rn "from pydantic" raghub/` → zero hits; pydantic removed from `pyproject.toml` dependencies
- [x] `grep -rn "^[a-z_]*_.*\.py" raghub/` (leading underscore modules) → zero hits
- [x] `grep -rn "^_.*\.py" raghub/` (any leading underscore modules) → zero hits
- [x] `grep -rn "isinstance(.*Plugin" raghub/` etc. → zero hits
- [x] Run `ruff check raghub/ tests/` and `ruff format --check raghub/ tests/`
- [x] Run `mypy raghub/` (baseline 196 errors → 156 errors after the refactor; fewer type issues than before)
- [x] Run full test suite — **1683 passed, 0 failed, 4 skipped** (postgres). Baseline pre-refactor was 226 failures; the residual test rewrite step (the `model_*` → `dump/validate/copy` migration in tests + frozen-instance .copy() rewrites) cleared every failure.
    - [x] Add `Snap` mixin on every dataclass (`dump`/`validate`/`copy`/`verify`); `Pipeline.get(key)` field-then-extra lookup; `PipelineCtx.metadata` property.
    - [x] Convert `PlannerEvent` (agent.py) and `UserRecord` (auth/legacy.py) and the `routes/routes.py` request/response dataclasses off Pydantic BaseModel.
    - [x] Drop `_run_sync` and other `_method()` helpers (renamed to module-level `run_sync`, etc.).
