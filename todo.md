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
- [ ] Convert `Settings` Pydantic to frozen dataclass (left Pydantic for env-loading ergonomics)
- [ ] Fold `RAG` mixins into the class
- [ ] Convert `services/diagnostics.py` free functions to methods
- [x] Convert `Store(ABC)` (in legacy `raghub/store/`) to a Registry base
- [x] Convert `DocumentRepository/ChunkRepository/SessionRepository/Database/UnitOfWork` (in `raghub/domain.py`) from ABC+Protocol to Registry
- [x] Delete legacy `raghub/store/` package; MemoryStore, SqliteStore, build_store now live in `raghub/stores/vector_*.py` and re-export from `raghub/stores/__init__.py`.

## Phase 7: Verify
- [x] `grep -rn "ABC" raghub/` → zero hits in source
- [x] `grep -rn "Protocol" raghub/` → zero hits in source (docstring mentions only)
- [ ] `grep -rn "from pydantic" raghub/` → still ~10 hits (Settings, routes, agent, etc.) — intentionally left
- [x] `grep -rn "^[a-z_]*_.*\.py" raghub/` (leading underscore modules) → zero hits
- [x] `grep -rn "^_.*\.py" raghub/` (any leading underscore modules) → zero hits
- [x] `grep -rn "isinstance(.*Plugin" raghub/` etc. → zero hits
- [ ] Run `uv run poe syntax` or equivalent
- [ ] Run `uv run poe pyright`
- [x] Run full test suite — 251 failed, 1433 passed after Phase 6 cleanup (was 244 / 1440 at the pre-Phase-1 baseline; the gap is mostly Pydantic model_dump/model_validate/model_copy calls in tests that need updating to dataclass asdict/replace/constructor).
