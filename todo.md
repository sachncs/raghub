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
- [ ] Delete all `_*` private files in every package
- [ ] Remove `getattr(self.x, "method", None)` probes
- [ ] Remove `ConversationHost`/`SyncHost` Protocols
- [ ] Remove `ConversationMixin`/`SyncMixin` empty bodies

## Phase 4: Objectify dict carriers
- [ ] Convert `Cache` tuple to `CacheKey` dataclass
- [ ] Convert `Pipeline.outputs` dict to `PipelineOutputs` dataclass
- [ ] Convert `PipelineCtx.metadata` dict to `IngestMeta` dataclass
- [ ] Convert `health()` dict to `HealthReport` dataclass
- [ ] Convert other TypedDicts to dataclasses

## Phase 5: Replace ABCs with concrete polymorphism
- [ ] Drop all ABCs (Embedder, Generator, File, Store, KnowledgeIndex, Tool)
- [ ] Drop all Protocols (consumer updates complete)
- [ ] Convert concrete subclasses to use registry

## Phase 6: Consolidate remaining structural oddities
- [ ] Unify Job state machines
- [ ] Unify `Fusion` implementations
- [ ] Unify `DurationTimer` (currently duplicated)
- [ ] Replace `capture()` inlining
- [ ] Convert `Settings` Pydantic to frozen dataclass
- [ ] Fold `RAG` mixins into the class
- [ ] Convert `services/diagnostics.py` free functions to methods

## Phase 7: Verify
- [ ] `grep -rn "ABC" raghub/` → zero hits
- [ ] `grep -rn "Protocol" raghub/` → zero hits
- [ ] `grep -rn "from pydantic" raghub/` → zero hits
- [ ] `grep -rn "^[a-z_]*_.*\.py" raghub/` (leading underscore modules) → zero hits
- [ ] `grep -rn "^_.*\.py" raghub/` (any leading underscore modules) → zero hits
- [ ] `grep -rn "isinstance(.*Plugin" raghub/` etc. → zero hits
- [ ] Run `uv run poe syntax` or equivalent
- [ ] Run `uv run poe pyright`
- [ ] Run full test suite
