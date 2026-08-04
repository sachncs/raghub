# Tier 10 — Architectural splits (Items 54-60)

`AGENTS.md` rule C2 — no god-modules. Several files exceed the
500-LOC threshold. Tier 10 splits them into focused packages.

Each item is one PR; each PR is independently shippable.

---

## Item 54 — Split `raghub/services/__init__.py`

- **File(s)**: `raghub/services/{container,auth,workers,feedback}.py` (new), `raghub/services/__init__.py`
- **Change**: Extract `RagContainer` into `container.py`; `AuthService` into `auth.py`; `Synchronous`, `ThreadPool`, `MemoryQueue`, `Worker`-related logic into `workers.py`; `Feedback`-related wiring into `feedback.py`. `__init__.py` re-exports.
- **Test**: All `test_services.py` tests pass.
- **Acceptance criteria**:
  - C2 — each file ≤ 350 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l raghub/services/{container,auth,workers,feedback}.py` ≤ 350 each.
  - All public symbols still importable from `raghub.services`.

---

## Item 55 — Split `raghub/retrieval/__init__.py`

- **File(s)**: `raghub/retrieval/{rerank,transforms,fusion,search,pipeline}.py` (new), `raghub/retrieval/__init__.py`
- **Change**: 5 modules covering the 20 classes.
- **Test**: All `test_retrieval.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 400 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l raghub/retrieval/{rerank,transforms,fusion,search,pipeline}.py` ≤ 400 each.

---

## Item 56 — Split `raghub/eval/__init__.py`

- **File(s)**: `raghub/eval/{metrics,judge,gate}.py` and `raghub/eval/benchmarks/{finance,frames}.py` (new), `raghub/eval/__init__.py`
- **Change**: 5 modules.
- **Test**: All `test_evaluation.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 350 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of each ≤ 350.

---

## Item 57 — Move `pipeline.py` into `pipeline/` package

- **File(s)**: `raghub/pipeline/{ingest,query,agent,cache}.py` (new), `raghub/pipeline/__init__.py`
- **Change**: 4 modules.
- **Test**: All `test_pipeline.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 400 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of each ≤ 400.

---

## Item 58 — Move `lifecycle/__init__.py` into `lifecycle/` package

- **File(s)**: `raghub/lifecycle/{state,scanner,converters}.py` (new), `raghub/lifecycle/__init__.py`
- **Change**: 3 modules.
- **Test**: All `test_lifecycle.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 350 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of each ≤ 350.

---

## Item 59 — Move `knowledge.py` into `knowledge/` package

- **File(s)**: `raghub/knowledge/{okf,manifest,raptor,graph}.py` (new), `raghub/knowledge/__init__.py`
- **Change**: 4 modules.
- **Test**: All `test_knowledge.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 350 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of each ≤ 350.

---

## Item 60 — Move `telemetry.py` into `telemetry/` package

- **File(s)**: `raghub/telemetry/{logger,metrics,redaction}.py` (new), `raghub/telemetry/__init__.py`
- **Change**: 3 modules.
- **Test**: All `test_telemetry.py` tests pass.
- **Acceptance criteria**:
  - C2 — each ≤ 350 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of each ≤ 350.

---

## Tier 10 acceptance gate

- `wc -l raghub/{services,retrieval,eval,pipeline,lifecycle,knowledge,telemetry}/*.py` — every file ≤ 400 LOC (350 for the smaller packages).
- `mypy --strict raghub/` passes.
- `pytest -q --no-cov` passes.
