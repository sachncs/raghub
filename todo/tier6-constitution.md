# Tier 6 — Constitution cleanup (Items 31-36)

`AGENTS.md` rules R2 (no single-underscore), N1 (no forbidden names
like `Mixin` / `Svc`), C1 (functions ≤ 40 LOC; `__init__` ≤ 30 LOC).

The v0.7.x → v0.8.0 work shipped 27 `__register_*` methods in
`raghub/api.py`, a `Mixin` class in `services/`, and a 100-LOC
`RAG.__init__`. Tier 6 fixes these.

---

## Item 31 — Rename `__register_*` → `_register_*` in `raghub/api.py`

- **File(s)**: `raghub/api.py`
- **Change**: `s/__register_/_register_/g` across all 27 sites. Update internal call sites.
- **Test**: All `test_api.py` tests pass.
- **Acceptance criteria**:
  - R2 — single-underscore is forbidden; these get renamed. (Strict reading: `__<one-word>` is required, so `__register_login` was a violation regardless of single-underscore.)
  - T3 — `ruff check` clean.
- **Success criteria**:
  - `grep -rn "def __register" raghub/` returns empty.
  - All `test_api.py` tests pass.

---

## Item 32 — Remove `Mixin` class from `services/__init__.py`

- **File(s)**: `raghub/services/__init__.py`, all importers
- **Change**: Inline `Mixin.log` and `Mixin.emit_metric` into the service classes (`DocumentSvc`, `Health`, `Query`, `Preference`).
- **Test**: All `test_services.py` tests pass.
- **Acceptance criteria**:
  - N1 — `Mixin` is on the forbidden name list; removed.
  - T3 — pass.
- **Success criteria**:
  - `grep -n "class Mixin" raghub/services/` returns empty.
  - `DocumentSvc`, `Health`, `Query`, `Preference` each have `log()` and `emit_metric()` directly.

---

## Item 33 — Rename `DocumentSvc` → `DocumentService`

- **File(s)**: `raghub/services/__init__.py`, `raghub/__init__.py`, all importers
- **Change**: Rename class; update imports.
- **Test**: All tests pass.
- **Acceptance criteria**:
  - N1 — `Svc` suffix removed.
  - T3 — pass.
- **Success criteria**:
  - `grep -rn "DocumentSvc" raghub/` returns empty.
  - `from raghub.services import DocumentService` works.

---

## Item 34 — Decompose `RAG.__init__` into helpers

- **File(s)**: `raghub/rag.py`
- **Change**: Extract `_wire_components`, `_wire_query_pipeline`, `_wire_ingest_pipeline`, `_wire_telemetry`. `__init__` becomes < 30 LOC.
- **Test**: All `test_rag_facade.py` tests pass.
- **Acceptance criteria**:
  - C1 — `__init__` ≤ 30 LOC.
  - T1, T3 — pass.
- **Success criteria**:
  - `wc -l` of `__init__` < 30.
  - `RAG()` constructs identically (same wiring).

---

## Item 35 — Decompose `Preference.query_with_flags`

- **File(s)**: `raghub/services/__init__.py`
- **Change**: Extract `_resolve_flags`, `_resolve_user_prefs`, `_invoke_query`. `query_with_flags` becomes < 30 LOC.
- **Test**: All `test_services.py` tests pass.
- **Acceptance criteria**:
  - C1 — function ≤ 40 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l` of `query_with_flags` < 30.

---

## Item 36 — Move `RAG` class into `raghub/rag/facade.py`

- **File(s)**: `raghub/rag.py`, `raghub/rag/__init__.py` (new), `raghub/rag/facade.py` (new)
- **Change**: Convert `raghub/rag.py` (1 file) into a package with `RAG` class in `facade.py` and re-exports in `__init__.py`.
- **Test**: `from raghub import RAG` still works.
- **Acceptance criteria**:
  - C2 — no god-module; each file ≤ 800 LOC.
  - T3 — pass.
- **Success criteria**:
  - `wc -l raghub/rag/facade.py < 800`.
  - `from raghub import RAG` returns the class.

---

## Tier 6 acceptance gate

- `grep -rn "class Mixin\|DocumentSvc\|def __register" raghub/` returns empty.
- `wc -l raghub/rag/facade.py < 800`.
- `RAG.__init__` < 30 LOC.
- `Preference.query_with_flags` < 30 LOC.
- `mypy --strict raghub/` passes.
