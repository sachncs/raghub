# Tier 7 — Type discipline (Items 37-40)

`AGENTS.md` rule R10 — "No `Any` outside `metadata`; `metadata` is the
only `Any` slot." The codebase has 592 `Any` annotations. Tier 7
removes the worst offenders in the service / tool / retrieval
layers and removes the one `# type: ignore`.

---

## Item 37 — `Any` cleanup in `services/__init__.py`

- **File(s)**: `raghub/services/__init__.py`
- **Change**: Replace `Any` annotations on `RagContainer` fields (`settings: Any`, `logger: Any`, `metrics: object`, etc.) with concrete Protocol / dataclass references. Replace `getattr(..., None)` returns with Protocol types.
- **Test**: `mypy --strict raghub/services/` clean.
- **Acceptance criteria**:
  - R10 — `Any` only in `metadata: dict[str, Any]`.
  - T1, T3 — pass.
- **Success criteria**:
  - `grep ": Any" raghub/services/__init__.py | wc -l < 10`.
  - `mypy --strict raghub/services/` returns zero errors.

---

## Item 38 — `Any` cleanup in `tools/__init__.py`

- **File(s)**: `raghub/tools/__init__.py`
- **Change**: Replace `Any` in `ToolContext`, `ToolResult`, `HybridSearch.execute`, `KeywordSearch.execute`, `GraphSearch.execute`, etc.
- **Test**: `mypy --strict raghub/tools/` clean.
- **Acceptance criteria**:
  - R10 — `Any` only in `metadata`.
  - T1, T3 — pass.
- **Success criteria**:
  - `grep ": Any" raghub/tools/__init__.py | wc -l < 5`.

---

## Item 39 — Remove `# type: ignore` from `pgvector.py`

- **File(s)**: `raghub/store/pgvector.py`, `pyproject.toml`
- **Change**: Move `asyncpg` from `[pgvector]` extra to `dependencies`. Remove the `# type: ignore[import-not-found]` comment.
- **Test**: `pip install raghub` (no extra) succeeds; `from raghub.store.pgvector import PgVectorStore` works.
- **Acceptance criteria**:
  - R1 — no `# type: ignore`.
  - T1, T3 — pass.
- **Success criteria**:
  - `grep "type: ignore" raghub/store/pgvector.py` returns empty.
  - `pip install -e .` (no extras) succeeds on a fresh venv.

---

## Item 40 — `Any` cleanup in `retrieval/__init__.py`

- **File(s)**: `raghub/retrieval/__init__.py`
- **Change**: Replace `Any` in the public Protocol-conformant classes (`Retrieval`, `Rerank`, `Hyde`, `MultiQuery`, `Decompose`, `StepBack`, `Compose`, `Search`, `SearchFilters`, `Fusion`).
- **Test**: `mypy --strict raghub/retrieval/` clean.
- **Acceptance criteria**:
  - R10 — `Any` only in `metadata`.
  - T1, T3 — pass.
- **Success criteria**:
  - `grep ": Any" raghub/retrieval/__init__.py | wc -l < 30`.

---

## Tier 7 acceptance gate

- `grep -rn ": Any" raghub/services raghub/tools raghub/retrieval raghub/store/pgvector.py` count below thresholds.
- `grep "type: ignore" raghub/` returns empty.
- `mypy --strict raghub/` passes.
