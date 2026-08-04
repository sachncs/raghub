# Tier 14 — Cleanup + version bump + CHANGELOG (Items 72-78)

Final tier: re-export new symbols, pin `asyncpg` to core, drop the
`[pgvector]` extra, remove `TODO`s, bump version, update CHANGELOG.

`AGENTS.md` rules:
- O4 — every public symbol in `docs/reference/public-api.md`.
- O5 — `CHANGELOG.md` updated per release.

---

## Item 72 — `__init__.py` re-exports dropped symbols

- **File(s)**: `raghub/__init__.py`
- **Change**: Add top-level re-exports: `Job`, `JobStatus`, `SqliteQueue`, `PersistentQueue`, `Worker`, `PgVectorStore`, `Feedback`, `FeedbackStore`, `Bm25BoostScorer`, `VectorDownWeightScorer`, `ArchiveManifest`, `ArchiveStore`, `LocalArchiveStore`, `IsolationStrategy`, `TenantContext`, `TenantResolver`, `CompositeTenantResolver`, `JwtClaimTenantResolver`, `HeaderTenantResolver`, `NoTenantResolver`, `TenantRegistry`, `TenantSecretCipher`, `RowLevel`, `SchemaPerTenant`, `DatabasePerTenant`, `QueueSaturatedError`, `JobStateError`.
- **Test**: `tests/test_imports.py` (new): `from raghub import X` for each new symbol.
- **Acceptance criteria**:
  - O4 — every public symbol reachable from `raghub`.
  - T3 — pass.
- **Success criteria**:
  - `from raghub import Job, SqliteQueue, PgVectorStore, Feedback, ArchiveManifest, IsolationStrategy` works.

---

## Item 73 — Pin `asyncpg` to core deps

- **File(s)**: `pyproject.toml`
- **Change**: Move `asyncpg>=0.27,<1` from `[pgvector]` extra to `dependencies`.
- **Test**: `pip install raghub` (no extras) succeeds.
- **Acceptance criteria**:
  - R1 — no `# type: ignore` needed for `asyncpg`.
  - T6 — `pip-audit --strict` passes.
- **Success criteria**:
  - `grep -A5 "^\[project\]" pyproject.toml | grep asyncpg` matches.

---

## Item 74 — Drop `[pgvector]` extra

- **File(s)**: `pyproject.toml`
- **Change**: Remove the `[pgvector]` extra; ensure `[all]` is the only meta-extra.
- **Test**: `pip install raghub[pgvector]` exits with a clear "no such extra" error.
- **Acceptance criteria**:
  - T6 — pass.
- **Success criteria**:
  - `grep "^\[project.optional-dependencies\]" -A 30 pyproject.toml | grep -c pgvector == 0` (no occurrences outside `[all]`).

---

## Item 75 — Add `[all]` extra only (verify)

- **File(s)**: `pyproject.toml`
- **Change**: Verify `[all]` lists `marker-pdf`, `pypdf`, etc. Update if needed.
- **Test**: N/A.
- **Acceptance criteria**:
  - T6 — pass.
- **Success criteria**:
  - `[all]` covers every shipped adapter.

---

## Item 76 — Remove `TODO: …` comments from code

- **File(s)**: `raghub/`
- **Change**: `grep -rn "TODO" raghub/` returns empty.
- **Test**: N/A.
- **Acceptance criteria**:
  - T3 — pass.
- **Success criteria**:
  - `grep -rn "TODO" raghub/ | wc -l == 0`.

---

## Item 77 — Bump version to `0.9.0`

- **File(s)**: `pyproject.toml`
- **Change**: `version = "0.9.0"`.
- **Test**: N/A.
- **Acceptance criteria**:
  - N4 — `0.9.0` matches `MAJOR.MINOR.PATCH` pattern.
- **Success criteria**:
  - `grep "version" pyproject.toml | head -1` shows `0.9.0`.

---

## Item 78 — Update `CHANGELOG.md` with v0.9.0 entry

- **File(s)**: `CHANGELOG.md`
- **Change**: New `## [0.9.0] - <date>` block summarising what was actually fixed (Tiers 1-5), the constitution cleanup (Tier 6), the type discipline (Tier 7), the coverage gate (Tier 8), the docs (Tiers 9, 12, 13), the splits (Tier 10), the ADRs (Tier 11), and the cleanup (Tier 14).
- **Test**: N/A.
- **Acceptance criteria**:
  - O5 — `CHANGELOG.md` updated per release.
- **Success criteria**:
  - Entry exists at the top of `CHANGELOG.md` (after `[Unreleased]`).

---

## Tier 14 acceptance gate

- `from raghub import Job, SqliteQueue, PgVectorStore, Feedback, ArchiveManifest, IsolationStrategy` works.
- `grep "TODO" raghub/` returns empty.
- `pyproject.toml` version is `0.9.0`.
- `CHANGELOG.md` has a `[0.9.0]` entry.
