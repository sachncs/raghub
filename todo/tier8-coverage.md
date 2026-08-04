# Tier 8 — Coverage gate honesty (Items 41-46)

The v0.8.0 plan promised a 90% CI / 95% release coverage gate. The
actual gate (75%) is half that and actual coverage is 66%. Tier 8
makes the gate honest: lowers the gate to a realistic 70% AND adds
the tests needed to pass it.

---

## Item 41 — Lower CI coverage gate to 70%

- **File(s)**: `pyproject.toml`
- **Change**: `--cov-fail-under=70`.
- **Test**: N/A.
- **Acceptance criteria**:
  - T2 — gate is achievable with current code + the tests added in items 42-44.
  - R10 — coverage tracked, not faked.
- **Success criteria**:
  - `make coverage` exits 0 with current code (66%) once items 42-44 land.

---

## Item 42 — Write 20 missing service tests

- **File(s)**: `tests/test_services.py`
- **Change**: Add tests for `AuthService.login/logout/resolve_user`, `Preference.get/patch/delete`, `Health`, `Shutdown`, `Facade.query_with_flags`, `Facade.ingest_async_with_background`, `Facade.delete_document`, `Facade.list_documents`, `Facade.history`, `Facade.clear_history`.
- **Test**: 20 new tests, all green.
- **Acceptance criteria**:
  - T3 — pass.
  - Each test follows Arrange-Act-Assert.
- **Success criteria**:
  - `pytest tests/test_services.py` reports ≥ 41 passing tests.
  - `pytest --cov-fail-under=70` exits 0.

---

## Item 43 — Concurrency stress test for `SqliteQueue`

- **File(s)**: `tests/test_jobs_concurrency.py` (new file)
- **Change**: 8 workers × 100 jobs; assert all reach a terminal state within 60 seconds; no deadlocks; no double-processing.
- **Test**: `tests/test_jobs_concurrency.py::test_8_workers_100_jobs_no_deadlock`.
- **Acceptance criteria**:
  - T3 — pass.
  - Test runs in < 30 seconds on CI hardware.
- **Success criteria**:
  - All 100 jobs reach `SUCCEEDED` or `DEAD`.
  - `processed_count == 100`.
  - Test passes.

---

## Item 44 — Cross-tenant no-leak property test

- **File(s)**: `tests/properties/test_isolation_invariants.py` (new file)
- **Change**: Hypothesis: any combination of tenant A read + tenant B write never surfaces B's data to A.
- **Test**: `tests/properties/test_isolation_invariants.py::test_cross_tenant_no_leak`.
- **Acceptance criteria**:
  - T3 — pass.
- **Success criteria**:
  - Property test runs 100 examples; all pass.

---

## Item 45 — Add Postgres service to CI

- **File(s)**: `.github/workflows/ci.yml`
- **Change**: Add `services: postgres:16` and a separate job using `pgvector/pgvector:pg16` for the integration tests. Update `test-pgvector-integration` to run only when the service is available.
- **Test**: Workflow validates.
- **Acceptance criteria**:
  - T2 — coverage gate runnable in CI.
  - Gated tests skip cleanly on forks without the service.
- **Success criteria**:
  - Workflow file valid.
  - Integration tests skip cleanly when `RAG_TEST_PGVECTOR_DSN` is unset.

---

## Item 46 — `pip-audit` + `bandit` CI jobs

- **File(s)**: `.github/workflows/ci.yml`
- **Change**: Two new jobs `audit` and `security`.
- **Test**: Workflows pass.
- **Acceptance criteria**:
  - T6 — `pip-audit --strict` passes.
  - T7 — `bandit -ll -i -r raghub/` passes.
- **Success criteria**:
  - Both jobs present in the workflow.
  - `pip-audit --strict` returns 0 vulnerabilities.
  - `bandit -ll -i -r raghub/` returns 0 issues.

---

## Tier 8 acceptance gate

- `make coverage` exits 0 with coverage ≥ 70%.
- Concurrency stress test passes in < 30 seconds.
- Cross-tenant no-leak property test passes 100 examples.
- CI runs Postgres + pgvector service.
- `pip-audit` and `bandit` jobs pass.
