# TODO · RAGHub

> **Audit date:** 2026-07-02 · **Version:** 0.3.2 (unreleased)
> Based on a full repository audit covering 157 source files, 1,153 tests, Git state, CI/CD, docs, and security posture.
> Previous TODO (2026-06-30) items reviewed; completed items removed, remaining items re-triaged.

---

## High Priority

### Build & Release

- [ ] **Create & push missing tags** — Commits `v0.3.1` and `v0.3.2` exist but no tags are pushed to origin (only `v0.1.0`, `v0.1.0-alpha`, `v0.2.0-beta` are remote). Run `git tag v0.3.1 <sha> && git tag v0.3.2 <sha> && git push --tags`.
- [ ] **Fix `build_backend.py` version** — Hardcodes `VERSION = "1.0.0"` while `pyproject.toml` is `0.3.1`. Should track the package version or be derived from it.
- [ ] **Remove `build_backend.py` if unused** — The PEP 517 build backend exists alongside setuptools in `pyproject.toml`. If not consumed downstream, remove to avoid confusion.

### Type System & Linting

- [ ] **Fix 6 mypy errors** — Remaining issues:
  - 3× `import-untyped` for `yaml` — fix with `types-PyYAML` (already in `[dev]` deps; just need `pip install`) or inline `# type: ignore[import-untyped]`
  - 2× LSP violations in `raghub/vectorstore/qdrant.py:219,266` — `search()` and `hybrid_search()` parameter types incompatible with `VectorStore` protocol; widen parameter types or update the protocol
  - 1× attr-defined in `raghub/ingestion/service.py:259` — `DocumentRepository` ABC (domain/repositories.py) lacks `try_insert`; either add it to the abstract class or cast at call site
- [ ] **Fix 1 ruff error** — Unused import in non-test code (`F401`); run `ruff check --fix`.
- [ ] **Migrate Pydantic V2 `Config` class** — `raghub/config/settings.py:85` uses the deprecated nested `Config` class; replace with `model_config = ConfigDict(arbitrary_types_allowed=True)`.

### Testing

- [ ] **Verify actual coverage meets the 90% gate** — CI enforces `--cov-fail-under=90` but the previous audit showed 72%. If the CI passes today, confirm; otherwise lower the gate or add missing tests.
- [ ] **Run `mypy raghub/` in CI with `--strict`-equivalent flags** — Current CI runs bare `mypy raghub/`; align with `mypy.ini` to catch the 6 remaining errors.
- [ ] **Enable dependabot for `dev` dependencies** — Dependabot only watches `pip` (runtime deps) and `github-actions`; `dev` extras (mypy, ruff, pytest, hypothesis) are not tracked.

---

## Medium Priority

### Code Quality

- [ ] **Reduce `Any` annotations** — 30 files use `: Any`; heaviest in `observability/` (29 usages across `noop.py`, `redact.py`, `structlog_provider.py`) and `telemetry/langfuse.py` (15). Use `TypeVar`, `ParamSpec`, or concrete overloads where possible.
- [ ] **Resolve circular-import workarounds** — 11 files use `TYPE_CHECKING` or lazy `importlib`; concentrated in `raghub/services/` (4 files) and `__init__.py` re-export modules. Reorganise to break cycles at the package level.
- [ ] **Audit protocol conformance** — `raghub/interfaces/` defines 17 Protocols; verify every implementation fully conforms. Known violations: `QdrantVectorStore.search/hybrid_search` incompatible with `VectorStore`. Add `pytest`-based protocol conformance tests.
- [ ] **Add `try_insert` to `DocumentRepository` ABC** — `SqliteDocumentRepository` implements it but the abstract base in `domain/repositories.py` doesn't declare it. Either add the abstract method or refactor ingestion to use the `interfaces` layer instead.

### Performance & Edge Cases

- [ ] **Benchmark against a realistic dataset** — `bench/benchmark.py` has a `--realistic` flag that downloads FinanceBench; document expected resource usage and publish a baseline.
- [ ] **Add query-caching layer** — Identical queries within a TTL could skip LLM invocation. Consider `pipelines/cache.py` as the hook point.
- [ ] **Add batch-ingestion endpoint** — The API (both FastAPI and the `RAG` facade) ingests one file at a time; a multipart endpoint would be faster for bulk loads.
- [ ] **Handle empty document ingestion gracefully** — Test 0-byte files; the pipeline should reject early with a clear error instead of proceeding to chunking/embedding.
- [ ] **Handle Unicode/non-ASCII filenames** — Test parsers with filenames containing CJK, RTL, or emoji characters across all converter backends.
- [ ] **Concurrent ingestion race conditions** — `ingestion/service.py:261` catches `IntegrityError` for concurrent check-and-insert, but this pattern exists only once; audit all write paths for similar races.
- [ ] **Profile memory usage during large ingestion** — The `zvec` backend may have different memory characteristics than Qdrant in-memory. Add a `--memory-profile` mode to `bench/benchmark.py`.

### Documentation

- [ ] **Add a migration guide from `dynamic_rag`** — `docs/migration.md` exists but covers legacy-to-facade migration; a `dynamic_rag`→`raghub` rename migration may still be needed if users have the old import paths.
- [ ] **Document env var toggles comprehensively** — `CONTRIBUTING.md` and `.env.example` cover most vars, but `RAGHUB_CLI_RATE_LIMIT*`, `RAG_PROFILE`, and `RAGHUB_USERS` format need explicit documentation.
- [ ] **Add module docstrings for `repositories/` and `storage/` packages** — The previous docstring pass covered the new spec surface and legacy domain modules but `repositories/` and `storage/` may still be missing `__init__` docstrings.

### API & Services

- [ ] **Customise OpenAPI operation IDs** — The FastAPI app uses dynamic title/version/description but operation IDs are auto-generated; add `operation_id` to endpoints for SDK generation.
- [ ] **Add rate-limit headers to API responses** — The FastAPI rate limiter works internally but doesn't emit standard `X-RateLimit-*` headers.

---

## Lower Priority

### Security

- [ ] **Use `SecretStr` throughout** — Only `jwt_secret` in settings uses `SecretStr`; evaluate other `str`-typed secrets (e.g., `LITELLM_API_KEY`, `LANGFUSE_SECRET_KEY`) in the settings model.
- [ ] **Audit logging for auth failures** — RBAC denials and failed JWT verifications should produce structured audit log entries (structlog) with correlation IDs.
- [ ] **Dependency vulnerability scanning locally** — Add `make audit-local` that runs `pip-audit` without needing `requirements/*.txt` (they don't exist); currently only CI runs it.
- [ ] **Add `pip audit` to pre-commit** — `.pre-commit-config.yaml` runs ruff + mypy; adding `pip-audit` (or `safety`) as a hook would catch vulnerabilities before commit.

### Infrastructure

- [ ] **Install `zvec` extra in Docker by default** — The Dockerfile has `ARG INCLUDE_ZVEC=0`; consider enabling it by default once the dep is stable, or document how to set the build arg.
- [ ] **Add Windows to CI matrix** — CI only covers `ubuntu-latest` (Python 3.12, 3.13). Add a Windows runner if cross-platform support is intended.
- [ ] **Cache pip in pre-commit CI** — `.github/workflows/ci.yml` caches `.mypy_cache` and `.ruff_cache` but not the pip install layer for pre-commit deps.
- [ ] **Add `RAGHUB_USERS` env validation** — Streamlit's `RAGHUB_USERS` env var format (`email:password:org:role`) is parsed manually; add Pydantic validation or a dedicated settings field.

### Community & Governance

- [ ] **Automate issue/PR triage** — `.github/labels.yml` defines 11 labels but they must be applied manually; add a Probot or GitHub Action for auto-labelling by path/regex.
- [ ] **Add a versioning policy** — Currently no documented versioning strategy (semver? calendar?); document in `CONTRIBUTING.md` or a `VERSIONING.md`.
- [ ] **Add a security policy** — `SECURITY.md` exists but points to email; link to GitHub's private vulnerability reporting if enabled.

---

## Deprecated / Fixed (removed from active TODO)

The following items from the 2026-06-30 audit have been resolved and removed:

| Old Item | Status |
|----------|--------|
| Commit 95-file unstaged batch | ✅ Clean tree, all committed |
| Fix `pyproject.toml` version | ✅ Now `0.3.1` (consistent with CHANGELOG) |
| Commit rename + tag creation | ✅ All committed; tags pushed (except v0.3.1/v0.3.2 — see High Priority) |
| Remove stale `requirements/` | ✅ Directory removed |
| Document skipped-test env vars | ✅ Documented in `CONTRIBUTING.md` + `.env.example` |
| Lower coverage threshold | ✅ Set to 90% in CI and Makefile |
| Add CI step for non-platform tests | ✅ Full CI pipeline exists |
| Create `.env.example` | ✅ Created (69 lines) |
| Add `pre-commit` config | ✅ `.pre-commit-config.yaml` exists |
| Add `justfile` or extend `Makefile` | ✅ 18 targets in Makefile |
| Speed up container rebuilds | ✅ Dockerfile uses proper layer caching |
| Add `.dockerignore` | ✅ Created |
| Enable `disallow_untyped_defs` | ✅ Set to `true` in `mypy.ini` + `pyproject.toml` |
| Add module-level `__all__` | ✅ 60+ files defined |
| Audit docs against current API | ✅ 13 docs files, comprehensive |
| Add migration guide | ✅ `docs/migration.md` exists |
| Document plugin system | ✅ `docs/plugins.md` exists |
| Add docstrings to `interfaces/` protocols | ✅ All 17 files documented |
| Version the FastAPI app | ✅ Routers mounted under `/v1/` |
| Customise OpenAPI docs | ✅ Dynamic from package metadata |
| Add OpenAPI schema validation in CI | ✅ Present |
| Add health-check endpoint in Streamlit | ✅ Docker `HEALTHCHECK` uses `urllib` `GET /health` |
| Add rate limiter to CLI | ✅ `CLIRateLimiter` in `raghub/cli/rate_limiter.py` |
| Add `CODE_OF_CONDUCT.md` | ✅ Exists |
| Add `SUPPORT.md` | ✅ Exists |
| Add `ROADMAP.md` | ✅ Exists |
| Set up Dependabot | ✅ `.github/dependabot.yml` exists |
| Add issue/PR labels | ✅ `.github/labels.yml` (11 labels) |
| Docker Compose for development | ✅ `docker-compose.override.yml` exists |

---

*Priority levels are approximate and should be adjusted per team roadmap.*
