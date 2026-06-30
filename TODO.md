# TODO · RAGHub

## High Priority

### Build & Release

- [ ] **Commit the 95-file unstaged batch** — docstrings, naming cleanup, type fixes, CHANGELOG rewrite — before it diverges further
- [ ] **Fix `pyproject.toml` version** — currently `1.0.0` but tags/CHANGELOG say `v0.2.0-beta`; needs to be `0.2.0` or bumped consistently
- [ ] **Commit `raghub/cli/_common.py → common.py` rename** — currently staged but not committed (blocks git-flow)
- [ ] **Stage + commit tag creation** — `v0.1.0-alpha`, `v0.1.0`, `v0.2.0-beta` tags only exist locally; push to origin (or confirm they are pushed)
- [ ] **Remove stale `requirements/` directory** — `setup.sh` now uses `pip install -e ".[dev,api,ui,zvec]"`; `requirements/*.txt` are dead documentation

### Testing

- [ ] **Document skipped-test env vars** — `RAGHUB_RUN_PLATFORM_TESTS=1`, `FINANCEBENCH_EVAL=1` should be documented in `CONTRIBUTING.md` or a `.env.example`
- [ ] **Lower coverage threshold** or raise it — currently 72%; either raise to 80% or document why 72% is acceptable
- [ ] **Add CI step** that runs the non-platform tests with the exact same flags as local (`pytest -q --cov=raghub`)

## Medium Priority

### Developer Experience

- [ ] **Create `.env.example`** — list all env vars the app reads (`JWT_SECRET`, `LITELLM_API_KEY`, `QDRANT_URL`, `LANGFUSE_*`, `OPENAI_API_KEY`, etc.)
- [ ] **Add `pre-commit` config** — ruff format + ruff check + mypy as a pre-commit hook
- [ ] **Add `justfile` or extend `Makefile`** — add targets for `dev-api`, `dev-ui`, `db-init`, `db-reset`
- [ ] **Speed up container rebuilds** — Dockerfile copies `pyproject.toml` and installs deps before copying source, which is good, but could add a `.dockerignore`
- [ ] **Add `.dockerignore`** — exclude `.git/`, `__pycache__/`, `*.db`, `data/`, `tests/`, `docs/`, `bench/`

### Code Quality

- [ ] **Enable `disallow_untyped_defs = true` in mypy** — or at least on a subset of modules; many functions still lack type annotations
- [ ] **Eliminate remaining circular imports** — the lazy-import wrapper for `migrate_from_json` in `raghub/storage` is a symptom; refactor to break the cycle properly
- [ ] **Audit protocol conformance** — `raghub/interfaces/` defines many Protocols; verify every implementation fully conforms
- [ ] **Add module-level `__all__`** to all public subpackages — currently missing from most packages

### Documentation

- [ ] **Audit docs against current API** — `docs/` may be stale after the `_`-prefix rename and docstring pass
- [ ] **Add migration guide from `dynamic_rag`** — `docs/migration.md` exists but may need updating after the rename
- [ ] **Document the plugin system** — `docs/plugins.md` covers how to write a plugin but not how entry-point discovery works in practice
- [ ] **Add docstrings to the `interfaces/` protocols** — consumers of the library need to know contract expectations

### API & Services

- [ ] **Version the FastAPI app** — mount routes under `/v1/` to allow future breaking changes
- [ ] **Customise OpenAPI docs** — add `title`, `description`, `version` from `pyproject.toml` dynamically
- [ ] **Add OpenAPI schema validation in CI** — validate that the generated OpenAPI spec is valid
- [ ] **Add a health-check endpoint in Streamlit** — the Docker `HEALTHCHECK` calls the CLI health command, which is fragile

## Lower Priority

### Edge Cases & Robustness

- [ ] **Handle empty document ingestion gracefully** — test what happens when a 0-byte file is ingested
- [ ] **Unicode/non-ASCII filenames** — test parsers with non-ASCII filenames and content
- [ ] **Concurrent ingestion race conditions** — SQLite `IntegrityError` handling when two requests insert the same document hash simultaneously
- [ ] **qdrant `grpc` vs `http`** — the client defaults may not be optimal for all deployments; document the trade-off
- [ ] **Large-file streaming** — test ingestion of files >100MB to verify memory bounds
- [ ] **Token-count edge cases** — what happens when context exceeds the model's max tokens? The prompt builder handles it but the pipeline should too

### Performance

- [ ] **Benchmark `bench/` against a realistic dataset** — currently uses synthetic documents; add a mode that downloads FinanceBench
- [ ] **Add query-caching layer** — identical queries within a TTL could skip LLM invocation
- [ ] **Add batch-ingestion endpoint** — the API only ingests one file at a time; a multipart endpoint would be faster for bulk loads
- [ ] **Profile memory usage during large ingestion** — the `zvec` backend may have different memory characteristics than Qdrant in-memory

### Security

- [ ] **Use `SecretStr` for `jwt_secret`** — deferred from v0.2.0-beta; move from `str` to Pydantic `SecretStr`
- [ ] **Rate-limit the CLI commands** — only the API has rate limiting; repeated CLI ingest/query calls should respect limits
- [ ] **Audit logging for auth failures** — RBAC denials and failed JWT verifications should produce structured audit log entries
- [ ] **Dependency vulnerability scanning DAST** — `pip-audit` runs in CI but not locally; add a `make audit-local` that works without `requirements/*.txt`

### Infrastructure

- [ ] **Docker Compose for development** — add a `docker-compose.override.yml` with volume mounts and hot-reload
- [ ] **Add `zvec` to the Docker image** — the optional extra is not installed in the Dockerfile; add `.[api,ui,zvec]` if the dep is stable
- [ ] **CI cache for pip** — the workflow has `pip cache` but could also cache `.mypy_cache` and `.ruff_cache`
- [ ] **Test on Windows** — the CI matrix only covers ubuntu-latest; add a Windows runner if cross-platform support is intended

### Community & Governance

- [ ] **Add a `CODE_OF_CONDUCT.md`**
- [ ] **Add a `SUPPORT.md`** — where to file issues, ask questions, get help
- [ ] **Add a `ROADMAP.md`** — extract from `docs/future.md` into a standalone roadmap file
- [ ] **Set up Dependabot** — `.github/dependabot.yml` for automated dependency updates
- [ ] **Add issue/PR labels** — automate triage with GitHub Actions or Probot

---

*Generated 2026-06-30 from a full repository audit. Priority levels are approximate and should be adjusted per team roadmap.*
