# Changelog

All notable changes to RAGHub are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Each entry below lists the originating Git commit (short SHA) and its
ISO 8601 timestamp with timezone. Entries are ordered from newest to
oldest.

## [0.4.0] - 2026-07-18

### Added
- New `RAG` facade as the canonical entry point (`raghub.RAG`):
  single object with `ingest`, `aquery`, `astream`, `sync_index`,
  `delete`, `evaluate`, and a configurable extra surface. Lazy-imports
  every collaborator; tests/e2e/test_cli_and_facade.py exercises the
  full public surface.
  - `b2fe70e` (2026-07-17T05:46:09+05:30)
- New `RAG.ingest_directory_*` and `RAG.sync_index` paths with
  `tqdm.tqdm` progress bars on the directory loop, the chunking
  loop, and the JSON-to-SQLite migration path. Show-progress is
  opt-out via `show_progress=False`.
  - `cbc527a` (2026-07-17T08:11:53+05:30)
- New `RAG.run` daemon-style subcommand (`raghub/api/cli/run_cmd.py`)
  that boots uvicorn against the `get_app` factory in the foreground.
  - `dbcf0e3` (2026-07-17T08:37:11+05:30)
- New `scripts/build_requirements_lock.py` and
  `requirements-lock.txt`; CI validates the lock is unchanged on
  every push so production images always install a vetted set of
  dependencies.
  - `cfbc1ea` (2026-07-18T01:32:01+05:30)
- New `RAGHUB_RUN_PLATFORM_TESTS` opt-out in `tests/conftest.py`;
  the platform and dynamic-application test suites now run as
  part of the normal test run instead of behind an env flag.
  - `ee00e93` (2026-07-17T11:28:04+05:30)
- New `tests/e2e/test_cli_and_facade.py` qualitative end-to-end
  suite covering `python -m raghub.cli` and the `RAG` facade in
  isolation.
  - `6ac388f` (2026-07-17T14:18:00+05:30)

### Changed
- Bumped package version to `0.4.0`; sdist + wheel rebuild from
  `requirements-lock.txt`.
  - `7a4a220` (2026-07-17T17:14:40+05:30)
  - `58168ab` (2026-07-18T03:21:23+05:30)
- Replaced `structlog` with `loguru` across the observability stack.
  `raghub.observability.logging.build_logger` returns a
  `LoguruLogger`; `RedactingTelemetry` and `LoguruTelemetryProvider`
  are the only public re-exports. The `Tracer` class replaces
  `OpenTelemetryTracer` with a public `add_otlp_exporter` method.
  - `a6d26b1` (2026-07-17T06:51:34+05:30)
- Renamed every semi-private module-level identifier in
  production code (`_limiter`, `metric_collectors`, `_RATE_LIMIT_EXEMPT`,
  `_native_filter`, `_canonical_filters`, `_key`, `_factory`,
  `_generate`, `_mod`, `_mod2`, `_parser_mod`, `_hf_mod`, `app_instance`,
  etc.) to public names. The RAG facade and CLI use these new names
  end-to-end.
  - `99907f7` (2026-07-17T08:06:46+05:30)
- Deleted the legacy JWT auth path (`raghub.auth.service`,
  `JwtAuthenticator`, `JwtSessionManager`). 0.4.0 mints opaque
  session tokens via `SqliteSessionStore`; the `JWT_SECRET` env var
  is repurposed to sign those tokens.
  - `68d7414` (2026-07-17T08:19:31+05:30)
- Canonical ingestion pipeline (single `IngestPipeline`):
  the `DocumentIngestionService` is a thin compatibility wrapper.
  Re-ingest of a `FAILED` document is allowed (regression fix);
  bundle is persisted only after a successful vector upsert.
  - `70866e2` (2026-07-17T08:24:05+05:30)
- Qdrant vector store now forwards `QDRANT_URL` and
  `QDRANT_API_KEY` correctly; `create_collection` is non-destructive
  (404 → `create_collection`; no `recreate_collection`).
  - `3def058` (2026-07-17T08:26:11+05:30)
- RBAC: empty `allowed_companies` is now fail-closed across every
  retrieval path. `allowed_company_filter(user)` returns
  `{"company": []}` for non-admin users with no allow-list and the
  vector stores match zero records.
  - `4b9df4b` (2026-07-17T08:26:52+05:30)
- SQLite durability: `DatabaseManager` uses `isolation_level=None`
  (autocommit); `PRAGMA wal_checkpoint(TRUNCATE)` runs on close.
  `UnitOfWork.close()` is reachable from `application.shutdown()`
  and `RAG.shutdown()`.
  - `a89aba7` (2026-07-17T08:37:54+05:30)
- Prometheus: `record_latency("ingest.*", ...)` routes to
  `raghub_ingestion_duration_ms`; `record_latency("auth.*", ...)`
  to `raghub_auth_duration_ms`; `record_latency("query.*", ...)`
  to `raghub_query_duration_ms`. Token counters route to
  `raghub_{prompt,completion}_tokens_total` with the `model` label.
  - `fff254a` (2026-07-17T08:38:46+05:30)
- API: `get_app()` is the canonical entry point; the CORS guard
  refuses wildcard+credentials at startup; the oversize-upload
  guard returns 413 from the request layer; admin endpoints redact
  `password_hash`; the default demo seed is suppressed in
  production and when `CORS_ORIGINS` is `*`.
  - `5a295f2` (2026-07-17T09:00:26+05:30)
- CLI: every `print()` replaced with `loguru.logger.{info,warn,error}`.
  Added `raghub run` and stable `print_json` / `write_json` helpers.
  - `dbcf0e3` (2026-07-17T08:37:11+05:30)
- CI: ruff and mypy are clean (238 files formatted; `mypy raghub/`
  passes with no issues). Pre-existing lint findings in untouched
  files are documented in the `pyproject.toml` ignore list.
  - `a6d26b1` (2026-07-17T06:51:34+05:30)
  - `6ac388f` (2026-07-17T14:18:00+05:30)
- Test runs are ungated by default; `RAGHUB_RUN_PLATFORM_TESTS`
  remains a manual opt-out, not a requirement.
  - `ee00e93` (2026-07-17T11:28:04+05:30)

### Fixed
- CI: bench format drift (`bench/benchmark.py`), e2e-smoke
  container cleanup (`if: always()` + `2>/dev/null || true` on
  `docker stop raghub-ci`), and the `piptools` missing-dependency
  issue that broke `scripts/build_requirements_lock.py` on Python
  3.13. The constraints file is now in-repo at
  `scripts/requirements-constraints.in`.
  - `58168ab` (2026-07-18T03:21:23+05:30)
- Mypy `no-redef` on `raghub/embeddings/sentence_transformer.py`:
  the module-level `SentenceTransformer: Any` annotation and the
  in-block `from sentence_transformers import SentenceTransformer`
  collided; renamed the import to `SentenceTransformerClass` and
  typed it as `type | None`.
  - `63f41a5` (2026-07-18T04:21:09+05:30)
- `pip-audit --strict` is clean on the locked production set:
  `pillow 12.3.0`, `pypdf 6.14.2`, `transformers 5.x` (transitive),
  `markdownify 0.14.1+` (transitive via marker-pdf, now an opt-in
  extra).
  - `cfbc1ea` (2026-07-18T01:32:01+05:30)

### Removed
- Legacy `requirements/*.txt` files; the single source of truth is
  `pyproject.toml` (for loose ranges) and `requirements-lock.txt`
  (for the production install).
  - `cfbc1ea` (2026-07-18T01:32:01+05:30)
- `JwtAuthenticator` and `JwtSessionManager` modules; tests that
  imported them now exercise the opaque-token path.
  - `68d7414` (2026-07-17T08:19:31+05:30)

## [0.3.3] - 2026-07-02

### Changed
- Bumped package version to `0.3.3`.
  - `b32a434` (2026-07-02T13:01:43+05:30)

## [0.3.2] - 2026-07-01

### Changed
- Bumped package version to `0.3.2`.
  - `4a3ac90` (2026-07-01T13:21:25+05:30)

### Dependencies
- `pypdf` requirement widened from `>=5.0,<6` to `>=5.0,<7`.
  - `4b99971` (2026-07-01T07:28:17Z)
- `numpy` requirement widened from `>=1.26,<2` to `>=1.26,<3`.
  - `34bafc6` (2026-07-01T07:28:12Z)
- `chonkie` requirement widened from `>=0.5,<1` to `>=0.5,<2`.
  - `535e4ba` (2026-07-01T07:28:08Z)
- `structlog` requirement widened from `>=24,<26` to `>=24,<27`.
  - `9ad6bdb` (2026-07-01T07:28:04Z)

## [0.3.1] - 2026-07-01

### Added
- New FastAPI endpoints, CLI extensions, and Streamlit UI refinements driving the
  `RAG` facade surface; see `raghub/api/`, `raghub/cli/`, and the new
  `tests/test_api_endpoints.py`, `tests/test_cli_commands.py`, and
  `tests/test_rag_facade.py` suites.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New ingestion, conversion, vector store, retrieval, generation, validation,
  document lifecycle, container, conversation, observability, knowledge,
  repository, storage, and migration test suites under `tests/`.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New `raghub.pipelines.cache` module.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New `raghub.repositories.sqlite_document_repo` module.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New `raghub.storage.sqlite_session_store` module.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New `bench/benchmark.py` performance harness refinements
  (`--documents`, `--queries`, `--concurrency` flags, JSON report output).
  - `ab75155` (2026-07-01T12:55:47+05:30)
- New project hygiene files: `.dockerignore`, `.env.example`,
  `.pre-commit-config.yaml`, `.github/dependabot.yml`, `.github/labels.yml`,
  `.github/workflows/ci.yml`, `.github/workflows/labeler.yml`,
  `CODE_OF_CONDUCT.md`, `ROADMAP.md`, `SUPPORT.md`, and a `Makefile` with
  install/test/lint/typecheck/coverage/docs/bench/security targets.
  - `ab75155` (2026-07-01T12:55:47+05:30)

### Changed
- Bumped package version to `0.3.1`.
  - `f7fc633` (2026-07-01T12:55:47+05:30)
  - `70408aa` (2026-07-01T12:56:08+05:30)
- Refactored `raghub.api.app`, `raghub.api.rag`, `raghub.api.schemas`,
  `raghub.auth.service`, `raghub.cli.main`, `raghub.cli.common`,
  `raghub.config.settings`, `raghub.conversation.memory`,
  `raghub.converters`, `raghub.documents.validation`, `raghub.evaluation`,
  `raghub.generation`, `raghub.ingestion.service`,
  `raghub.interfaces.observability`, `raghub.interfaces.storage`,
  `raghub.interfaces.vectorstore`, `raghub.models.api`,
  `raghub.pipelines.rag`, `raghub.plugins`, `raghub.services.application`,
  `raghub.services.auth_service`, `raghub.storage`, `raghub.utils`, and
  `raghub.vectorstore.qdrant` in support of the new facade and updated
  dependency bounds.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- Updated `Dockerfile`, `Makefile`, `pyproject.toml`,
  `docs/architecture/decisions.md`, `docs/guide/getting-started.md`,
  `docs/plugins.md`, and `docs/reference/api.md` to reflect the new install,
  test, lint, and typecheck workflow.
  - `ab75155` (2026-07-01T12:55:47+05:30)

### Removed
- Legacy `requirements/*.txt` files (`base.txt`, `dev.txt`, `ml.txt`,
  `observability.txt`, `parsers.txt`, `all.txt`); the single source of
  truth is now `pyproject.toml`.
  - `ab75155` (2026-07-01T12:55:47+05:30)
- `.hypothesis/` test cache directory that was inadvertently committed in
  `ab75155`.
  - `70408aa` (2026-07-01T12:56:08+05:30)

## [0.3.0] - 2026-06-30

### Changed
- Bumped package version to `0.3.0`. No source-code changes were made
  between this commit and `32c7b4a`.
  - `8a06ce0` (2026-06-30T11:41:24+05:30)

## [0.3.0-pre] - 2026-06-30

### Changed
- Internal pre-release marker; no source-code changes.
  - `32c7b4a` (2026-06-30T08:57:54+05:30)

## [0.2.0-beta] - 2026-06-30

### Changed
- Labelled the rewrite as the first platform beta. No source-code
  changes between this commit and `9f625e5`.
  - `69c446c` (2026-06-30T00:04:52+05:30)

## [0.1.0] - 2026-06-28

### Changed
- Marked the first versioned (non-prerelease) cut of the rewritten
  `raghub` package. No source-code changes between this commit and
  `69e48a5`.
  - `9f625e5` (2026-06-28T19:35:15+05:30)

## [0.1.0-alpha] - 2026-06-28

### Added
- Initial alpha of the `raghub` package rewrite, bundling the work
  between commits `edbcb43` and `69e48a5`:
  - Renamed the project from `dynamic_rag` to `raghub`.
    - `ee7e3e8` (2026-06-26T15:50:55+05:30)
  - Added a virtualenv-friendly developer setup (`setup.sh` and
    `requirements/*.txt`).
    - `a26b3c2` (2026-06-26T16:26:12+05:30)
  - Renamed `_`-prefixed names to public identifiers for Google-style
    compliance.
    - `1961889` (2026-06-26T16:26:15+05:30)
  - Added multi-format document parsers: PDF, HTML, image OCR, Office,
    CSV, and plain text.
    - `7904ab2` (2026-06-26T16:26:18+05:30)
  - Added NVIDIA NV-Embed-QA and sentence-transformers embedding
    providers; ChatNVIDIA MiniMax M3 multimodal LLM; and a token-aware
    prompt builder.
    - `73463fe` (2026-06-26T16:26:22+05:30)
  - Added SQLite-backed persistence for the document registry, session
    store, and user store.
    - `d8ec4ca` (2026-06-26T16:26:26+05:30)
  - Added JWT authentication, RBAC authorization, and a SQLite user
    store.
    - `8a37864` (2026-06-26T16:26:29+05:30)
  - Added structlog logging, Prometheus metrics, and OpenTelemetry
    tracing.
    - `9086e73` (2026-06-26T16:26:32+05:30)
  - Added a rate limiter, sliding-window conversation store, async
    ingestion, admin API, and faceted search.
    - `263305a` (2026-06-26T16:26:36+05:30)
  - Wired the new components into `build_container` and the API layer.
    - `2a57071` (2026-06-26T16:26:39+05:30)
  - Added the initial test suites (parsers, SQLite, JWT auth,
    embeddings, prompts, observability, etc.).
    - `c161c74` (2026-06-26T16:26:42+05:30)
  - Added `.gitignore` entries for `__pycache__`, `.venv`, and the
    `data/` directory.
    - `5a88996` (2026-06-26T16:26:55+05:30)
- Cut the alpha release.
  - `69e48a5` (2026-06-28T16:51:20+05:30)

## [0.0.1] - 2026-06-26

### Added
- Project inception. Codebase forked from the `dynamic_rag` prototype
  under a working `raghub` import path; no production assets shipped.
  - `edbcb43` (2026-06-26T14:56:06+05:30)

[0.3.3]: https://github.com/sachncs/raghub/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/sachncs/raghub/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/sachncs/raghub/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/sachncs/raghub/compare/v0.3.0-pre...v0.3.0
[0.3.0-pre]: https://github.com/sachncs/raghub/compare/v0.2.0-beta...v0.3.0-pre
[0.2.0-beta]: https://github.com/sachncs/raghub/compare/v0.1.0...v0.2.0-beta
[0.1.0]: https://github.com/sachncs/raghub/compare/v0.1.0-alpha...v0.1.0
[0.1.0-alpha]: https://github.com/sachncs/raghub/compare/v0.0.1...v0.1.0-alpha
[0.0.1]: https://github.com/sachncs/raghub/releases/tag/v0.0.1
