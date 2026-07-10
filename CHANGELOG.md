# Changelog

All notable changes to RAGHub are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Each entry below lists the originating Git commit (short SHA) and its
ISO 8601 timestamp with timezone. Entries are ordered from newest to
oldest.

## [0.3.3] - 2026-07-02

### Changed
- Bumped package version to `0.3.3`.
  - `5b14d47` (2026-07-02T13:01:43+05:30)

## [0.3.2] - 2026-07-01

### Changed
- Bumped package version to `0.3.2`.
  - `b9cff8a` (2026-07-01T13:21:25+05:30)

### Dependencies
- `pypdf` requirement widened from `>=5.0,<6` to `>=5.0,<7`.
  - `3c4a762` (2026-07-01T07:28:17Z)
- `numpy` requirement widened from `>=1.26,<2` to `>=1.26,<3`.
  - `7c587aa` (2026-07-01T07:28:12Z)
- `chonkie` requirement widened from `>=0.5,<1` to `>=0.5,<2`.
  - `d8ee793` (2026-07-01T07:28:08Z)
- `structlog` requirement widened from `>=24,<26` to `>=24,<27`.
  - `0c6111c` (2026-07-01T07:28:04Z)

## [0.3.1] - 2026-07-01

### Added
- New FastAPI endpoints, CLI extensions, and Streamlit UI refinements driving the
  `RAG` facade surface; see `raghub/api/`, `raghub/cli/`, and the new
  `tests/test_api_endpoints.py`, `tests/test_cli_commands.py`, and
  `tests/test_rag_facade.py` suites.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New ingestion, conversion, vector store, retrieval, generation, validation,
  document lifecycle, container, conversation, observability, knowledge,
  repository, storage, and migration test suites under `tests/`.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New `raghub.pipelines.cache` module.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New `raghub.repositories.sqlite_document_repo` module.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New `raghub.storage.sqlite_session_store` module.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New `bench/benchmark.py` performance harness refinements
  (`--documents`, `--queries`, `--concurrency` flags, JSON report output).
  - `d95d650` (2026-07-01T12:55:47+05:30)
- New project hygiene files: `.dockerignore`, `.env.example`,
  `.pre-commit-config.yaml`, `.github/dependabot.yml`, `.github/labels.yml`,
  `.github/workflows/ci.yml`, `.github/workflows/labeler.yml`,
  `CODE_OF_CONDUCT.md`, `ROADMAP.md`, `SUPPORT.md`, and a `Makefile` with
  install/test/lint/typecheck/coverage/docs/bench/security targets.
  - `d95d650` (2026-07-01T12:55:47+05:30)

### Changed
- Bumped package version to `0.3.1`.
  - `d95d650` (2026-07-01T12:55:47+05:30)
  - `ba7a890` (2026-07-01T12:56:08+05:30)
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
  - `d95d650` (2026-07-01T12:55:47+05:30)
- Updated `Dockerfile`, `Makefile`, `pyproject.toml`,
  `docs/architecture/decisions.md`, `docs/guide/getting-started.md`,
  `docs/plugins.md`, and `docs/reference/api.md` to reflect the new install,
  test, lint, and typecheck workflow.
  - `d95d650` (2026-07-01T12:55:47+05:30)

### Removed
- Legacy `requirements/*.txt` files (`base.txt`, `dev.txt`, `ml.txt`,
  `observability.txt`, `parsers.txt`, `all.txt`); the single source of
  truth is now `pyproject.toml`.
  - `d95d650` (2026-07-01T12:55:47+05:30)
- `.hypothesis/` test cache directory that was inadvertently committed in
  `d95d650`.
  - `ba7a890` (2026-07-01T12:56:08+05:30)

## [0.3.0] - 2026-06-30

### Changed
- Bumped package version to `0.3.0`. No source-code changes were made
  between this commit and `6e5199e`.
  - `b39ed86` (2026-06-30T11:41:24+05:30)

## [0.3.0-pre] - 2026-06-30

### Changed
- Internal pre-release marker; no source-code changes.
  - `6e5199e` (2026-06-30T08:57:54+05:30)

## [0.2.0-beta] - 2026-06-30

### Changed
- Labelled the rewrite as the first platform beta. No source-code
  changes between this commit and `4146805`.
  - `8fbdbcd` (2026-06-30T00:04:52+05:30)

## [0.1.0] - 2026-06-28

### Changed
- Marked the first versioned (non-prerelease) cut of the rewritten
  `raghub` package. No source-code changes between this commit and
  `71d4be0`.
  - `4146805` (2026-06-28T19:35:15+05:30)

## [0.1.0-alpha] - 2026-06-28

### Added
- Initial alpha of the `raghub` package rewrite, bundling the work
  between commits `edbcb43` and `71d4be0`:
  - Renamed the project from `dynamic_rag` to `raghub`.
    - `74ef554` (2026-06-26T15:50:55+05:30)
  - Added a virtualenv-friendly developer setup (`setup.sh` and
    `requirements/*.txt`).
    - `bf16cbc` (2026-06-26T16:26:12+05:30)
  - Renamed `_`-prefixed names to public identifiers for Google-style
    compliance.
    - `512177e` (2026-06-26T16:26:15+05:30)
  - Added multi-format document parsers: PDF, HTML, image OCR, Office,
    CSV, and plain text.
    - `150adc2` (2026-06-26T16:26:18+05:30)
  - Added NVIDIA NV-Embed-QA and sentence-transformers embedding
    providers; ChatNVIDIA MiniMax M3 multimodal LLM; and a token-aware
    prompt builder.
    - `d8fdcee` (2026-06-26T16:26:22+05:30)
  - Added SQLite-backed persistence for the document registry, session
    store, and user store.
    - `972ffd0` (2026-06-26T16:26:26+05:30)
  - Added JWT authentication, RBAC authorization, and a SQLite user
    store.
    - `c9fbdc6` (2026-06-26T16:26:29+05:30)
  - Added structlog logging, Prometheus metrics, and OpenTelemetry
    tracing.
    - `f9937f6` (2026-06-26T16:26:32+05:30)
  - Added a rate limiter, sliding-window conversation store, async
    ingestion, admin API, and faceted search.
    - `0430033` (2026-06-26T16:26:36+05:30)
  - Wired the new components into `build_container` and the API layer.
    - `5e93074` (2026-06-26T16:26:39+05:30)
  - Added the initial test suites (parsers, SQLite, JWT auth,
    embeddings, prompts, observability, etc.).
    - `ffd8fad` (2026-06-26T16:26:42+05:30)
  - Added `.gitignore` entries for `__pycache__`, `.venv`, and the
    `data/` directory.
    - `2299158` (2026-06-26T16:26:55+05:30)
- Cut the alpha release.
  - `71d4be0` (2026-06-28T16:51:20+05:30)

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
