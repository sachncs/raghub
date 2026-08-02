# Changelog

All notable changes to RAGHub are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

Each entry below lists the originating Git commit (short SHA) and its
ISO 8601 timestamp with timezone. Entries are ordered from newest to
oldest.

## [0.6.0] - 2026-07-31

### Post-release hardening (2026-08-01)

Additional commits landed on master after the 0.5.x series. The
public surface continues to harden; release notes are tracked in
`RELEASE_NOTES.md`.

### Added

- New test files: `tests/test_auth.py`, `tests/test_cli.py`,
  `tests/test_evaluation.py`, `tests/test_api.py`,
  `tests/test_knowledge.py`, `tests/test_parsers.py`,
  `tests/test_stores.py`, `tests/test_repos.py`,
  `tests/test_retrieval.py`, `tests/test_telemetry.py`,
  `tests/test_misc.py`, `tests/test_sessions.py`,
  `tests/test_ingest_module.py`, `tests/test_llm.py`. Each file
  exercises a previously under-covered module with content assertions.
- `raghub/__init__.py` now exposes a flat public surface with all
  `__all__` entries resolving.

### Fixed

- `raghub.auth.AuthService.login` previously read `user.id` from a
  `UserRecord` (which exposes `user_id`); corrected to use `user.user_id`
  and `record.user_id` in `resolve_user`.
- Removed duplicate `Pipeline` and dead entries in
  `raghub/__init__.py:__all__`.

### Changed (BREAKING)

The 0.6 series continues the restructuring of the 0.5.x releases. There
are **no backward-compat aliases** and **no deprecation period**.
Old names simply do not exist any more. See `docs/migration.md` for the
old → new rename table; the rename is mechanical.

### Added

- **Universal entity schema.** Every canonical entity carries
  `id`, `type`, `<source|target|parent|identity>`, direct child
  collections, and a public `verify()` method. Storage and API
  layers call `verify()` at every boundary.
- **`<Entity>Type` discriminator enums.** One enum per entity class.
- **`VerificationError`** for `verify()` invariant failures.
- **Manifest v2 on-disk format** with `raghub.migrate --root DIR`
  CLI for one-shot upgrades.
- **`Tokenizer.load()`** class factory (replaces
  `try_load_gigatoken()`).
- **`Citation.chunk`** reference; `Citations` aggregate with its own
  `verify(chunks)`.

### Renamed

Following the universal schema, no `_`-prefix private names remain
(R2):

- `_evaluate` → `evaluate` (public), or `__evaluate` deep-private.
- `_is_aiosqlite_row` → `__keyed`.
- `_resolve_config_dir` → `__resolve`.
- `_env_int` / `_env_float` → inlined at every call site.

Single-word class names:

- `ChunkRecord` → `Chunk`; field `chunk_id` → `id`.
- `DocumentRecord` → `Document`; field `document_id` → `id`.
- `SessionRecord` → `Session`.
- `ConversationTurn` → `Turn`.
- `UserRecord` → `User`; field `user_id` → `id`.
- `IngestionJob` → `Job`.
- `PersistentJobStore` → `JobStore`.
- `QueryCache` → `Cache`.
- `ConversationRouter` → `Router`.
- `ConversationManager` → `Conversations`.
- `SlidingWindowManager` → `SlidingWindow`.
- `ConversationStore` → `Store`.
- `DocumentBlock` → `Block`.
- `DocumentSection` → `Section`.

Exception rename (`*Error` suffix, R7):

- `LLMError` → `GenerationError`.
- `MissingDep` → `MissingDepError`.
- `CacheMiss` → `CacheMissError`.
- `AgentBudgetExceeded` → `AgentBudgetError`.
- `TokenBudgetExceeded` → `TokenBudgetError`.
- `PipelineFailed` → `PipelineFailedError`.

### Removed

- **`raghub.helper/`**. Its six modules move to:
  - `raghub.api_auth` (was `helper/auth`).
  - `raghub.api_response` (was `helper/response`).
  - `raghub.api_ratelimit` (was `helper/rate_limit`).
  - `raghub.api_sse` (was `helper/sse`).
  - `raghub.cli_commands` (was `helper/cli`).
  - `helper/search` deleted; `Tool.call` is the canonical path.

### Quality

- 416+ unit tests pass (offline-deterministic providers).
- `ruff check`: 0 errors.
- `interrogate --fail-under=100`: passes.
- `mypy raghub/`: 0 errors in 40 source files.
- `lint/naming.py` (local hook, gitignored): passes.

### Migration

`sed` recipes are in `docs/migration.md`. Run
`python -m raghub.migrate --root /your/data/dir` to upgrade on-disk
manifest files.

## [0.5.0] - 2026-07-30

### Changed (BREAKING)

The v0.5 release is a renaming-and-restructuring refactor. There are
**no backward-compat aliases** — code importing the old names will
fail to import. The migration is mechanical (see
`docs/migration.md`).
