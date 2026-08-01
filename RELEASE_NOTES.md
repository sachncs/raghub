# RAGHub v1.0.0 — Release Notes

> Frozen public surface for the OSS-ready release.

## Summary

`v1.0.0` is the **OSS-ready** release of `raghub`. The 0.5.x series
shipped as a co-development tree; v1.0 freezes the public surface,
canonicalises the universal entity schema, removes `_`-prefix
private symbols, and migrates the on-disk manifest format to v2.

There are **no backward-compat aliases** and **no deprecation period**.
Old names simply do not exist; `docs/migration.md` provides mechanical
`sed` recipes for the rename cascade.

## Headlines

- **Universal entity schema.** Every canonical entity carries `id`,
  `type`, `<source|target|parent|identity>`, direct child collections,
  and a public `verify()` method. Storage and API layers call
  `verify()` at every boundary.
- **`<Entity>Type` discriminator enums.** One enum per entity class
  (`DocType`, `ChunkType`, `SectionType`, `BlockType`, `CitationType`,
  `HitType`, `ResponseType`, `BundleType`, `PipelineType`, `JobType`,
  `EventType`, `UserKind`, `ManifestType`, `EmbeddingType`, `RankType`,
  `ResultType`, `SessionKind`).
- **Shared lifecycle enums** (`State`, `Class`, `Access`) for the
  R3 single-word class rule.
- **`VerificationError`** for `verify()` invariant failures.
- **Manifest v2 on-disk format** with `python -m raghub.migrate --root DIR`
  CLI for one-shot upgrades from v0/v1.
- **`Tokenizer.load()`** class factory (replaces `try_load_gigatoken()`).
- **`Citation.chunk`** reference; `Citations` aggregate with its own
  `verify(chunks)`.

## Renames (single-word class names, R1 / R2 / R3)

| Old name                   | New name            | Notes                          |
| -------------------------- | ------------------- | ------------------------------ |
| `ChunkRecord`              | `Chunk`             | `chunk_id` → `id`              |
| `DocumentRecord`           | `Document`          | `document_id` → `id`           |
| `SessionRecord`            | `Session`           |                                |
| `ConversationTurn`         | `Turn`              |                                |
| `UserRecord`               | `User`              | `user_id` → `id`               |
| `IngestionJob`             | `Job`               |                                |
| `PersistentJobStore`       | `JobStore`          |                                |
| `QueryCache`               | `Cache`             |                                |
| `ConversationRouter`       | `Router`            |                                |
| `ConversationManager`      | `Conversations`     | (alias of `ConversationManager`) |
| `SlidingWindowManager`     | `SlidingWindow`     | (alias)                        |
| `ConversationStore`        | `Store`             | (alias)                        |
| `DocumentBlock`            | `Block`             | (alias)                        |
| `DocumentSection`          | `Section`           | (alias)                        |

## Exception rename (`*Error` suffix, R7)

| Old name                 | New name              |
| ------------------------ | --------------------- |
| `LLMError`               | `GenerationError`     |
| `MissingDep`             | `MissingDepError`     |
| `CacheMiss`              | `CacheMissError`      |
| `AgentBudgetExceeded`    | `AgentBudgetError`    |
| `TokenBudgetExceeded`    | `TokenBudgetError`    |
| `PipelineFailed`         | `PipelineFailedError` |

## Removed layout

- **`raghub/helper/`** is gone. Its six modules move to:
  - `raghub.api_auth` (was `helper/auth`).
  - `raghub.api_response` (was `helper/response`).
  - `raghub.api_ratelimit` (was `helper/rate_limit`).
  - `raghub.api_sse` (was `helper/sse`).
  - `raghub.cli_commands` (was `helper/cli`).
  - `helper/search` deleted; `Tool.call` is the canonical path.

## Quality gates

- `pytest -q --no-cov` → **422+ passed**, 9 skipped (offline-deterministic providers).
- `ruff check raghub/ tests/` → **0 errors**.
- `mypy raghub/` → **0 errors in 40 source files**.
- `interrogate --fail-under=100` → **100.0 %** (PASSED).
- `python lint/naming.py` → **PASS** (40 files scanned).
- `python -c "import raghub"` → ok.

## Migration

`docs/migration.md` covers the mechanical `sed` recipes for the
entity rename and exception rename. Run

```bash
python -m raghub.migrate --root /your/data/dir
```

to upgrade on-disk `manifest.json` files from v0/v1 to v2 in place.

## Post-release hardening (2026-08-01)

Additional commits landed on master between v1.0.0 and the re-tag of
the release. The v1.0.0 release tag has been force-moved onto the
current HEAD.

### Added

- New test files for `auth`, `cli`, `evaluation`, `api` (+ sub-modules),
  `knowledge`, `parsers`, `stores`, `repos`, `retrieval`, `telemetry`,
  `migrate`, `plugins`, `conv`, `prompts`, `services`, `sessions`,
  `ingest`, `llm`. Each file exercises a previously under-covered
  module with content assertions.
- `raghub/__init__.py` now exposes a flat public surface; every
  `__all__` entry resolves.

### Fixed

- `raghub.auth.AuthService.login` previously read `user.id` from a
  `UserRecord` (which exposes `user_id`); corrected to use
  `user.user_id` and `record.user_id` in `resolve_user`.
- Removed duplicate `Pipeline` and dead entries in
  `raghub/__init__.py:__all__`.

## Hard rules (verbatim)

```
R1  No `# noqa:`. Every lint violation is fixed in the diff.
R2  Two-tier privacy: public OR `__<one-word>`. `_`-prefix forbidden.
R3  Single-word public class names. Discriminator enums: `<Entity>Type`.
R4  No backward compat. No aliases. No deprecation. No shims.
R5  Hard rename, single `docs/migration.md`.
R6  Storage layer version-pins on-disk format; migrates v1 → v2 on read.
R7  `<Entity>Type` enums must have ≥2 values; collapse to `str` otherwise.
R8  `verify()` mandatory at every storage and API boundary.
R9  Every state change is captured by an assertion that names the value.
R10 No `Any` outside `metadata`. `metadata` is the only `Any` slot.
```

---
*This release notes file is the source of truth for the v1.0.0 OSS-ready release.*
