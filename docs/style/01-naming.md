> ⚠️ **ARCHIVED** — This document describes the **raghub 0.9.x Python
> release**, preserved for historical reference in
> [`archive/`](../../archive/). It does **not** describe the active
> **Revex** TypeScript codebase.
>
> For current documentation, see [`README.md`](../../README.md) and the
> TypeScript source under `packages/`.

# Naming Rules

This document codifies the rules every public symbol in raghub
follows. The `lint/naming.py` script enforces them automatically
(it is intentionally not in the public OSS tree; see
`.gitignore`).

## Rule R1 — No `# noqa:`. Every lint violation is fixed in the diff.

Lint suppressions are not allowed. When ruff complains, fix the
underlying code, even if the fix is "rename the variable to make
the line fit." Several ignore patterns stay in `pyproject.toml`
because they are project-wide (e.g., per-file ignores for tests/).

## Rule R2 — Two-tier privacy: public OR `__<one-word>`; `_`-prefix forbidden.

A module-level name is either:
- **public** — no leading underscore, listed in `__all__`, has a Google-style docstring.
- **deep private** — exactly the `__` prefix plus one English word, no internal underscores.
- **not in Raghub**: only `from raghub.X import _Y` (leading underscore) is an error.

Examples:

```python
def evaluate(...) -> None:    # public, no underscore
    """Evaluate the benchmark."""

def __keyed(...) -> bool:     # deep private, name-mangled
    """Whether ``row`` is an aiosqlite Row."""
```

The single-underscore prefix `_evaluate` is the forbidden middle
tier. It is removed in v1; everything that was `_evaluate` is now
either public (`evaluate`) or deep-private (`__evaluate`).

## Rule R3 — Single-word class names; `<Entity>Type` for discriminator enums.

Public class names are single nouns:

- `Chunk`, `Document`, `Hit`, `Citation`, `Citations`, `Response`,
  `Bundle`, `Section`, `Block`, `Turn`, `Session`, `User`, `Job`,
  `JobStore`, `Manifest`, `Context`, `Cache`, `Router`, `Rerank`,
  `Identity`, `Retrieval`, `Search`, `RerankerFactory`, `Result`,
  `RankedItem`, `RankedList`, `Embedding`, etc.

`<Entity>Type` is the discriminator enum per entity:

- `DocType` for `Document`,
- `ChunkType` for `Chunk`,
- `SectionType` for `Section`,
- `BlockType` for `Block`,
- `CitationType` for `Citation`,
- `HitType` for `Hit`,
- `ResponseType` for `Response`,
- `BundleType` for `Bundle`,
- `PipelineType` for `PipelineResult`,
- `JobType` for `Job`,
- `ManifestType` for `Manifest`,
- `UserType` / `SessionType` / `TurnType` for auth domain,
- `EventType` for `Event`,
- `EmbeddingType` for `Embedding`,
- `RankType` for `RankedList`,
- `ResultType` for `Result` (eval).

Shared enums (`State`, `Class`, `Access`) live in `raghub.models`.

## Rule R4 — Universal entity schema.

Every canonical entity carries the same five-attribute shape:

```
entity.id         — primary identifier (no *_id suffix on the canonical class)
entity.type       — discriminator enum (one per entity)
entity.source     — canonical locator (string), per entity semantic
entity.<children> — direct child collections (no FK string lists)
entity.verify()   — public method, raises VerificationError on inconsistency
```

`verify()` is mandatory at storage and API boundaries; opt-in via
`verify_children=False` for lazy paths. `PipelineResult.verify()`
enforces `success ⇔ error is None`.

## Rule R5 — Every reachable name is in `__all__`; every `__all__` name is reachable.

Modules declare `__all__ = [...]`. External consumers that do
`from raghub.X import *` see only the listed names. Callers using
explicit imports can reach whatever the module defines, but the
canonical public surface is exactly the `__all__` list.

If a name is reachable from outside the module but not in `__all__`,
it's a bug; if a name is in `__all__` but unreachable, that's also
a bug. The naming hook `lint/naming.py` checks both directions.

## Rule R6 — No deprecated aliases, no compatibility shims.

`DeprecationWarning` is not used. Old names simply do not exist
after the rename. The migration document (`docs/migration.md`)
is the single source of truth for old-name → new-name mapping.

## Rule R7 — Exception names end in `Error`.

Every concrete exception class is named `<Something>Error`. Naming
violations (e.g. `MissingDep`, `LLMError`, `CacheMiss`, `IngestionJob`)
are renamed in v1.
