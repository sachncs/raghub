# v0.9.0 — Todo Index

This folder holds the **acceptance criteria** and **success criteria**
for every atomic item in the v0.9.0 release.

v0.9.0's theme is **"what the plan claimed was shipped, actually
shipped"** — Tiers 1-5 of the prior plan promised v0.7.x feature wiring,
CLI, and isolation tiers that were never delivered.

## Reading convention

Each item has **five fields**:

| Field | Meaning |
|---|---|
| **File(s)** | Exactly which source files this PR edits. One PR = one item. |
| **Change** | Concrete diff description; one sentence per behaviour touched. |
| **Test** | Exact new test(s) added; pytest path. |
| **Acceptance criteria** | What must be true for the PR to merge, drawn from `AGENTS.md` rules (R1-R10, naming, typing, visibility, dataclass, magic values, type safety, etc.). |
| **Success criteria** | Objective, measurable outcomes verifiable by running a single command or reading a single artefact. |

A PR is **done** when both its acceptance and success criteria are met.

## Tier index

| Tier | Theme | Items | File |
|---|---|---|---|
| 1 | Wire v0.7.x collaborators into `RAG.__init__` | 1-7 | [`tier1-rag-wiring.md`](tier1-rag-wiring.md) |
| 2 | Make v0.7.6 isolation tiers real | 8-15 | [`tier2-isolation.md`](tier2-isolation.md) |
| 3 | Make v0.7.7 feedback loops real | 16-20 | [`tier3-feedback.md`](tier3-feedback.md) |
| 4 | Make v0.7.4 ingestion queue actually used | 21-24 | [`tier4-queue.md`](tier4-queue.md) |
| 5 | CLI for v0.7.5 / v0.7.6 / v0.7.8 | 25-30 | [`tier5-cli.md`](tier5-cli.md) |
| 6 | Constitution cleanup (R2, naming) | 31-36 | [`tier6-constitution.md`](tier6-constitution.md) |
| 7 | Type discipline (R10) | 37-40 | [`tier7-types.md`](tier7-types.md) |
| 8 | Coverage gate honesty | 41-46 | [`tier8-coverage.md`](tier8-coverage.md) |
| 9 | Documentation (remove Docker, update for v0.7.x) | 47-53 | [`tier9-docs.md`](tier9-docs.md) |
| 10 | Architectural splits (god-modules) | 54-60 | [`tier10-splits.md`](tier10-splits.md) |
| 11 | ADRs (decisions recorded) | 61-65 | [`tier11-adrs.md`](tier11-adrs.md) |
| 12 | Adapter reference docs | 66-70 | [`tier12-adapter-docs.md`](tier12-adapter-docs.md) |
| 13 | `mkdocs build --strict` | 71 | [`tier13-mkdocs.md`](tier13-mkdocs.md) |
| 14 | Cleanup + version bump + CHANGELOG | 72-78 | [`tier14-cleanup.md`](tier14-cleanup.md) |

**Total: 78 atomic items.**

## Acceptance criteria vocabulary

These are extracted from `AGENTS.md`. Every item references the
relevant ones; they are repeated here for self-contained reading.

### Hard rules

- **R1** — No `# noqa` (or `# type: ignore`) anywhere in `raghub/`. Every lint violation is fixed in the diff.
- **R2** — Two-tier visibility: every identifier is **public** OR `__<one-word>__`. Single-underscore `_foo` is forbidden.
- **R3** — Single-word class names; discriminator enums `<Entity>Type` with ≥ 2 values.
- **R4** — No backward-compat aliases, shims, or deprecation periods.
- **R5** — Single `docs/migration.md` for renames.
- **R6** — Storage layers version-pin on-disk format; migrate on read.
- **R7** — `<Entity>Type` enums must have ≥ 2 values (otherwise collapse to `str`).
- **R8** — `verify()` invoked at every storage and API boundary.
- **R9** — State changes captured by assertions that name the value.
- **R10** — No `Any` outside `metadata`. `metadata` is the only `Any` slot.

### Naming

- **N1** — Forbidden names: `tmp`, `temp`, `foo`, `bar`, `baz`, `obj`, `var`, `item`, `misc`, `thing`, `manager`, `helper`, `utils`, `processor`, `handler`, `service`, `engine`, `Mixin`, `Svc`, `Worker`, etc.
- **N2** — No abbreviations. Spell out the word.
- **N3** — Module names are snake_case.
- **N4** — Constants are `UPPER_CASE`.

### Type discipline

- **T1** — `mypy --strict raghub/` returns zero errors.
- **T2** — `pytest --cov-fail-under=70` (the gate v0.9.0 enforces; 90% is the v1.0 aspirational gate).
- **T3** — `ruff check raghub/ tests/` returns zero errors.
- **T4** — `interrogate -c pyproject.toml` passes at 100% docstring coverage.
- **T5** — `python lint/naming.py` passes.
- **T6** — `pip-audit --strict` passes.
- **T7** — `bandit -ll -i -r raghub/` passes.

### Function / class discipline

- **C1** — Functions ≤ 40 LOC; `__init__` ≤ 30 LOC.
- **C2** — No god-modules; modules ≤ 500 LOC unless justified.
- **C3** — Dataclasses use `frozen=True` when value-object semantics apply; mutable only when justified.
- **C4** — No utility classes; prefer modules.

### Other

- **O1** — Magic numbers named in `raghub/constants.py`.
- **O2** — Single source of truth for SQL schemas in `raghub/store/schema.py`.
- **O3** — Storage format version pinned in a `format_version: int` field (R6).
- **O4** — Every public symbol in `docs/reference/public-api.md` (declared snapshot).
- **O5** — `CHANGELOG.md` updated per release.

## Order of execution

| Week | Items | Theme |
|---|---|---|
| 1 | 1-30 | Tiers 1-5: make v0.7.x features real + CLI |
| 2 | 31-40 | Tiers 6-7: constitution + types |
| 3 | 41-53 | Tiers 8-9: coverage + docs |
| 4 | 54-78 | Tiers 10-14: splits + ADRs + cleanup |
