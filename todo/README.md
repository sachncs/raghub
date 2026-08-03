# RAGHub Release Plan — v0.7.x → v0.8.0

This folder is the **acceptance criteria** for each release. Every file
in this directory is a release spec; an item is "done" only when every
Success Criterion and every Validation Criterion is met, and every
checkbox in the Acceptance gate is ticked.

## Forward-only development

Per `AGENTS.md` rule R4 and the explicit project policy confirmed by
the maintainer: **no backward compatibility, no aliases, no shims, no
deprecation period.** Old names simply stop existing after the release
that removes them. `CHANGELOG.md` is updated as part of every release;
there are no upgrade notes because there is nothing to upgrade from.

## Release index

| Release | Theme | File |
|---|---|---|
| v0.7.0 | Correctness + Prometheus removal + Langfuse core | [`v0.7.0-correctness-and-prometheus-removal.md`](v0.7.0-correctness-and-prometheus-removal.md) |
| v0.7.1 | Naming, constants, schema SoT | [`v0.7.1-naming-and-constants.md`](v0.7.1-naming-and-constants.md) |
| v0.7.2 | Type discipline, `Any` reduction, dataclasses | [`v0.7.2-type-discipline.md`](v0.7.2-type-discipline.md) |
| v0.7.3 | TenantResolver + per-tenant rate limiting | [`v0.7.3-tenant-resolver-and-rate-limiting.md`](v0.7.3-tenant-resolver-and-rate-limiting.md) |
| v0.7.4 | Persistent ingestion queue | [`v0.7.4-persistent-ingestion-queue.md`](v0.7.4-persistent-ingestion-queue.md) |
| v0.7.5 | pgvector adapter | [`v0.7.5-pgvector-adapter.md`](v0.7.5-pgvector-adapter.md) |
| v0.7.6 | Multi-tenant isolation (3 strategies) | [`v0.7.6-multi-tenant-isolation.md`](v0.7.6-multi-tenant-isolation.md) |
| v0.7.7 | Feedback loops | [`v0.7.7-feedback-loops.md`](v0.7.7-feedback-loops.md) |
| v0.7.8 | Backup / restore | [`v0.7.8-backup-restore.md`](v0.7.8-backup-restore.md) |
| v0.8.0 | Tests, docs, API surface, hygiene | [`v0.8.0-tests-docs-api.md`](v0.8.0-tests-docs-api.md) |

Total: **109 atomic items** across **10 releases**.

## Per-release file structure

Every release file contains the following sections, in order:

1. **Theme** — one-line summary of the release.
2. **Breaking changes** — explicit list (R4 forward-only).
3. **Prerequisites** — what must already be released for this release to ship.
4. **Atomic items** — numbered list of PRs; each PR is independently mergeable.
5. **Doc updates owned by this release** — files added or modified.
6. **CHANGELOG entry** — block to paste into `CHANGELOG.md`.
7. **Test baseline** — coverage floor for new code; overall floor.
8. **Success criteria** — qualitative outcomes.
9. **Validation criteria** — concrete commands / outputs that must pass.
10. **Security review** — threat model + mitigations (where applicable).
11. **Acceptance gate** — checklist of "done" conditions.

## CHANGELOG template

Each release uses the same CHANGELOG template; see
[`CHANGELOG.template.md`](CHANGELOG.template.md). The release file
includes a pre-filled block to paste into `CHANGELOG.md` as the final
acceptance step.

## Conventions

- Each atomic item fits in **one PR** (≤ 500 LOC for the bulk; the
  god-object splits in v0.7.0 item 24 and the schema SoT in v0.7.1
  item 38 are flagged exceptions and may exceed).
- Each atomic item has **at least one regression test**.
- Each atomic item updates `CHANGELOG.md` *if and only if* it is the
  final item in the release; intermediate items do not edit CHANGELOG.
- Each release has **one designated release captain** who runs the
  Validation criteria end-to-end before tagging.

## How to use this folder

1. Pick the next release file.
2. Open it end-to-end.
3. For each atomic item, open a PR; reference the item number in the
   PR title (`[v0.7.0] #3: ChunkRef.__init__ annotation`).
4. After all items land, run every Validation Criterion command.
5. Paste the pre-filled CHANGELOG block into `CHANGELOG.md`.
6. Tick the Acceptance gate.
7. Tag and release: `git tag v0.7.x && git push origin v0.7.x`.

## What this plan is NOT

- It is **not** a replacement for `AGENTS.md`. The constitution in
  `AGENTS.md` still governs.
- It is **not** a roadmap in the marketing sense. It is an
  acceptance-criteria document for engineering work.
- It is **not** a commitment to ship dates. Each release ships when
  every item below its heading is done.
