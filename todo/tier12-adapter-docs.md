# Tier 12 — Adapter reference docs (Items 66-70)

The v0.7.x release notes reference `docs/adapters/{queue,
feedback, pgvector, isolation, archive}.md` but they were never
written. Tier 12 writes them.

Each file is one PR. Each is independently mergeable.

---

## Item 66 — `docs/adapters/queue.md`

- **File(s)**: `docs/adapters/queue.md` (new), `mkdocs.yml`
- **Change**: Document `SqliteQueue` (constructor, methods, schema, idempotency), `Worker`, `Job`, `JobStatus`, the CLI commands. Cross-link ADR 0015.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5 — docs match shipped behaviour.
  - T3 — pass.
- **Success criteria**:
  - File exists; cross-references resolve.
  - Listed in `mkdocs.yml` `nav:`.

---

## Item 67 — `docs/adapters/pgvector.md`

- **File(s)**: `docs/adapters/pgvector.md` (new), `mkdocs.yml`
- **Change**: Document `PgVectorStore` (constructor, methods, schema, indexes, hybrid search, RLS hooks).
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5.
  - T3 — pass.
- **Success criteria**:
  - File exists; cross-references resolve.

---

## Item 68 — `docs/adapters/feedback.md`

- **File(s)**: `docs/adapters/feedback.md` (new), `mkdocs.yml`
- **Change**: Document `Feedback`, `FeedbackStore`, `FeedbackScorer` algorithms, API endpoints, CLI commands.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5.
  - T3 — pass.
- **Success criteria**:
  - File exists; formulas cited.

---

## Item 69 — `docs/adapters/isolation.md`

- **File(s)**: `docs/adapters/isolation.md` (new), `mkdocs.yml`
- **Change**: Document the three isolation tiers with PostgreSQL examples; per-tenant secrets; migration CLI.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5.
  - T3 — pass.
- **Success criteria**:
  - File exists; all tiers documented.

---

## Item 70 — `docs/adapters/archive.md`

- **File(s)**: `docs/adapters/archive.md` (new), `mkdocs.yml`
- **Change**: Document `ArchiveManifest` (format_version 1), `LocalArchiveStore`, the CLI commands, HMAC signing.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5.
  - T3 — pass.
- **Success criteria**:
  - File exists; format spec included.

---

## Tier 12 acceptance gate

- `ls docs/adapters/` returns 5 files (queue, pgvector, feedback, isolation, archive).
- All 5 listed in `mkdocs.yml` `nav:`.
- `mkdocs build --strict` passes.
