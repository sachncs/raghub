# Tier 9 — Documentation (Items 47-53)

The v0.7.x → v0.8.0 work shipped docs/operations/{backup,
monitoring, scaling, runbook}.md and docs/guide/deployment.md that
still reference Docker / docker-compose, even though the locked
decision was "no Docker artefacts." Tier 9 cleans those up and
updates the high-level docs for v0.7.x features.

`AGENTS.md` rule O5 — `CHANGELOG.md` updated per release.

---

## Item 47 — Update `docs/ADVANCED_RAG.md`

- **File(s)**: `docs/ADVANCED_RAG.md`
- **Change**: Add sections for `SqliteQueue`, `PgVectorStore`, isolation tiers, `FeedbackStore` + scorers, `ArchiveStore`. Cross-reference the v0.7.x ADRs.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5 — docs match shipped behaviour.
  - T3 — pass.
- **Success criteria**:
  - Each v0.7.x feature has at least one subsection.
  - All cross-references resolve.

---

## Item 48 — Clean Docker refs from `docs/operations/backup.md`

- **File(s)**: `docs/operations/backup.md`
- **Change**: Replace `docker compose -f docker-compose.yml --profile production up -d` with the v0.7.8 archive CLI workflow.
- **Test**: N/A.
- **Acceptance criteria**:
  - O5 — docs match shipped behaviour.
- **Success criteria**:
  - `grep -i docker docs/operations/backup.md` returns empty.

---

## Item 49 — Clean Docker refs from `docs/operations/monitoring.md`

- **File(s)**: `docs/operations/monitoring.md`
- **Change**: Same as item 48.
- **Test**: N/A.
- **Acceptance criteria**:
  - O5 — docs match shipped behaviour.
- **Success criteria**:
  - `grep -i docker docs/operations/monitoring.md` returns empty.

---

## Item 50 — Clean Docker refs from `docs/operations/scaling.md`

- **File(s)**: `docs/operations/scaling.md`
- **Change**: Same.
- **Test**: N/A.
- **Acceptance criteria**:
  - O5 — docs match shipped behaviour.
- **Success criteria**:
  - `grep -i docker docs/operations/scaling.md` returns empty.

---

## Item 51 — Update `docs/guide/deployment.md`

- **File(s)**: `docs/guide/deployment.md`
- **Change**: Drop Docker content. Link to the v0.7.x operational docs.
- **Test**: N/A.
- **Acceptance criteria**:
  - O5.
- **Success criteria**:
  - `grep -i docker docs/guide/deployment.md` returns empty.

---

## Item 52 — Update `README.md` to remove Qdrant claim

- **File(s)**: `README.md`
- **Change**: Replace the "Qdrant" claim with "pgvector (recommended) / SQLite (in-process)". Describe the actual shipped story.
- **Test**: N/A.
- **Acceptance criteria**:
  - O5.
- **Success criteria**:
  - `grep -i "Qdrant" README.md` returns empty (or matches only in CHANGELOG historical entries).

---

## Item 53 — Update `docs/api.md` for `/v1/feedback/*` + rate-limit headers

- **File(s)**: `docs/api.md`
- **Change**: Document the new feedback endpoints; document `Retry-After`, `X-RateLimit-Remaining`, `X-RateLimit-Limit` headers; document queue CLI sub-commands.
- **Test**: `mkdocs build --strict` passes.
- **Acceptance criteria**:
  - O5.
- **Success criteria**:
  - Sections present; cross-references resolve.

---

## Tier 9 acceptance gate

- `grep -ri docker docs/` returns empty.
- `grep -i "Qdrant" README.md` returns empty.
- `mkdocs build --strict` passes.
