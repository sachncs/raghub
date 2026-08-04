# Tier 11 — ADRs (Items 61-65)

`AGENTS.md` says design decisions should be recorded as ADRs.
The v0.7.x release notes reference ADRs 0014-0018 but they were
never written. Tier 11 writes them, retrospectively, for each
multi-tenant / pgvector / queue / feedback / archive decision.

Each ADR follows the [MADR](https://adr.github.io/madr/) template
(Context, Decision, Consequences, Alternatives considered).

---

## Item 61 — ADR 0014 — multi-tenant isolation

- **File(s)**: `docs/adr/0014-multi-tenant-isolation.md` (new)
- **Change**: Context (multi-tenant requirement); Decision (three tiers: row-level default, schema-per-tenant for Postgres+pgsql, database-per-tenant for hard isolation); Consequences (operational cost, RLS policies required, per-tenant backups); Alternatives (single-tenant, namespace-per-tenant).
- **Test**: N/A.
- **Acceptance criteria**:
  - Cross-references the v0.7.6 threat model.
  - References the v0.7.6 isolation release.
  - T3 — pass.
- **Success criteria**:
  - File exists in `docs/adr/`.
  - Contains `## Context`, `## Decision`, `## Consequences`, `## Alternatives`.

---

## Item 62 — ADR 0015 — persistent queue

- **File(s)**: `docs/adr/0015-persistent-queue.md` (new)
- **Change**: Context (the need for durable ingestion); Decision (SQLite as default; Celery/Arq as entry-point contracts); Consequences (operational simplicity vs community flexibility); Alternatives (Redis, RabbitMQ).
- **Test**: N/A.
- **Acceptance criteria**:
  - References the v0.7.4 queue release.
  - T3 — pass.
- **Success criteria**:
  - File exists; sections present.

---

## Item 63 — ADR 0016 — feedback loop

- **File(s)**: `docs/adr/0016-feedback-loop.md` (new)
- **Change**: Context (need for user feedback capture); Decision (`bm25-boost` and `vector-down-weight` algorithms, formulas stated); Consequences (no training loop in this release); Alternatives (always-on retraining, RLHF).
- **Test**: N/A.
- **Acceptance criteria**:
  - Formulas explicitly stated.
  - References the v0.7.7 release.
  - T3 — pass.
- **Success criteria**:
  - File exists with explicit formulas.

---

## Item 64 — ADR 0017 — backup format

- **File(s)**: `docs/adr/0017-backup-format.md` (new)
- **Change**: Context (need for portable backups); Decision (deterministic tar + manifest.json + zstd compression, format_version=1, HMAC-SHA256 signed); Consequences (rotation required for `RAGHUB_ARCHIVE_SIGNING_KEY`); Alternatives (encrypted tar, plain zip).
- **Test**: N/A.
- **Acceptance criteria**:
  - References the v0.7.8 archive release.
  - T3 — pass.
- **Success criteria**:
  - File exists with format spec.

---

## Item 65 — ADR 0018 — pgvector as first adapter

- **File(s)**: `docs/adr/0018-pgvector-as-first-adapter.md` (new)
- **Change**: Context (Qdrant default was claimed but never shipped); Decision (pgvector is the recommended production adapter; Qdrant/FAISS/Chroma/Milvus remain pluggable); Consequences (Postgres is now a hard dependency for production); Alternatives (Qdrant first-class, FAISS first-class).
- **Test**: N/A.
- **Acceptance criteria**:
  - References the v0.7.5 release.
  - T3 — pass.
- **Success criteria**:
  - File exists; rationale clear.

---

## Tier 11 acceptance gate

- `ls docs/adr/00{14,15,16,17,18}*.md` returns 5 files.
- Each file has Context, Decision, Consequences, Alternatives sections.
- `mkdocs build --strict` passes.
