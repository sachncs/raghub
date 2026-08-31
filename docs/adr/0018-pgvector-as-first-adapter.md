# ADR 0018 — pgvector as first adapter

## Status

Accepted (v0.7.5)

## Context

The v0.7.x roadmap listed Qdrant as the default vector store, but
Qdrant was never shipped as a first-class integration. The codebase
relied on the in-process SQLite vector store for development and had no
production-grade vector backend.

## Decision

**pgvector** is the recommended production vector adapter, integrated
via the `PgVectorStore` class. Qdrant, FAISS, Chroma, and Milvus
remain pluggable via the `Store` protocol but are not bundled.

Key properties of `PgVectorStore`:

- **Schema**: `CREATE EXTENSION IF NOT EXISTS vector;` followed by a
  `chunks` table with a `vector` column of type `vector(1024)`.
- **Indexes**: IVFFlat (default) or HNSW, configurable via
  `REVEX_PGV_INDEX_TYPE`.
- **Hybrid search**: Combines pgvector cosine distance with BM25
  via the `RetrievalPipeline` scorer.
- **RLS hooks**: Integrates with the multi-tenant isolation model
  (see ADR 0014).

## Consequences

- **Postgres dependency**: Production deployments now require
  PostgreSQL 15+ with the `vector` extension. SQLite remains the
  development default.
- **Operational maturity**: pgvector benefits from the existing
  Postgres backup, replication, and monitoring ecosystem.
- **Pluggability preserved**: The `Store` protocol ensures that
  alternative backends can be swapped in without changing application
  code.

## Alternatives considered

- **Qdrant first-class**: Deferred — Qdrant is a strong option but
  adds a separate service dependency. May be promoted in a future
  release.
- **FAISS first-class**: Rejected — FAISS lacks persistence and
  multi-tenancy primitives; better suited for batch processing.
- **Chroma first-class**: Considered but deferred — Chroma's
  embedding-focused API does not align with Revex's separation of
  concerns.
