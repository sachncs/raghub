/**
 * pgvector vector store adapter (stub).
 *
 * First-class adapter for Postgres + pgvector. The base package
 * ships a stub that throws on every call so consumers see the
 * missing-dep error up front. Production deployments add
 * `pg` + `@pgvector/pg` and use this implementation.
 *
 * Schema:
 *   CREATE EXTENSION IF NOT EXISTS vector;
 *   CREATE TABLE IF NOT EXISTS revex_chunks (
 *     id TEXT PRIMARY KEY,
 *     workspace_id TEXT NOT NULL,
 *     owner_id TEXT NOT NULL,
 *     document_id TEXT NOT NULL,
 *     embedding VECTOR(768) NOT NULL,
 *     ...
 *   );
 */

import { ConfigurationError, VectorStoreError } from '../index.js';

export interface PgVectorOptions {
  readonly connectionString: string;
  readonly tableName?: string;
  readonly dim?: number;
}

export class PgVectorStore {
  readonly name = 'pgvector';

  constructor(_opts: PgVectorOptions) {
    throw new ConfigurationError(
      'PgVectorStore is a stub; install `pg` and `@pgvector/pg` and wire a real adapter.',
    );
  }
}

export const SCHEMA_SQL = (table: string, dim: number): string => `
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS ${table} (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    modality TEXT NOT NULL DEFAULT 'text',
    text TEXT NOT NULL,
    embedding VECTOR(${dim}) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );
  CREATE INDEX IF NOT EXISTS ${table}_ws ON ${table} (workspace_id);
  CREATE INDEX IF NOT EXISTS ${table}_vec ON ${table} USING ivfflat (embedding vector_cosine_ops);
`;

export const EMBED_DIM_DEFAULT = 768;

void VectorStoreError;