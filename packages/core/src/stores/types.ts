/**
 * Vector store contract.
 *
 * All retrieval reads go through this interface. The Phase 1
 * implementation is `SqliteVecStore` (sqlite-vec extension + FTS5).
 * Multi-tenant and per-user filtering is enforced here, not at the
 * caller — every method requires a `StoreFilter`.
 */

import type { Chunk, Hit } from '../domain/index.js';
import type { ChunkId, DocumentId, TenantId, UserId } from '../domain/index.js';

export interface StoreFilter {
  readonly tenantId: TenantId;
  /** When `null`, the store returns rows owned by any user in the tenant. */
  readonly userId: UserId | null;
  readonly collectionId: string | null;
  /** RBAC: only chunks whose metadata.company ∈ allowedCompanies. */
  readonly allowedCompanies: readonly string[];
}

export interface VectorSearchOptions {
  readonly vector: readonly number[];
  readonly topK: number;
  readonly filter: StoreFilter;
  /** Optional minimum cosine similarity; results below are dropped. */
  readonly minScore?: number;
}

export interface KeywordSearchOptions {
  readonly query: string;
  readonly topK: number;
  readonly filter: StoreFilter;
}

/**
 * Plain text search hit from FTS5.
 */
export interface KeywordHit {
  readonly chunkId: ChunkId;
  readonly score: number;
  readonly text: string;
}

export interface VectorStore {
  /** Persist a chunk. Idempotent on `chunk.id`. */
  add(chunk: Chunk): Promise<void>;

  /** Bulk persist; default impl just loops; concrete stores may batch. */
  addBatch(chunks: readonly Chunk[]): Promise<void>;

  /** Cosine-similarity search; returns hits in score-desc order. */
  searchVector(opts: VectorSearchOptions): Promise<Hit[]>;

  /** BM25 keyword search via SQLite FTS5; returns hits in score-desc order. */
  searchKeyword(opts: KeywordSearchOptions): Promise<KeywordHit[]>;

  /** Fetch a chunk by id, scoped by tenant. */
  getById(tenantId: TenantId, id: ChunkId): Promise<Chunk | null>;

  /** Cascade-delete every chunk belonging to `documentId`. */
  deleteByDocument(documentId: DocumentId, tenantId: TenantId): Promise<number>;

  /** Close the underlying handle. Idempotent. */
  close(): Promise<void>;
}