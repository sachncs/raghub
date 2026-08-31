/**
 * Vector store contract.
 *
 * All retrieval reads go through this interface. The Phase 1
 * implementation is `SqliteVecStore` (sqlite-vec extension + FTS5).
 * Per-workspace, per-user, per-collection, and per-document-ACL
 * filtering is enforced here, not at the caller.
 */

import type { Chunk, Hit } from '../domain/index.js';
import type { ChunkId, DocumentId, WorkspaceId, UserId } from '../domain/index.js';

export interface Principal {
  readonly type: 'user' | 'role' | 'group';
  readonly id: string;
}

export interface StoreFilter {
  readonly workspaceId: WorkspaceId;
  /** When `null`, the store returns rows owned by any user in the workspace (admin only). */
  readonly userId: UserId | null;
  readonly collectionId: string | null;
  /** ACL principals (resolved user / role / group IDs) that grant document access. */
  readonly principals: readonly Principal[];
  /** RBAC: only chunks whose metadata.company ∈ allowedCompanies. Empty array disables the filter. */
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

  /** Fetch a chunk by id, scoped by workspace. */
  getById(workspaceId: WorkspaceId, id: ChunkId): Promise<Chunk | null>;

  /** Cascade-delete every chunk belonging to `documentId`. */
  deleteByDocument(documentId: DocumentId, workspaceId: WorkspaceId): Promise<number>;

  /**
   * Aggregate statistics for a workspace. Used by the memory
   * dashboard. Concrete stores compute this however they like;
   * the default impl returns zeros so non-SQLite stores stay
   * compatible.
   */
  stats(workspaceId: WorkspaceId): Promise<StoreStats>;

  /** Close the underlying handle. Idempotent. */
  close(): Promise<void>;
}

export interface StoreStats {
  readonly documentCount: number;
  readonly chunkCount: number;
  readonly embeddingBytes: number;
  readonly totalTokenEstimate: number;
  readonly bytesOnDisk: number;
  readonly lastIngestedAt: number | null;
  readonly statusCounts: Readonly<Record<string, number>>;
}
