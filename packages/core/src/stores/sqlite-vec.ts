/**
 * sqlite-vec store — the only Phase 1 vector store.
 *
 * Two virtual tables back the read path:
 * - `vec_chunks` (sqlite-vec) for cosine-similarity search
 * - `fts_chunks` (SQLite FTS5, BM25) for keyword search
 *
 * The base `chunks` table holds metadata + raw text; the virtual
 * tables reference it by id.
 *
 * C-03: takes the shared `Database` handle. Schema creation lives in
 * `Workspace.open()` — this file only owns the SQL operations.
 * ACL filter (document_principal) is wired in here.
 */

import type { Chunk, ChunkId, CollectionId, DocumentId, Hit, WorkspaceId } from '../domain/index.js';
import { Chunk as ChunkClass } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';
import type { Database } from '../workspace.js';
import type {
  KeywordHit,
  KeywordSearchOptions,
  Principal,
  VectorSearchOptions,
  VectorStore,
} from './types.js';

const FTS_TABLE = 'fts_chunks';
const VEC_TABLE = 'vec_chunks';
const CHUNKS_TABLE = 'chunks';

const rowToChunk = (row: Record<string, unknown>): Chunk => {
  const id = brandId<ChunkId>(String(row['id']));
  const workspaceId = brandId<WorkspaceId>(String(row['workspace_id']));
  const ownerId = brandId<import('../domain/index.js').UserId>(String(row['owner_id']));
  const collectionId = brandId<CollectionId>(String(row['collection_id']));
  const documentId = brandId<DocumentId>(String(row['document_id']));
  const modality = (String(row['modality']) ?? 'text') as Chunk['modality'];
  const metadataJson = String(row['metadata_json'] ?? '{}');
  const metadata = JSON.parse(metadataJson) as Record<string, string>;
  return new ChunkClass({
    id,
    workspaceId,
    ownerId,
    collectionId,
    documentId,
    modality,
    text: String(row['text']),
    embedding: [],
    metadata,
    tokenCount: Number(row['token_count'] ?? 0),
    createdAt: new Date(Number(row['created_at'])),
  });
};

export interface SqliteVecStoreOptions {
  readonly db: Database;
  readonly embeddingDim: number;
}

export class SqliteVecStore implements VectorStore {
  private readonly db: Database;
  private readonly dim: number;

  constructor(opts: SqliteVecStoreOptions) {
    this.db = opts.db;
    this.dim = opts.embeddingDim;
  }

  public async add(chunk: Chunk): Promise<void> {
    const tx = this.db.transaction
      ? this.db.transaction(() => this.insertOne(chunk))
      : () => this.insertOne(chunk);
    (tx as () => void)();
  }

  private insertOne(chunk: Chunk): void {
    const now = Date.now();
    const metadataJson = JSON.stringify(chunk.metadata);
    this.db
      .prepare(
        `INSERT OR REPLACE INTO ${CHUNKS_TABLE}
         (id, workspace_id, owner_id, collection_id, document_id, modality, text, token_count, metadata_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        chunk.id,
        chunk.workspaceId,
        chunk.ownerId,
        chunk.collectionId,
        chunk.documentId,
        chunk.modality,
        chunk.text,
        chunk.tokenCount,
        metadataJson,
        now,
      );
    const embedBlob = new Float32Array(chunk.embedding);
    this.db
      .prepare(`INSERT OR REPLACE INTO ${VEC_TABLE} (id, embedding) VALUES (?, ?)`)
      .run(chunk.id, embedBlob);
    this.db
      .prepare(
        `INSERT INTO ${FTS_TABLE} (rowid, text) VALUES ((SELECT rowid FROM ${CHUNKS_TABLE} WHERE id = ?), ?)`,
      )
      .run(chunk.id, chunk.text);
  }

  public async addBatch(chunks: readonly Chunk[]): Promise<void> {
    for (const c of chunks) await this.add(c);
  }

  public async searchVector(opts: VectorSearchOptions): Promise<Hit[]> {
    if (opts.vector.length !== this.dim) {
      throw new VectorStoreError(
        `vector dimension ${opts.vector.length} != store dim ${this.dim}`,
        { details: { expected: this.dim, got: opts.vector.length } },
      );
    }
    const f = opts.filter;
    const { clause: principalsClause, params: principalParams } = buildAclClause(f.principals);
    const userClause = f.userId ? 'AND c.owner_id = ?' : '';
    const collClause = f.collectionId ? 'AND c.collection_id = ?' : '';
    const sql = `
      SELECT v.id AS id, v.distance AS distance, c.*
      FROM ${VEC_TABLE} v
      JOIN ${CHUNKS_TABLE} c ON c.id = v.id
      WHERE c.workspace_id = ?
        ${userClause}
        ${collClause}
        ${principalsClause}
      ORDER BY v.distance ASC
      LIMIT ?
    `;
    const params: unknown[] = [f.workspaceId, ...principalParams];
    if (f.userId) params.push(f.userId);
    if (f.collectionId) params.push(f.collectionId);
    params.push(opts.topK);

    const rows = this.db.prepare(sql).all(...params) as Record<string, unknown>[];
    const hits: Hit[] = [];
    for (const row of rows) {
      const score = 1 - Number(row['distance']);
      if (opts.minScore !== undefined && score < opts.minScore) break;
      const c = rowToChunk(row);
      hits.push({
        chunk: new ChunkClass({ ...c.toJSON(), embedding: [...opts.vector] }),
        score,
      });
    }
    return hits;
  }

  public async searchKeyword(opts: KeywordSearchOptions): Promise<KeywordHit[]> {
    const f = opts.filter;
    const { clause: principalsClause, params: principalParams } = buildAclClause(f.principals);
    const userClause = f.userId ? 'AND c.owner_id = ?' : '';
    const collClause = f.collectionId ? 'AND c.collection_id = ?' : '';
    const sql = `
      SELECT c.id AS id, c.text AS text, bm25(${FTS_TABLE}) AS rank
      FROM ${FTS_TABLE}
      JOIN ${CHUNKS_TABLE} c ON c.rowid = ${FTS_TABLE}.rowid
      WHERE ${FTS_TABLE} MATCH ?
        AND c.workspace_id = ?
        ${userClause}
        ${collClause}
        ${principalsClause}
      ORDER BY rank ASC
      LIMIT ?
    `.trim();
    const params: unknown[] = [opts.query, f.workspaceId, ...principalParams];
    if (f.userId) params.push(f.userId);
    if (f.collectionId) params.push(f.collectionId);
    params.push(opts.topK);

    const rows = this.db.prepare(sql).all(...params) as Record<string, unknown>[];
    return rows.map((row) => ({
      chunkId: brandId<ChunkId>(String(row['id'])),
      score: 1 / (1 + Number(row['rank'])),
      text: String(row['text']),
    }));
  }

  public async getById(workspaceId: WorkspaceId, id: ChunkId): Promise<Chunk | null> {
    const row = this.db
      .prepare(`SELECT * FROM ${CHUNKS_TABLE} WHERE workspace_id = ? AND id = ?`)
      .get(workspaceId, id) as Record<string, unknown> | undefined;
    return row ? rowToChunk(row) : null;
  }

  public async deleteByDocument(documentId: DocumentId, workspaceId: WorkspaceId): Promise<number> {
    const tx = this.db.transaction
      ? this.db.transaction(() => this.deleteChunksForDocument(documentId, workspaceId))
      : () => this.deleteChunksForDocument(documentId, workspaceId);
    return (tx as () => number)();
  }

  private deleteChunksForDocument(documentId: DocumentId, workspaceId: WorkspaceId): number {
    const chunks = this.db
      .prepare(`SELECT id FROM ${CHUNKS_TABLE} WHERE document_id = ? AND workspace_id = ?`)
      .all(documentId, workspaceId) as Record<string, unknown>[];
    let n = 0;
    for (const row of chunks) {
      const cid = String(row['id']);
      this.db.prepare(`DELETE FROM ${VEC_TABLE} WHERE id = ?`).run(cid);
      this.db
        .prepare(`DELETE FROM ${FTS_TABLE} WHERE rowid = (SELECT rowid FROM ${CHUNKS_TABLE} WHERE id = ?)`)
        .run(cid);
      n += this.db.prepare(`DELETE FROM ${CHUNKS_TABLE} WHERE id = ?`).run(cid).changes;
    }
    return n;
  }

  public async close(): Promise<void> {
    // No-op.
  }
}

interface AclClause {
  readonly clause: string;
  readonly params: readonly unknown[];
}

/**
 * Build the SQL fragment that joins `document_principal` to filter
 * chunks the active user's principals can see. A chunk is reachable
 * if any of the user's principals (user, role, or group) has `read`
 * or `admin` on its parent document.
 *
 * Returns { clause, params } so callers can append with `?` and
 * pass the params into the prepared statement.
 */
const buildAclClause = (principals: readonly Principal[] | undefined): AclClause => {
  if (!principals || principals.length === 0) return { clause: '', params: [] };
  const conds: string[] = [];
  const params: unknown[] = [];
  for (const p of principals) {
    conds.push('(dp.principal_type = ? AND dp.principal_id = ?)');
    params.push(p.type, p.id);
  }
  return {
    clause: `AND EXISTS (
      SELECT 1 FROM document_principal dp
      WHERE dp.document_id = c.document_id
        AND (${conds.join(' OR ')})
        AND dp.permission IN ('read', 'admin')
    )`,
    params,
  };
};
