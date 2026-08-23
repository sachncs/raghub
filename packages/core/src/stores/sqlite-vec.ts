/**
 * sqlite-vec store — the only Phase 1 vector store.
 *
 * Two virtual tables back the read path:
 * - `vec_chunks` (sqlite-vec) for cosine-similarity search
 * - `fts_chunks` (SQLite FTS5, BM25) for keyword search
 *
 * The base `chunks` table holds metadata + raw text; the virtual
 * tables reference it by `rowid`/`id`.
 *
 * All read paths require a `StoreFilter` that names a tenant. The
 * filter is enforced in SQL (`WHERE workspace_id = ?`); userId is
 * optional and treated as "any user in tenant" when `null` (admin
 * only). RBAC on `metadata.company` is enforced via a `json_extract`
 * predicate.
 */

import type { Chunk, Hit } from '../domain/index.js';
import type {
  ChunkId,
  CollectionId,
  DocumentId,
  WorkspaceId,
  UserId,
} from '../domain/index.js';
import { brandId, Chunk as ChunkClass } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';
import type {
  KeywordHit,
  KeywordSearchOptions,
  VectorSearchOptions,
  VectorStore,
} from './types.js';

const EMBEDDING_DIM_DEFAULT = 3072;
const FTS_TABLE = 'fts_chunks';
const VEC_TABLE = 'vec_chunks';
const CHUNKS_TABLE = 'chunks';

interface BetterSqliteDatabase {
  prepare(sql: string): BetterSqliteStatement;
  exec(sql: string): void;
  pragma(source: string): unknown;
  close(): void;
  loadExtension(path: string): void;
  transaction<T extends (...args: never[]) => unknown>(fn: T): T;
}

interface BetterSqliteStatement {
  all(...params: unknown[]): unknown[];
  get(...params: unknown[]): unknown;
  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint };
}

interface BetterSqlite3Module {
  (filename: string): BetterSqliteDatabase;
}

interface SqliteVecModule {
  load(db: BetterSqliteDatabase): void;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadBetterSqlite3 = async (): Promise<BetterSqlite3Module> => {
  try {
    const mod = (await dynamicImport('better-sqlite3')) as { default: BetterSqlite3Module };
    return mod.default;
  } catch (cause) {
    throw new VectorStoreError('better-sqlite3 is not installed', {
      cause,
      details: { hint: 'pnpm add better-sqlite3' },
    });
  }
};

const loadSqliteVec = async (): Promise<SqliteVecModule> => {
  try {
    const mod = (await dynamicImport('@sqlite.org/sqlite-vec')) as {
      default: SqliteVecModule;
    };
    return mod.default;
  } catch (cause) {
    throw new VectorStoreError('@sqlite.org/sqlite-vec is not installed', {
      cause,
      details: { hint: 'pnpm add @sqlite.org/sqlite-vec' },
    });
  }
};

const rowToChunk = (row: Record<string, unknown>): Chunk => {
  const id = brandId<ChunkId>(String(row['id']));
  const workspaceId = brandId<WorkspaceId>(String(row['workspace_id']));
  const ownerId = brandId<UserId>(String(row['owner_id']));
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
  readonly path: string;
  readonly embeddingDim?: number;
}

export class SqliteVecStore implements VectorStore {
  private db: BetterSqliteDatabase | null = null;
  private readonly path: string;
  private readonly dim: number;
  private initPromise: Promise<void> | null = null;

  constructor(opts: SqliteVecStoreOptions) {
    this.path = opts.path;
    this.dim = opts.embeddingDim ?? EMBEDDING_DIM_DEFAULT;
  }

  private async ensureInit(): Promise<BetterSqliteDatabase> {
    if (this.db) return this.db;
    if (!this.initPromise) {
      this.initPromise = (async () => {
        const sqlite = await loadBetterSqlite3();
        const vec = await loadSqliteVec();
        const db = sqlite(this.path);
        db.pragma('journal_mode = WAL');
        db.pragma('foreign_keys = ON');
        vec.load(db);
        this.bootstrapSchema(db);
        this.db = db;
      })();
    }
    await this.initPromise;
    if (!this.db) throw new VectorStoreError('sqlite-vec store failed to initialise');
    return this.db;
  }

  private bootstrapSchema(db: BetterSqliteDatabase): void {
    db.exec(`
      CREATE TABLE IF NOT EXISTS ${CHUNKS_TABLE} (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        collection_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        modality TEXT NOT NULL DEFAULT 'text',
        text TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON ${CHUNKS_TABLE}(workspace_id);
      CREATE INDEX IF NOT EXISTS idx_chunks_owner ON ${CHUNKS_TABLE}(workspace_id, owner_id);
      CREATE INDEX IF NOT EXISTS idx_chunks_document ON ${CHUNKS_TABLE}(document_id);

      CREATE VIRTUAL TABLE IF NOT EXISTS ${VEC_TABLE} USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[${this.dim}]
      );

      CREATE VIRTUAL TABLE IF NOT EXISTS ${FTS_TABLE} USING fts5(
        text,
        content='${CHUNKS_TABLE}',
        content_rowid='rowid',
        tokenize='porter unicode61'
      );
    `);
  }

  public async add(chunk: Chunk): Promise<void> {
    const db = await this.ensureInit();
    const tx = db.transaction((c: Chunk) => {
      const now = Date.now();
      const metadataJson = JSON.stringify(c.metadata);
      db.prepare(
        `INSERT OR REPLACE INTO ${CHUNKS_TABLE}
         (id, workspace_id, owner_id, collection_id, document_id, modality, text, token_count, metadata_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        c.id,
        c.workspaceId,
        c.ownerId,
        c.collectionId,
        c.documentId,
        c.modality,
        c.text,
        c.tokenCount,
        metadataJson,
        now,
      );
      const embedBlob = new Float32Array(c.embedding);
      db.prepare(`INSERT OR REPLACE INTO ${VEC_TABLE} (id, embedding) VALUES (?, ?)`).run(
        c.id,
        embedBlob,
      );
      db.prepare(
        `INSERT INTO ${FTS_TABLE} (rowid, text) VALUES ((SELECT rowid FROM ${CHUNKS_TABLE} WHERE id = ?), ?)`,
      ).run(c.id, c.text);
    });
    tx(chunk);
  }

  public async addBatch(chunks: readonly Chunk[]): Promise<void> {
    for (const c of chunks) await this.add(c);
  }

  public async searchVector(opts: VectorSearchOptions): Promise<Hit[]> {
    const db = await this.ensureInit();
    if (opts.vector.length !== this.dim) {
      throw new VectorStoreError(
        `vector dimension ${opts.vector.length} != store dim ${this.dim}`,
        { details: { expected: this.dim, got: opts.vector.length } },
      );
    }
    const f = opts.filter;
    const userClause = f.userId ? 'AND c.owner_id = ?' : '';
    const collClause = f.collectionId ? 'AND c.collection_id = ?' : '';
    const rbacClause = f.allowedCompanies.length
      ? `AND (json_extract(c.metadata_json, '$.company') IS NULL OR json_extract(c.metadata_json, '$.company') IN (${f.allowedCompanies.map(() => '?').join(',')}))`
      : '';
    const sql = `
      SELECT v.id AS id, v.distance AS distance, c.*
      FROM ${VEC_TABLE} v
      JOIN ${CHUNKS_TABLE} c ON c.id = v.id
      WHERE c.workspace_id = ?
        ${userClause}
        ${collClause}
        ${rbacClause}
      ORDER BY v.distance ASC
      LIMIT ?
    `;
    const params: unknown[] = [f.workspaceId];
    if (f.userId) params.push(f.userId);
    if (f.collectionId) params.push(f.collectionId);
    if (f.allowedCompanies.length) params.push(...f.allowedCompanies);
    params.push(opts.topK);

    const rows = db.prepare(sql).all(...params) as Record<string, unknown>[];
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
    const db = await this.ensureInit();
    const f = opts.filter;
    const userClause = f.userId ? 'AND c.owner_id = ?' : '';
    const collClause = f.collectionId ? 'AND c.collection_id = ?' : '';
    const rbacClause = f.allowedCompanies.length
      ? `AND (json_extract(c.metadata_json, '$.company') IS NULL OR json_extract(c.metadata_json, '$.company') IN (${f.allowedCompanies.map(() => '?').join(',')}))`
      : '';
    const sql = `
      SELECT c.id AS id, c.text AS text, bm25(${FTS_TABLE}) AS rank
      FROM ${FTS_TABLE}
      JOIN ${CHUNKS_TABLE} c ON c.rowid = ${FTS_TABLE}.rowid
      WHERE ${FTS_TABLE} MATCH ?
        AND c.workspace_id = ?
        ${userClause}
        ${collClause}
        ${rbacClause}
      ORDER BY rank ASC
      LIMIT ?
    `.trim();
    const params: unknown[] = [opts.query, f.workspaceId];
    if (f.userId) params.push(f.userId);
    if (f.collectionId) params.push(f.collectionId);
    if (f.allowedCompanies.length) params.push(...f.allowedCompanies);
    params.push(opts.topK);

    const rows = db.prepare(sql).all(...params) as Record<string, unknown>[];
    return rows.map((row) => ({
      chunkId: brandId<ChunkId>(String(row['id'])),
      score: 1 / (1 + Number(row['rank'])),
      text: String(row['text']),
    }));
  }

  public async getById(
    workspaceId: WorkspaceId,
    id: ChunkId,
  ): Promise<Chunk | null> {
    const db = await this.ensureInit();
    const row = db
      .prepare(`SELECT * FROM ${CHUNKS_TABLE} WHERE workspace_id = ? AND id = ?`)
      .get(workspaceId, id) as Record<string, unknown> | undefined;
    return row ? rowToChunk(row) : null;
  }

  public async deleteByDocument(
    documentId: DocumentId,
    workspaceId: WorkspaceId,
  ): Promise<number> {
    const db = await this.ensureInit();
    const tx = db.transaction(() => {
      const chunks = db
        .prepare(`SELECT id FROM ${CHUNKS_TABLE} WHERE document_id = ? AND workspace_id = ?`)
        .all(documentId, workspaceId) as Record<string, unknown>[];
      let n = 0;
      for (const row of chunks) {
        const cid = String(row['id']);
        db.prepare(`DELETE FROM ${VEC_TABLE} WHERE id = ?`).run(cid);
        db.prepare(
          `DELETE FROM ${FTS_TABLE} WHERE rowid = (SELECT rowid FROM ${CHUNKS_TABLE} WHERE id = ?)`,
        ).run(cid);
        n += db.prepare(`DELETE FROM ${CHUNKS_TABLE} WHERE id = ?`).run(cid).changes;
      }
      return n;
    });
    return tx() as number;
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
      this.initPromise = null;
    }
  }
}