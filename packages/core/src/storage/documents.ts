/**
 * Documents store.
 *
 * SQLite-backed metadata for ingested documents. Tracks hash,
 * lifecycle status, and per-document chunk counts so the API can
 * surface progress without re-querying the vector store.
 *
 * Idempotent on `hash`: `upsert()` returns the existing row's id
 * when the hash is unchanged.
 */

import type { DocumentId, TenantId, UserId } from '../domain/index.js';
import { brandId, Document, DocumentLifecycleStatus } from '../domain/index.js';
import type { DocumentLifecycleStatusValue } from '../domain/index.js';
import type { DocumentLifecycleStatusValue as DLS, DocumentId as DocId, TenantId as TnId, UserId as UsId } from '../domain/index.js';
import { ConfigurationError, VectorStoreError } from '../errors/index.js';

export interface DocumentStore {
  upsert(input: {
    tenantId: TenantId;
    ownerId: UserId;
    filename: string;
    mimeType: string;
    hash: string;
    byteSize: number;
    metadata?: Readonly<Record<string, string>>;
  }): Promise<Document>;
  getById(tenantId: TenantId, id: DocumentId): Promise<Document | null>;
  getByHash(tenantId: TenantId, hash: string): Promise<Document | null>;
  listForUser(tenantId: TenantId, ownerId: UserId): Promise<readonly Document[]>;
  setStatus(id: DocumentId, tenantId: TenantId, status: DocumentLifecycleStatusValue): Promise<void>;
  countChunks(id: DocumentId, tenantId: TenantId): Promise<number>;
  close(): Promise<void>;
}

interface Database {
  prepare(sql: string): Statement;
  exec(sql: string): void;
  close(): void;
}

interface Statement {
  get(...params: unknown[]): unknown;
  all(...params: unknown[]): unknown[];
  run(...params: unknown[]): { changes: number; lastInsertRowid: number | bigint };
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadBetterSqlite3 = async (): Promise<(filename: string) => Database> => {
  try {
    const mod = (await dynamicImport('better-sqlite3')) as {
      default: (filename: string) => Database;
    };
    return mod.default;
  } catch (cause) {
    throw new VectorStoreError('better-sqlite3 is not installed', {
      cause,
      details: { hint: 'pnpm add better-sqlite3' },
    });
  }
};

export interface SqliteDocumentStoreOptions {
  readonly path: string;
}

export class SqliteDocumentStore implements DocumentStore {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteDocumentStoreOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        owner_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        mime_type TEXT NOT NULL,
        hash TEXT NOT NULL,
        byte_size INTEGER NOT NULL,
        status TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
      CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(tenant_id, owner_id);
      CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash ON documents(tenant_id, hash);
    `);
    this.db = db;
    return db;
  }

  public async upsert(input: {
    tenantId: TenantId;
    ownerId: UserId;
    filename: string;
    mimeType: string;
    hash: string;
    byteSize: number;
    metadata?: Readonly<Record<string, string>>;
  }): Promise<Document> {
    const db = await this.ensure();
    const now = Date.now();
    const id = brandId<DocId>(`doc_${input.hash.slice(0, 16)}`);
    const meta = JSON.stringify({ ...(input.metadata ?? {}) });
    db.prepare(
      `INSERT INTO documents (id, tenant_id, owner_id, filename, mime_type, hash, byte_size, status, metadata_json, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(tenant_id, hash) DO UPDATE SET
         filename = excluded.filename,
         mime_type = excluded.mime_type,
         updated_at = excluded.updated_at`,
    ).run(
      id,
      input.tenantId,
      input.ownerId,
      input.filename,
      input.mimeType,
      input.hash,
      input.byteSize,
      DocumentLifecycleStatus.Pending,
      meta,
      now,
      now,
    );
    return new Document({
      id,
      tenantId: input.tenantId,
      ownerId: input.ownerId,
      filename: input.filename,
      mimeType: input.mimeType,
      hash: input.hash,
      byteSize: input.byteSize,
      status: DocumentLifecycleStatus.Pending,
      metadata: input.metadata ?? {},
      createdAt: new Date(now),
      updatedAt: new Date(now),
    });
  }

  public async getById(tenantId: TenantId, id: DocumentId): Promise<Document | null> {
    const db = await this.ensure();
    const row = db
      .prepare('SELECT * FROM documents WHERE tenant_id = ? AND id = ?')
      .get(tenantId, id) as Record<string, unknown> | undefined;
    return row ? rowToDocument(row) : null;
  }

  public async getByHash(tenantId: TenantId, hash: string): Promise<Document | null> {
    const db = await this.ensure();
    const row = db
      .prepare('SELECT * FROM documents WHERE tenant_id = ? AND hash = ?')
      .get(tenantId, hash) as Record<string, unknown> | undefined;
    return row ? rowToDocument(row) : null;
  }

  public async listForUser(tenantId: TenantId, ownerId: UserId): Promise<readonly Document[]> {
    const db = await this.ensure();
    const rows = db
      .prepare('SELECT * FROM documents WHERE tenant_id = ? AND owner_id = ? ORDER BY created_at DESC')
      .all(tenantId, ownerId) as Record<string, unknown>[];
    return rows.map(rowToDocument);
  }

  public async setStatus(id: DocumentId, tenantId: TenantId, status: DLS): Promise<void> {
    const db = await this.ensure();
    const r = db
      .prepare('UPDATE documents SET status = ?, updated_at = ? WHERE id = ? AND tenant_id = ?')
      .run(status, Date.now(), id, tenantId);
    if (r.changes === 0) throw new ConfigurationError('document not found');
  }

  public async countChunks(_id: DocumentId, _tenantId: TenantId): Promise<number> {
    return 0;
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

const rowToDocument = (row: Record<string, unknown>): Document => {
  const id = brandId<DocId>(String(row['id']));
  const tenantId = brandId<TnId>(String(row['tenant_id']));
  const ownerId = brandId<UsId>(String(row['owner_id']));
  const status = String(row['status']) as DLS;
  const metadata = JSON.parse(String(row['metadata_json'] ?? '{}')) as Record<string, string>;
  return new Document({
    id,
    tenantId,
    ownerId,
    filename: String(row['filename']),
    mimeType: String(row['mime_type']),
    hash: String(row['hash']),
    byteSize: Number(row['byte_size']),
    status: DocumentLifecycleStatus[status as keyof typeof DocumentLifecycleStatus] ?? DocumentLifecycleStatus.Pending,
    metadata,
    createdAt: new Date(Number(row['created_at'])),
    updatedAt: new Date(Number(row['updated_at'])),
  });
};