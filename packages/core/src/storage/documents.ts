/**
 * Document store — sqlite-backed, shared Database handle.
 *
 * C-03: takes the shared `Database` handle. Schema creation lives in
 * `Workspace.open()`. Idempotent on hash via UNIQUE constraint.
 */

import type { Document, DocumentId, DocumentLifecycleStatusValue, WorkspaceId, UserId } from '../domain/index.js';
import { Document as DocumentClass, DocumentLifecycleStatus } from '../domain/index.js';
import { brandId, type ChunkId, type CollectionId, type WorkspacePlanValue } from '../domain/index.js';
import { ConfigurationError } from '../errors/index.js';
import type { Database } from '../workspace.js';

export interface DocumentStore {
  upsert(input: {
    workspaceId: WorkspaceId;
    ownerId: UserId;
    filename: string;
    mimeType: string;
    hash: string;
    byteSize: number;
    metadata?: Readonly<Record<string, string>>;
  }): Promise<Document>;
  getById(workspaceId: WorkspaceId, id: DocumentId): Promise<Document | null>;
  getByHash(workspaceId: WorkspaceId, hash: string): Promise<Document | null>;
  listForUser(workspaceId: WorkspaceId, ownerId: UserId): Promise<readonly Document[]>;
  setStatus(id: DocumentId, workspaceId: WorkspaceId, status: DocumentLifecycleStatusValue): Promise<void>;
  countChunks(id: DocumentId, workspaceId: WorkspaceId): Promise<number>;
  close(): Promise<void>;
}

export interface SqliteDocumentStoreOptions {
  readonly db: Database;
}

export class SqliteDocumentStore implements DocumentStore {
  private readonly db: Database;

  constructor(opts: SqliteDocumentStoreOptions) {
    this.db = opts.db;
  }

  public async upsert(input: {
    workspaceId: WorkspaceId;
    ownerId: UserId;
    filename: string;
    mimeType: string;
    hash: string;
    byteSize: number;
    metadata?: Readonly<Record<string, string>>;
  }): Promise<Document> {
    const id = brandId<DocumentId>(`doc_${input.hash.slice(0, 16)}`);
    const now = Date.now();
    const meta = JSON.stringify({ ...(input.metadata ?? {}) });
    this.db
      .prepare(
        `INSERT INTO documents (id, workspace_id, owner_id, filename, mime_type, hash, byte_size, status, metadata_json, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(workspace_id, hash) DO UPDATE SET
           filename = excluded.filename,
           mime_type = excluded.mime_type,
           updated_at = excluded.updated_at`,
      )
      .run(
        id,
        input.workspaceId,
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
    return new DocumentClass({
      id,
      workspaceId: input.workspaceId,
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

  public async getById(workspaceId: WorkspaceId, id: DocumentId): Promise<Document | null> {
    const row = this.db
      .prepare('SELECT * FROM documents WHERE workspace_id = ? AND id = ?')
      .get(workspaceId, id) as Record<string, unknown> | undefined;
    return row ? rowToDocument(row) : null;
  }

  public async getByHash(workspaceId: WorkspaceId, hash: string): Promise<Document | null> {
    const row = this.db
      .prepare('SELECT * FROM documents WHERE workspace_id = ? AND hash = ?')
      .get(workspaceId, hash) as Record<string, unknown> | undefined;
    return row ? rowToDocument(row) : null;
  }

  public async listForUser(workspaceId: WorkspaceId, ownerId: UserId): Promise<readonly Document[]> {
    const rows = this.db
      .prepare('SELECT * FROM documents WHERE workspace_id = ? AND owner_id = ? ORDER BY created_at DESC')
      .all(workspaceId, ownerId) as Record<string, unknown>[];
    return rows.map(rowToDocument);
  }

  public async setStatus(id: DocumentId, workspaceId: WorkspaceId, status: DocumentLifecycleStatusValue): Promise<void> {
    const r = this.db
      .prepare('UPDATE documents SET status = ?, updated_at = ? WHERE id = ? AND workspace_id = ?')
      .run(status, Date.now(), id, workspaceId);
    if (r.changes === 0) throw new ConfigurationError('document not found');
  }

  public async countChunks(_id: DocumentId, _workspaceId: WorkspaceId): Promise<number> {
    return 0;
  }

  public async close(): Promise<void> {
    // No-op: db is owned by Workspace.
  }
}

const rowToDocument = (row: Record<string, unknown>): Document => {
  const id = brandId<DocumentId>(String(row['id']));
  const workspaceId = String(row['workspace_id']) as WorkspaceId;
  const ownerId = String(row['owner_id']) as UserId;
  const status = String(row['status']) as DocumentLifecycleStatusValue;
  const metadata = JSON.parse(String(row['metadata_json'] ?? '{}')) as Record<string, string>;
  return new DocumentClass({
    id,
    workspaceId,
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
