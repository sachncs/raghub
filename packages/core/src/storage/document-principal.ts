/**
 * Document ACL — per-document principal grants.
 *
 * `document_principal(document_id, principal_type, principal_id, permission)`
 * is the join table between documents and the principals (user /
 * role / group) that may see them. Each (document_id, principal_type,
 * principal_id, permission) tuple is unique — re-granting the same
 * permission is a no-op via `INSERT OR IGNORE`. To revoke a permission,
 * delete the row.
 *
 * Default ACL on document upload is applied by the ingest pipeline:
 *   - (doc, 'user', owner_id, 'admin')
 *   - (doc, 'user', admin_id, 'admin') for each workspace admin
 *
 * The retrieval filter (in `SqliteVecStore`) joins
 * `document_principal` to deny chunks the active user's resolved
 * principals don't reach.
 */

import type { Database } from '../workspace.js';
import type { DocumentId, UserId } from '../domain/index.js';

export const DocumentPermission = {
  Read: 'read',
  Admin: 'admin',
} as const;

export type DocumentPermissionValue =
  (typeof DocumentPermission)[keyof typeof DocumentPermission];

export type DocumentPrincipalType = 'user' | 'role' | 'group';

export interface DocumentPrincipal {
  readonly documentId: DocumentId;
  readonly principalType: DocumentPrincipalType;
  readonly principalId: string;
  readonly permission: DocumentPermissionValue;
  readonly grantedBy: UserId;
  readonly grantedAt: Date;
}

export interface GrantInput {
  readonly documentId: DocumentId;
  readonly principalType: DocumentPrincipalType;
  readonly principalId: string;
  readonly permission: DocumentPermissionValue;
  readonly grantedBy: UserId;
}

export interface DocumentPrincipalStore {
  grant(input: GrantInput): Promise<void>;
  revoke(input: Omit<GrantInput, 'grantedBy'>): Promise<void>;
  listByDocument(documentId: DocumentId): Promise<readonly DocumentPrincipal[]>;
  listByPrincipal(
    principalType: DocumentPrincipalType,
    principalId: string,
  ): Promise<readonly DocumentPrincipal[]>;
  hasAccess(
    documentId: DocumentId,
    principals: readonly { type: DocumentPrincipalType; id: string }[],
  ): Promise<boolean>;
  applyDefaultAcl(input: {
    documentId: DocumentId;
    ownerId: UserId;
    adminUserIds: readonly UserId[];
  }): Promise<void>;
  close(): Promise<void>;
}

export interface SqliteDocumentPrincipalStoreOptions {
  readonly db: Database;
}

export interface PrincipalRef {
  readonly type: DocumentPrincipalType;
  readonly id: string;
}

const rowToPrincipal = (row: Record<string, unknown>): DocumentPrincipal => ({
  documentId: String(row['document_id']) as DocumentId,
  principalType: row['principal_type'] as DocumentPrincipalType,
  principalId: String(row['principal_id']),
  permission: String(row['permission']) as DocumentPermissionValue,
  grantedBy: String(row['granted_by']) as UserId,
  grantedAt: new Date(Number(row['granted_at'])),
});

export class SqliteDocumentPrincipalStore implements DocumentPrincipalStore {
  private readonly db: Database;
  constructor(opts: SqliteDocumentPrincipalStoreOptions) {
    this.db = opts.db;
  }

  public async grant(input: GrantInput): Promise<void> {
    this.db
      .prepare(
        `INSERT OR IGNORE INTO document_principal
         (document_id, principal_type, principal_id, permission, granted_by, granted_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        input.documentId,
        input.principalType,
        input.principalId,
        input.permission,
        input.grantedBy,
        Date.now(),
      );
  }

  public async revoke(input: Omit<GrantInput, 'grantedBy'>): Promise<void> {
    this.db
      .prepare(
        `DELETE FROM document_principal
         WHERE document_id = ? AND principal_type = ? AND principal_id = ? AND permission = ?`,
      )
      .run(
        input.documentId,
        input.principalType,
        input.principalId,
        input.permission,
      );
  }

  public async listByDocument(documentId: DocumentId): Promise<readonly DocumentPrincipal[]> {
    const rows = this.db
      .prepare('SELECT * FROM document_principal WHERE document_id = ? ORDER BY granted_at ASC')
      .all(documentId) as Record<string, unknown>[];
    return rows.map(rowToPrincipal);
  }

  public async listByPrincipal(
    principalType: DocumentPrincipalType,
    principalId: string,
  ): Promise<readonly DocumentPrincipal[]> {
    const rows = this.db
      .prepare(
        `SELECT * FROM document_principal
         WHERE principal_type = ? AND principal_id = ?
         ORDER BY granted_at DESC`,
      )
      .all(principalType, principalId) as Record<string, unknown>[];
    return rows.map(rowToPrincipal);
  }

  public async hasAccess(
    documentId: DocumentId,
    principals: readonly PrincipalRef[],
  ): Promise<boolean> {
    if (principals.length === 0) return false;
    const tuples = principals.map((p) => `(principal_type = ? AND principal_id = ?)`).join(' OR ');
    const params: unknown[] = [documentId];
    for (const p of principals) {
      params.push(p.type);
      params.push(p.id);
    }
    const row = this.db
      .prepare(
        `SELECT 1 FROM document_principal
         WHERE document_id = ? AND (${tuples})
         LIMIT 1`,
      )
      .get(...params);
    return row !== undefined;
  }

  public async applyDefaultAcl(input: {
    documentId: DocumentId;
    ownerId: UserId;
    adminUserIds: readonly UserId[];
  }): Promise<void> {
    await this.grant({
      documentId: input.documentId,
      principalType: 'user',
      principalId: input.ownerId,
      permission: 'admin',
      grantedBy: input.ownerId,
    });
    for (const adminId of input.adminUserIds) {
      await this.grant({
        documentId: input.documentId,
        principalType: 'user',
        principalId: adminId,
        permission: 'admin',
        grantedBy: input.ownerId,
      });
    }
  }

  public async close(): Promise<void> {
    // No-op.
  }
}