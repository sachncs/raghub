/**
 * Audit event store.
 *
 * Lightweight append-only log of high-value actions: auth events,
 * ACL grants, role changes, document uploads, settings updates.
 * Backed by the `audit_event` table already in workspace.db.
 */

import type { Database } from '../workspace.js';
import type { UserId, WorkspaceId } from '../domain/index.js';

export type AuditEventKind =
  | 'auth.register'
  | 'auth.login.success'
  | 'auth.login.failure'
  | 'auth.logout'
  | 'workspace.member.add'
  | 'workspace.member.remove'
  | 'workspace.member.role_change'
  | 'document.upload'
  | 'document.delete'
  | 'document.acl.grant'
  | 'document.acl.revoke'
  | 'settings.update'
  | 'ingest.complete'
  | 'ingest.failure';

export interface AuditEvent {
  readonly kind: AuditEventKind;
  readonly workspaceId: WorkspaceId;
  readonly actorId: UserId | null;
  readonly resourceId: string | null;
  readonly detail: Readonly<Record<string, unknown>>;
  readonly createdAt: Date;
}

export interface AuditEventStore {
  record(input: Omit<AuditEvent, 'createdAt'>): Promise<void>;
  list(input: { workspaceId: WorkspaceId; limit?: number; since?: Date }): Promise<readonly AuditEvent[]>;
  close(): Promise<void>;
}

export interface SqliteAuditEventStoreOptions {
  readonly db: Database;
}

const rowToEvent = (row: Record<string, unknown>): AuditEvent => ({
  kind: String(row['kind']) as AuditEventKind,
  workspaceId: String(row['workspace_id']) as WorkspaceId,
  actorId: row['actor_id'] === null ? null : (String(row['actor_id']) as UserId),
  resourceId: row['resource_id'] === null ? null : String(row['resource_id']),
  detail: JSON.parse(String(row['detail_json'] ?? '{}')) as Record<string, unknown>,
  createdAt: new Date(Number(row['created_at'])),
});

export class SqliteAuditEventStore implements AuditEventStore {
  private readonly db: Database;
  constructor(opts: SqliteAuditEventStoreOptions) {
    this.db = opts.db;
  }

  public async record(input: Omit<AuditEvent, 'createdAt'>): Promise<void> {
    this.db
      .prepare(
        `INSERT INTO audit_event (workspace_id, kind, actor_id, resource_id, detail_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        input.workspaceId,
        input.kind,
        input.actorId,
        input.resourceId,
        JSON.stringify(input.detail),
        Date.now(),
      );
  }

  public async list(input: { workspaceId: WorkspaceId; limit?: number; since?: Date }): Promise<readonly AuditEvent[]> {
    const limit = input.limit ?? 200;
    const since = input.since ?? new Date(0);
    const rows = this.db
      .prepare(
        `SELECT * FROM audit_event
         WHERE workspace_id = ? AND created_at >= ?
         ORDER BY created_at DESC
         LIMIT ?`,
      )
      .all(input.workspaceId, since.getTime(), limit) as Record<string, unknown>[];
    return rows.map(rowToEvent);
  }

  public async close(): Promise<void> {
    // No-op.
  }
}