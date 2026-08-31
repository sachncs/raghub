/**
 * Workspace RBAC — `workspace_member` rows assign a role to every
 * user in the workspace. The first user to onboard becomes `owner`.
 * Owners and admins can manage users, groups, roles, and document
 * ACL. Members can ingest and query. Viewers are read-only.
 */

import type { Database } from '../workspace.js';
import type { User, UserId, WorkspaceId } from '../domain/index.js';

export const WorkspaceMemberRole = {
  Owner: 'owner',
  Admin: 'admin',
  Member: 'member',
  Viewer: 'viewer',
} as const;

export type WorkspaceMemberRoleValue =
  (typeof WorkspaceMemberRole)[keyof typeof WorkspaceMemberRole];

export interface WorkspaceMember {
  readonly userId: UserId;
  readonly role: WorkspaceMemberRoleValue;
  readonly joinedAt: Date;
  readonly displayName: string | null;
}

export interface WorkspaceMemberStore {
  upsert(input: {
    workspaceId: WorkspaceId;
    userId: UserId;
    role: WorkspaceMemberRoleValue;
    displayName?: string | null;
  }): Promise<WorkspaceMember>;
  get(workspaceId: WorkspaceId, userId: UserId): Promise<WorkspaceMember | null>;
  list(workspaceId: WorkspaceId): Promise<readonly WorkspaceMember[]>;
  remove(workspaceId: WorkspaceId, userId: UserId): Promise<void>;
  close(): Promise<void>;
}

export interface SqliteWorkspaceMemberStoreOptions {
  readonly db: Database;
}

const rowToMember = (row: Record<string, unknown>): WorkspaceMember => ({
  userId: String(row['user_id']) as UserId,
  role: (row['role'] as WorkspaceMemberRoleValue) ?? WorkspaceMemberRole.Member,
  joinedAt: new Date(Number(row['joined_at'])),
  displayName: row['display_name'] === null || row['display_name'] === undefined
    ? null
    : String(row['display_name']),
});

export class SqliteWorkspaceMemberStore implements WorkspaceMemberStore {
  private readonly db: Database;

  constructor(opts: SqliteWorkspaceMemberStoreOptions) {
    this.db = opts.db;
  }

  public async upsert(input: {
    workspaceId: WorkspaceId;
    userId: UserId;
    role: WorkspaceMemberRoleValue;
    displayName?: string | null;
  }): Promise<WorkspaceMember> {
    const now = Date.now();
    const existing = this.db
      .prepare('SELECT display_name FROM workspace_member WHERE user_id = ?')
      .get(input.userId) as { display_name: string | null } | undefined;
    const nextDisplayName =
      input.displayName === undefined
        ? (existing?.display_name ?? null)
        : input.displayName;
    this.db
      .prepare(
        `INSERT INTO workspace_member (user_id, role, joined_at, display_name)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(user_id) DO UPDATE SET
           role = excluded.role,
           display_name = COALESCE(excluded.display_name, workspace_member.display_name)`,
      )
      .run(input.userId, input.role, now, nextDisplayName);
    return {
      userId: input.userId,
      role: input.role,
      joinedAt: new Date(now),
      displayName: nextDisplayName,
    };
  }

  public async get(workspaceId: WorkspaceId, userId: UserId): Promise<WorkspaceMember | null> {
    const row = this.db
      .prepare('SELECT user_id, role, joined_at FROM workspace_member WHERE user_id = ?')
      .get(userId) as Record<string, unknown> | undefined;
    void workspaceId;
    return row ? rowToMember(row) : null;
  }

  public async list(workspaceId: WorkspaceId): Promise<readonly WorkspaceMember[]> {
    void workspaceId;
    const rows = this.db
      .prepare('SELECT user_id, role, joined_at FROM workspace_member ORDER BY joined_at ASC')
      .all() as Record<string, unknown>[];
    return rows.map(rowToMember);
  }

  public async remove(workspaceId: WorkspaceId, userId: UserId): Promise<void> {
    void workspaceId;
    this.db.prepare('DELETE FROM workspace_member WHERE user_id = ?').run(userId);
  }

  public async close(): Promise<void> {
    // No-op.
  }
}

export const canManageWorkspace = (role: WorkspaceMemberRoleValue): boolean =>
  role === WorkspaceMemberRole.Owner || role === WorkspaceMemberRole.Admin;

export const canIngest = (role: WorkspaceMemberRoleValue): boolean =>
  role !== WorkspaceMemberRole.Viewer;

export const resolveRoleFor = (
  user: User,
  member: WorkspaceMember | null,
): WorkspaceMemberRoleValue => {
  if (user.isAdmin) return WorkspaceMemberRole.Owner;
  if (member === null) return WorkspaceMemberRole.Viewer;
  return member.role;
};
