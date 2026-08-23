/**
 * Roles + groups storage.
 *
 * Two parallel tables:
 *  - `role` / `role_member`: named roles (analyst, engineer, ...)
 *    with members that can be either users or groups.
 *  - `workspace_group` / `workspace_group_member`: groups of users.
 *
 * Document ACL (commit 7) references these via document_principal
 * (principal_type = 'role' | 'group', principal_id = role_id / group_id).
 */

import type { Database } from '../workspace.js';

export interface Role {
  readonly id: string;
  readonly name: string;
  readonly description: string | null;
  readonly createdAt: Date;
}

export interface RoleAssignment {
  readonly roleId: string;
  readonly principalType: 'user' | 'group';
  readonly principalId: string;
}

export interface Group {
  readonly id: string;
  readonly name: string;
  readonly description: string | null;
  readonly createdAt: Date;
}

export interface GroupMembership {
  readonly groupId: string;
  readonly userId: string;
}

export interface RoleStore {
  create(input: { id?: string; name: string; description?: string }): Promise<Role>;
  get(id: string): Promise<Role | null>;
  getByName(name: string): Promise<Role | null>;
  list(): Promise<readonly Role[]>;
  addMember(roleId: string, principalType: 'user' | 'group', principalId: string): Promise<void>;
  removeMember(roleId: string, principalType: 'user' | 'group', principalId: string): Promise<void>;
  members(roleId: string): Promise<readonly RoleAssignment[]>;
  close(): Promise<void>;
}

export interface GroupStore {
  create(input: { id?: string; name: string; description?: string }): Promise<Group>;
  get(id: string): Promise<Group | null>;
  getByName(name: string): Promise<Group | null>;
  list(): Promise<readonly Group[]>;
  addMember(groupId: string, userId: string): Promise<void>;
  removeMember(groupId: string, userId: string): Promise<void>;
  members(groupId: string): Promise<readonly GroupMembership[]>;
  userGroups(userId: string): Promise<readonly Group[]>;
  close(): Promise<void>;
}

export interface SqliteRoleStoreOptions {
  readonly db: Database;
}

export interface SqliteGroupStoreOptions {
  readonly db: Database;
}

const rowToRole = (row: Record<string, unknown>): Role => ({
  id: String(row['id']),
  name: String(row['name']),
  description: row['description'] === null ? null : String(row['description']),
  createdAt: new Date(Number(row['created_at'])),
});

const rowToGroup = (row: Record<string, unknown>): Group => ({
  id: String(row['id']),
  name: String(row['name']),
  description: row['description'] === null ? null : String(row['description']),
  createdAt: new Date(Number(row['created_at'])),
});

const newId = (prefix: string): string =>
  `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;

export class SqliteRoleStore implements RoleStore {
  private readonly db: Database;
  constructor(opts: SqliteRoleStoreOptions) {
    this.db = opts.db;
  }

  public async create(input: { id?: string; name: string; description?: string }): Promise<Role> {
    const id = input.id ?? newId('role');
    const now = Date.now();
    this.db
      .prepare('INSERT INTO role (id, name, description, created_at) VALUES (?, ?, ?, ?)')
      .run(id, input.name, input.description ?? null, now);
    return { id, name: input.name, description: input.description ?? null, createdAt: new Date(now) };
  }

  public async get(id: string): Promise<Role | null> {
    const row = this.db.prepare('SELECT * FROM role WHERE id = ?').get(id) as Record<string, unknown> | undefined;
    return row ? rowToRole(row) : null;
  }

  public async getByName(name: string): Promise<Role | null> {
    const row = this.db
      .prepare('SELECT * FROM role WHERE name = ?')
      .get(name) as Record<string, unknown> | undefined;
    return row ? rowToRole(row) : null;
  }

  public async list(): Promise<readonly Role[]> {
    const rows = this.db.prepare('SELECT * FROM role ORDER BY name ASC').all() as Record<string, unknown>[];
    return rows.map(rowToRole);
  }

  public async addMember(roleId: string, principalType: 'user' | 'group', principalId: string): Promise<void> {
    this.db
      .prepare(
        `INSERT OR IGNORE INTO role_member (role_id, principal_type, principal_id)
         VALUES (?, ?, ?)`,
      )
      .run(roleId, principalType, principalId);
  }

  public async removeMember(roleId: string, principalType: 'user' | 'group', principalId: string): Promise<void> {
    this.db
      .prepare(
        `DELETE FROM role_member WHERE role_id = ? AND principal_type = ? AND principal_id = ?`,
      )
      .run(roleId, principalType, principalId);
  }

  public async members(roleId: string): Promise<readonly RoleAssignment[]> {
    const rows = this.db
      .prepare('SELECT role_id, principal_type, principal_id FROM role_member WHERE role_id = ?')
      .all(roleId) as Record<string, unknown>[];
    return rows.map((r) => ({
      roleId: String(r['role_id']),
      principalType: r['principal_type'] as 'user' | 'group',
      principalId: String(r['principal_id']),
    }));
  }

  public async close(): Promise<void> {
    // No-op.
  }
}

export class SqliteGroupStore implements GroupStore {
  private readonly db: Database;
  constructor(opts: SqliteGroupStoreOptions) {
    this.db = opts.db;
  }

  public async create(input: { id?: string; name: string; description?: string }): Promise<Group> {
    const id = input.id ?? newId('grp');
    const now = Date.now();
    this.db
      .prepare('INSERT INTO workspace_group (id, name, description, created_at) VALUES (?, ?, ?, ?)')
      .run(id, input.name, input.description ?? null, now);
    return { id, name: input.name, description: input.description ?? null, createdAt: new Date(now) };
  }

  public async get(id: string): Promise<Group | null> {
    const row = this.db
      .prepare('SELECT * FROM workspace_group WHERE id = ?')
      .get(id) as Record<string, unknown> | undefined;
    return row ? rowToGroup(row) : null;
  }

  public async getByName(name: string): Promise<Group | null> {
    const row = this.db
      .prepare('SELECT * FROM workspace_group WHERE name = ?')
      .get(name) as Record<string, unknown> | undefined;
    return row ? rowToGroup(row) : null;
  }

  public async list(): Promise<readonly Group[]> {
    const rows = this.db
      .prepare('SELECT * FROM workspace_group ORDER BY name ASC')
      .all() as Record<string, unknown>[];
    return rows.map(rowToGroup);
  }

  public async addMember(groupId: string, userId: string): Promise<void> {
    this.db
      .prepare('INSERT OR IGNORE INTO workspace_group_member (group_id, user_id) VALUES (?, ?)')
      .run(groupId, userId);
  }

  public async removeMember(groupId: string, userId: string): Promise<void> {
    this.db
      .prepare('DELETE FROM workspace_group_member WHERE group_id = ? AND user_id = ?')
      .run(groupId, userId);
  }

  public async members(groupId: string): Promise<readonly GroupMembership[]> {
    const rows = this.db
      .prepare('SELECT group_id, user_id FROM workspace_group_member WHERE group_id = ?')
      .all(groupId) as Record<string, unknown>[];
    return rows.map((r) => ({ groupId: String(r['group_id']), userId: String(r['user_id']) }));
  }

  public async userGroups(userId: string): Promise<readonly Group[]> {
    const rows = this.db
      .prepare(
        `SELECT g.* FROM workspace_group g
         JOIN workspace_group_member gm ON gm.group_id = g.id
         WHERE gm.user_id = ?
         ORDER BY g.name ASC`,
      )
      .all(userId) as Record<string, unknown>[];
    return rows.map(rowToGroup);
  }

  public async close(): Promise<void> {
    // No-op.
  }
}
