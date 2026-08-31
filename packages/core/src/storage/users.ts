/**
 * User store — sqlite-backed, shared Database handle from a
 * `Workspace.open(path)`.
 *
 * C-03: takes the shared `Database` handle. Schema creation lives
 * in `Workspace.open()` so this file only owns the SQL operations.
 */

import type {
  WorkspaceId,
  UserId,
  User,
  DocumentId,
  CollectionId,
} from '../domain/index.js';
import { User as UserClass, UserRole } from '../domain/index.js';
import { AuthError, ConfigurationError } from '../errors/index.js';
import type { Database } from '../workspace.js';

export interface UserStore {
  getByEmail(email: string): Promise<{ user: User; passwordHash: string } | null>;
  getById(workspaceId: WorkspaceId, id: UserId): Promise<{ user: User; passwordHash: string } | null>;
  create(input: {
    workspaceId: WorkspaceId;
    email: string;
    passwordHash: string;
    role: keyof typeof UserRole;
    allowedCompanies: readonly string[];
  }): Promise<User>;
  updatePassword(workspaceId: WorkspaceId, id: UserId, passwordHash: string): Promise<boolean>;
  close(): Promise<void>;
}

export interface SqliteUserStoreOptions {
  readonly db: Database;
}

export class SqliteUserStore implements UserStore {
  private readonly db: Database;

  constructor(opts: SqliteUserStoreOptions) {
    this.db = opts.db;
  }

  public async getByEmail(email: string): Promise<{ user: User; passwordHash: string } | null> {
    const row = this.db
      .prepare('SELECT * FROM users WHERE email = ?')
      .get(email) as Record<string, unknown> | undefined;
    if (!row) return null;
    const user = rowToUser(row);
    return { user, passwordHash: String(row['password_hash']) };
  }

  public async getById(workspaceId: WorkspaceId, id: UserId): Promise<{ user: User; passwordHash: string } | null> {
    const row = this.db
      .prepare('SELECT * FROM users WHERE workspace_id = ? AND id = ?')
      .get(workspaceId, id) as Record<string, unknown> | undefined;
    if (!row) return null;
    return { user: rowToUser(row), passwordHash: String(row['password_hash']) };
  }

  public async updatePassword(
    workspaceId: WorkspaceId,
    id: UserId,
    passwordHash: string,
  ): Promise<boolean> {
    const info = this.db
      .prepare('UPDATE users SET password_hash = ? WHERE workspace_id = ? AND id = ?')
      .run(passwordHash, workspaceId, id);
    return info.changes > 0;
  }

  public async create(input: {
    workspaceId: WorkspaceId;
    email: string;
    passwordHash: string;
    role: keyof typeof UserRole;
    allowedCompanies: readonly string[];
  }): Promise<User> {
    if (!input.email.includes('@')) throw new AuthError('invalid email');
    const id = `usr_${Math.random().toString(36).slice(2, 14)}` as UserId;
    const now = Date.now();
    try {
      this.db
        .prepare(
          `INSERT INTO users (id, workspace_id, email, password_hash, role, allowed_companies_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)`,
        )
        .run(
          id,
          input.workspaceId,
          input.email,
          input.passwordHash,
          input.role,
          JSON.stringify([...input.allowedCompanies]),
          now,
        );
    } catch (e) {
      throw new ConfigurationError(`failed to create user: ${String(e)}`, {
        details: { email: input.email },
      });
    }
    return new UserClass({
      id,
      workspaceId: input.workspaceId,
      email: input.email,
      role: UserRole[input.role],
      allowedCompanies: input.allowedCompanies,
      createdAt: new Date(now),
    });
  }

  public async close(): Promise<void> {
    // No-op: db is owned by Workspace.
  }
}

const rowToUser = (row: Record<string, unknown>): User => {
  const id = String(row['id']) as UserId;
  const workspaceId = String(row['workspace_id']) as WorkspaceId;
  const roleKey = String(row['role']) as keyof typeof UserRole;
  const allowedJson = String(row['allowed_companies_json'] ?? '[]');
  return new UserClass({
    id,
    workspaceId,
    email: String(row['email']),
    role: UserRole[roleKey] ?? UserRole.Member,
    allowedCompanies: JSON.parse(allowedJson) as string[],
    createdAt: new Date(Number(row['created_at'])),
  });
};
