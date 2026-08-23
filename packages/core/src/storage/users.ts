/**
 * SQLite-backed user store.
 *
 * Schema: tenants(id, name, plan, created_at, is_admin),
 * users(id, workspace_id, email, password_hash, role,
 * allowed_companies_json, created_at).
 *
 * Single-writer (better-sqlite3 is synchronous; wrap in p-queue if
 * multi-writer is needed). All reads filter on workspace_id.
 */

import type { Workspace, WorkspaceId, User, UserId } from '../domain/index.js';
import {
  brandId,
  Workspace as TenantClass,
  User as UserClass,
  UserRole,
} from '../domain/index.js';
import type { WorkspaceId as TenantIdType, UserId as UserIdType } from '../domain/index.js';
import { AuthError, ConfigurationError, VectorStoreError } from '../errors/index.js';

export interface UserStore {
  getByEmail(email: string): Promise<{ user: User; passwordHash: string } | null>;
  getById(workspaceId: WorkspaceId, id: UserId): Promise<User | null>;
  create(input: {
    workspaceId: WorkspaceId;
    email: string;
    passwordHash: string;
    role: keyof typeof UserRole;
    allowedCompanies: readonly string[];
  }): Promise<User>;
  upsertWorkspace(input: { id: WorkspaceId; name: string; plan: 'Free' | 'Pro' | 'Enterprise' }): Promise<Workspace>;
  getWorkspace(id: WorkspaceId): Promise<Workspace | null>;
  close(): Promise<void>;
}

interface Database {
  prepare(sql: string): Statement;
  exec(sql: string): void;
  close(): void;
  pragma?(source: string): unknown;
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

export interface SqliteUserStoreOptions {
  readonly path: string;
}

export class SqliteUserStore implements UserStore {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteUserStoreOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    if (db.pragma) db.pragma('journal_mode = WAL');
    db.exec(`
      CREATE TABLE IF NOT EXISTS tenants (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        plan TEXT NOT NULL,
        is_admin INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        allowed_companies_json TEXT NOT NULL DEFAULT '[]',
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(workspace_id);
    `);
    this.db = db;
    return db;
  }

  public async upsertWorkspace(input: {
    id: WorkspaceId;
    name: string;
    plan: 'Free' | 'Pro' | 'Enterprise';
  }): Promise<Workspace> {
    const db = await this.ensure();
    const now = Date.now();
    const planLower = input.plan.toLowerCase() as 'free' | 'pro' | 'enterprise';
    db.prepare(
      `INSERT INTO tenants (id, name, plan, is_admin, created_at)
       VALUES (?, ?, ?, 0, ?)
       ON CONFLICT(id) DO UPDATE SET name=excluded.name, plan=excluded.plan`,
    ).run(input.id, input.name, planLower, now);
    return new TenantClass({
      id: input.id,
      name: input.name,
      plan: planLower,
      createdAt: new Date(now),
      isAdmin: false,
    });
  }

  public async getWorkspace(id: WorkspaceId): Promise<Workspace | null> {
    const db = await this.ensure();
    const row = db.prepare('SELECT * FROM tenants WHERE id = ?').get(id) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    const planKey = String(row['plan']) as 'free' | 'pro' | 'enterprise';
    return new TenantClass({
      id: brandId<TenantIdType>(String(row['id'])),
      name: String(row['name']),
      plan: planKey,
      createdAt: new Date(Number(row['created_at'])),
      isAdmin: Number(row['is_admin']) === 1,
    });
  }

  public async getByEmail(email: string): Promise<{ user: User; passwordHash: string } | null> {
    const db = await this.ensure();
    const row = db.prepare('SELECT * FROM users WHERE email = ?').get(email) as
      | Record<string, unknown>
      | undefined;
    if (!row) return null;
    const user = rowToUser(row);
    return { user, passwordHash: String(row['password_hash']) };
  }

  public async getById(workspaceId: WorkspaceId, id: UserId): Promise<User | null> {
    const db = await this.ensure();
    const row = db
      .prepare('SELECT * FROM users WHERE workspace_id = ? AND id = ?')
      .get(workspaceId, id) as Record<string, unknown> | undefined;
    return row ? rowToUser(row) : null;
  }

  public async create(input: {
    workspaceId: WorkspaceId;
    email: string;
    passwordHash: string;
    role: keyof typeof UserRole;
    allowedCompanies: readonly string[];
  }): Promise<User> {
    if (!input.email.includes('@')) throw new AuthError('invalid email');
    const db = await this.ensure();
    const id = brandId<UserIdType>(`usr_${Math.random().toString(36).slice(2, 14)}`);
    const now = Date.now();
    try {
      db.prepare(
        `INSERT INTO users (id, workspace_id, email, password_hash, role, allowed_companies_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
      ).run(
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
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

const rowToUser = (row: Record<string, unknown>): User => {
  const id = brandId<UserIdType>(String(row['id']));
  const workspaceId = brandId<TenantIdType>(String(row['workspace_id']));
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