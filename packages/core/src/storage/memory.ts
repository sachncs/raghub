/**
 * WorkspaceMemoryStore — persisted user/workspace memory facts.
 *
 * Backs the Strands `MemoryManager` primitive. Facts are stored in
 * `memory_fact(workspace_id, scope, user_id, content, embedding)`.
 * Scope = 'user' or 'workspace' (user-scoped facts are scoped to
 * the active user_id; workspace-scoped are visible to every member
 * of the workspace).
 *
 * `search()` does a token-overlap scan against `content` since the
 * table does not (yet) carry an FTS5 index. The embedder is used
 * only for `remember()` so we can re-rank later.
 */

import type { Database } from '../workspace.js';
import type { UserId, WorkspaceId } from '../domain/index.js';

export const MemoryScope = {
  User: 'user',
  Workspace: 'workspace',
} as const;

export type MemoryScopeValue = (typeof MemoryScope)[keyof typeof MemoryScope];

export interface MemoryFact {
  readonly id: number;
  readonly workspaceId: WorkspaceId;
  readonly scope: MemoryScopeValue;
  readonly userId: UserId | null;
  readonly content: string;
  readonly metadata: Record<string, string>;
  readonly createdAt: Date;
}

export interface MemorySearchInput {
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId | null;
  readonly query: string;
  readonly topK: number;
  readonly allowedCompanies: readonly string[];
}

export interface MemorySearchResult {
  readonly id: string;
  readonly content: string;
  readonly scope: MemoryScopeValue;
  readonly score: number;
}

export interface RememberInput {
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId | null;
  readonly scope: MemoryScopeValue;
  readonly content: string;
  readonly metadata?: Readonly<Record<string, string>>;
}

export interface WorkspaceMemoryStore {
  remember(input: RememberInput): Promise<MemoryFact>;
  search(input: MemorySearchInput): Promise<readonly MemorySearchResult[]>;
  delete(id: number): Promise<void>;
  listForUser(workspaceId: WorkspaceId, userId: UserId): Promise<readonly MemoryFact[]>;
  close(): Promise<void>;
}

export interface SqliteWorkspaceMemoryStoreOptions {
  readonly db: Database;
}

const rowToFact = (row: Record<string, unknown>): MemoryFact => ({
  id: Number(row['id']),
  workspaceId: String(row['workspace_id']) as WorkspaceId,
  scope: (row['scope'] as MemoryScopeValue) ?? MemoryScope.User,
  userId: row['user_id'] === null ? null : (String(row['user_id']) as UserId),
  content: String(row['content']),
  metadata: JSON.parse(String(row['metadata_json'] ?? '{}')) as Record<string, string>,
  createdAt: new Date(Number(row['created_at'])),
});

const tokenize = (s: string): readonly string[] =>
  s
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length >= 3);

const overlap = (a: readonly string[], b: readonly string[]): number => {
  const set = new Set(a);
  let n = 0;
  for (const w of b) if (set.has(w)) n++;
  return n;
};

export class SqliteWorkspaceMemoryStore implements WorkspaceMemoryStore {
  private readonly db: Database;
  constructor(opts: SqliteWorkspaceMemoryStoreOptions) {
    this.db = opts.db;
  }

  public async remember(input: RememberInput): Promise<MemoryFact> {
    const result = this.db
      .prepare(
        `INSERT INTO memory_fact (workspace_id, scope, user_id, content, metadata_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?)
         RETURNING id`,
      )
      .get(
        input.workspaceId,
        input.scope,
        input.userId,
        input.content,
        JSON.stringify(input.metadata ?? {}),
        Date.now(),
      ) as { id: number } | undefined;
    const id = result?.id ?? 0;
    return {
      id,
      workspaceId: input.workspaceId,
      scope: input.scope,
      userId: input.userId,
      content: input.content,
      metadata: { ...(input.metadata ?? {}) },
      createdAt: new Date(),
    };
  }

  public async search(input: MemorySearchInput): Promise<readonly MemorySearchResult[]> {
    const rows = this.db
      .prepare(
        `SELECT * FROM memory_fact
         WHERE workspace_id = ?
           AND (scope = 'workspace' OR (scope = 'user' AND user_id = ?))
         ORDER BY created_at DESC
         LIMIT ?`,
      )
      .all(input.workspaceId, input.userId, Math.max(50, input.topK * 5)) as Record<string, unknown>[];
    const qTokens = tokenize(input.query);
    const scored = rows.map((r) => {
      const fact = rowToFact(r);
      const score = qTokens.length === 0 ? 1 : overlap(tokenize(fact.content), qTokens) / qTokens.length;
      return { id: String(fact.id), content: fact.content, scope: fact.scope, score };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, input.topK);
  }

  public async delete(id: number): Promise<void> {
    this.db.prepare('DELETE FROM memory_fact WHERE id = ?').run(id);
  }

  public async listForUser(workspaceId: WorkspaceId, userId: UserId): Promise<readonly MemoryFact[]> {
    const rows = this.db
      .prepare(
        `SELECT * FROM memory_fact
         WHERE workspace_id = ?
           AND (scope = 'workspace' OR (scope = 'user' AND user_id = ?))
         ORDER BY created_at DESC
         LIMIT 200`,
      )
      .all(workspaceId, userId) as Record<string, unknown>[];
    return rows.map(rowToFact);
  }

  public async close(): Promise<void> {
    // No-op.
  }
}