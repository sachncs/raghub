/**
 * Entity graph + traversal.
 *
 * C-03: takes the shared `Database` handle. Phase 1 keeps this
 * intentionally small — entities are capitalised phrases extracted
 * from chunk text via `extractEntities`, edges are co-occurrence in
 * the same chunk. The graph is stored in the same `workspace.db` so
 * retrieval stays zero-dep beyond better-sqlite3.
 */

import type { WorkspaceId } from '../domain/index.js';
import type { ChunkId } from '../domain/index.js';
import type { Database } from '../workspace.js';

export interface GraphEntity {
  readonly name: string;
  readonly workspaceId: WorkspaceId;
  readonly chunkCount: number;
}

export interface GraphEdge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
}

export interface GraphStore {
  addMentions(workspaceId: WorkspaceId, chunkId: ChunkId, entities: readonly string[]): Promise<void>;
  searchEntities(workspaceId: WorkspaceId, query: string, limit?: number): Promise<readonly GraphEntity[]>;
  expandNeighborhood(
    workspaceId: WorkspaceId,
    seeds: readonly string[],
    hop: number,
    limit?: number,
  ): Promise<readonly GraphEntity[]>;
  close(): Promise<void>;
}

const STOPWORDS = new Set([
  'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those', 'are', 'was',
  'were', 'been', 'have', 'has', 'had', 'into', 'than', 'then', 'them', 'they', 'our',
  'your', 'their', 'his', 'her', 'its', 'what', 'when', 'where', 'which', 'while',
  'about', 'would', 'could', 'should', 'there', 'because', 'before', 'after',
]);

export const extractEntities = (text: string): readonly string[] => {
  const out = new Set<string>();
  const re = /\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+){0,3})\b/g;
  for (const m of text.matchAll(re)) {
    const e = (m[1] ?? '').trim();
    if (e.length < 3) continue;
    const words = e.toLowerCase().split(/\s+/);
    if (words.some((w) => STOPWORDS.has(w))) continue;
    out.add(e);
  }
  return [...out];
};

export interface SqliteGraphStoreOptions {
  readonly db: Database;
}

export class SqliteGraphStore implements GraphStore {
  private readonly db: Database;

  constructor(opts: SqliteGraphStoreOptions) {
    this.db = opts.db;
  }

  public async addMentions(
    workspaceId: WorkspaceId,
    chunkId: ChunkId,
    entities: readonly string[],
  ): Promise<void> {
    if (entities.length === 0) return;
    const stmt = this.db.prepare(
      `INSERT OR IGNORE INTO graph_entities (name, workspace_id, chunk_id) VALUES (?, ?, ?)`,
    );
    for (const name of entities) {
      stmt.run(name, workspaceId, chunkId);
    }
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const a = entities[i];
        const b = entities[j];
        if (!a || !b) continue;
        this.db
          .prepare(
            `INSERT INTO graph_edges (workspace_id, from_name, to_name, weight)
             VALUES (?, ?, ?, 1)
             ON CONFLICT(workspace_id, from_name, to_name) DO UPDATE SET weight = weight + 1`,
          )
          .run(workspaceId, a, b);
      }
    }
  }

  public async searchEntities(
    workspaceId: WorkspaceId,
    query: string,
    limit: number = 10,
  ): Promise<readonly GraphEntity[]> {
    const like = `%${query.replace(/[%_]/g, '\\$&')}%`;
    const rows = this.db
      .prepare(
        `SELECT name, COUNT(chunk_id) AS cnt FROM graph_entities
         WHERE workspace_id = ? AND name LIKE ? ESCAPE '\\'
         GROUP BY name ORDER BY cnt DESC LIMIT ?`,
      )
      .all(workspaceId, like, limit) as Record<string, unknown>[];
    return rows.map((r) => ({
      name: String(r['name']),
      workspaceId,
      chunkCount: Number(r['cnt'] ?? 0),
    }));
  }

  public async expandNeighborhood(
    workspaceId: WorkspaceId,
    seeds: readonly string[],
    hop: number,
    limit: number = 20,
  ): Promise<readonly GraphEntity[]> {
    if (seeds.length === 0 || hop < 1) return [];
    const visited = new Set<string>(seeds);
    let frontier = [...seeds];
    for (let h = 0; h < hop; h++) {
      if (frontier.length === 0) break;
      const placeholders = frontier.map(() => '?').join(',');
      const rows = this.db
        .prepare(
          `SELECT to_name AS name, COUNT(*) AS w FROM graph_edges
           WHERE workspace_id = ? AND from_name IN (${placeholders})
           GROUP BY to_name ORDER BY w DESC LIMIT 100`,
        )
        .all(workspaceId, ...frontier) as Record<string, unknown>[];
      const next: string[] = [];
      for (const r of rows) {
        const n = String(r['name']);
        if (!visited.has(n)) {
          visited.add(n);
          next.push(n);
        }
      }
      frontier = next;
    }
    return [...visited].slice(0, limit).map((name) => ({ name, workspaceId, chunkCount: 0 }));
  }

  public async close(): Promise<void> {
    // No-op.
  }
}
