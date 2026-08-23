/**
 * Simple entity graph + traversal.
 *
 * Phase 1 keeps this intentionally small: entities are capitalized
 * phrases extracted from chunk text via a regex pass, edges are
 * co-occurrence in the same chunk. The graph is stored in the
 * same sqlite-vec database as a side-table so retrieval stays
 * zero-dep beyond better-sqlite3.
 *
 * `searchEntities` runs a substring match; `expandNeighborhood`
 * walks the graph to depth `hop` from a starting set.
 */

import type { TenantId } from '../domain/ids.js';
import { brandId } from '../domain/ids.js';
import type { ChunkId } from '../domain/ids.js';
import { VectorStoreError } from '../errors/index.js';

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

export interface GraphEntity {
  readonly name: string;
  readonly tenantId: TenantId;
  readonly chunkCount: number;
}

export interface GraphEdge {
  readonly from: string;
  readonly to: string;
  readonly weight: number;
}

export interface GraphStore {
  addMentions(tenantId: TenantId, chunkId: ChunkId, entities: readonly string[]): Promise<void>;
  searchEntities(tenantId: TenantId, query: string, limit?: number): Promise<readonly GraphEntity[]>;
  expandNeighborhood(tenantId: TenantId, seeds: readonly string[], hop: number, limit?: number): Promise<readonly GraphEntity[]>;
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
  readonly path: string;
}

export class SqliteGraphStore implements GraphStore {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteGraphStoreOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS graph_entities (
        name TEXT NOT NULL,
        tenant_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        PRIMARY KEY (name, tenant_id, chunk_id)
      );
      CREATE INDEX IF NOT EXISTS idx_graph_entities_tenant ON graph_entities(tenant_id);
      CREATE TABLE IF NOT EXISTS graph_edges (
        tenant_id TEXT NOT NULL,
        from_name TEXT NOT NULL,
        to_name TEXT NOT NULL,
        weight INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (tenant_id, from_name, to_name)
      );
    `);
    this.db = db;
    return db;
  }

  public async addMentions(
    tenantId: TenantId,
    chunkId: ChunkId,
    entities: readonly string[],
  ): Promise<void> {
    if (entities.length === 0) return;
    const db = await this.ensure();
    const stmt = db.prepare(
      `INSERT OR IGNORE INTO graph_entities (name, tenant_id, chunk_id) VALUES (?, ?, ?)`,
    );
    for (const name of entities) {
      stmt.run(name, tenantId, chunkId);
    }
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const a = entities[i];
        const b = entities[j];
        if (!a || !b) continue;
        db.prepare(
          `INSERT INTO graph_edges (tenant_id, from_name, to_name, weight)
           VALUES (?, ?, ?, 1)
           ON CONFLICT(tenant_id, from_name, to_name) DO UPDATE SET weight = weight + 1`,
        ).run(tenantId, a, b);
      }
    }
  }

  public async searchEntities(
    tenantId: TenantId,
    query: string,
    limit: number = 10,
  ): Promise<readonly GraphEntity[]> {
    const db = await this.ensure();
    const like = `%${query.replace(/[%_]/g, '\\$&')}%`;
    const rows = db
      .prepare(
        `SELECT name, COUNT(chunk_id) AS cnt FROM graph_entities
         WHERE tenant_id = ? AND name LIKE ? ESCAPE '\\'
         GROUP BY name ORDER BY cnt DESC LIMIT ?`,
      )
      .all(tenantId, like, limit) as Record<string, unknown>[];
    return rows.map((r) => ({
      name: String(r['name']),
      tenantId,
      chunkCount: Number(r['cnt'] ?? 0),
    }));
  }

  public async expandNeighborhood(
    tenantId: TenantId,
    seeds: readonly string[],
    hop: number,
    limit: number = 20,
  ): Promise<readonly GraphEntity[]> {
    const db = await this.ensure();
    if (seeds.length === 0 || hop < 1) return [];
    const visited = new Set<string>(seeds);
    let frontier = [...seeds];
    for (let h = 0; h < hop; h++) {
      if (frontier.length === 0) break;
      const placeholders = frontier.map(() => '?').join(',');
      const rows = db
        .prepare(
          `SELECT to_name AS name, COUNT(*) AS w FROM graph_edges
           WHERE tenant_id = ? AND from_name IN (${placeholders})
           GROUP BY to_name ORDER BY w DESC LIMIT 100`,
        )
        .all(tenantId, ...frontier) as Record<string, unknown>[];
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
    return [...visited].slice(0, limit).map((name) => ({ name, tenantId, chunkCount: 0 }));
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

void brandId;