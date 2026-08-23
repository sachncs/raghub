/**
 * T3 trace corpus (Phase 1 surface).
 *
 * Implements the storage + retrieval half of the T3 pipeline
 * (arXiv 2605.03344). The thinker runner + transformations live
 * in the `@raghub/traces` namespace; this module is the SQLite
 * store the @raghub/orchestrator's `trace_search` tool reads
 * from.
 *
 * Schema: trace_corpus(trace_id, workspace_id, user_id, source_problem,
 * raw, struct, semantic, reflect, embedding, created_at).
 *
 * Phase 1 ships the storage + similarity search; populating the
 * table is the `raghub traces build` command (Phase 2). User-
 * supplied problem JSONL only.
 */

import type { ChunkId, WorkspaceId, TraceId, UserId } from '../domain/ids.js';
import { brandId } from '../domain/ids.js';
import { VectorStoreError } from '../errors/index.js';

export type TraceRepresentation = 'struct' | 'semantic' | 'reflect';

export interface TraceRecord {
  readonly id: TraceId;
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId | null;
  readonly sourceProblem: string;
  readonly raw: string;
  readonly struct: string | null;
  readonly semantic: string | null;
  readonly reflect: string | null;
  readonly embedding: readonly number[];
  readonly createdAt: Date;
}

export interface TraceInsert {
  readonly id: TraceId;
  readonly workspaceId: WorkspaceId;
  readonly userId: UserId | null;
  readonly sourceProblem: string;
  readonly raw: string;
  readonly struct?: string | null;
  readonly semantic?: string | null;
  readonly reflect?: string | null;
  readonly embedding: readonly number[];
}

export interface TraceQuery {
  readonly workspaceId: WorkspaceId;
  readonly vector: readonly number[];
  readonly representation: TraceRepresentation;
  readonly topK: number;
}

export interface TraceHit {
  readonly id: TraceId;
  readonly score: number;
  readonly text: string;
  readonly sourceProblem: string;
}

export interface TraceCorpus {
  insert(record: TraceInsert): Promise<void>;
  insertBatch(records: readonly TraceInsert[]): Promise<void>;
  search(query: TraceQuery): Promise<readonly TraceHit[]>;
  count(workspaceId: WorkspaceId): Promise<number>;
  close(): Promise<void>;
}

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

export interface SqliteTraceCorpusOptions {
  readonly path: string;
}

export class SqliteTraceCorpus implements TraceCorpus {
  private db: Database | null = null;
  private readonly path: string;

  constructor(opts: SqliteTraceCorpusOptions) {
    this.path = opts.path;
  }

  private async ensure(): Promise<Database> {
    if (this.db) return this.db;
    const sqlite = await loadBetterSqlite3();
    const db = sqlite(this.path);
    db.exec(`
      CREATE TABLE IF NOT EXISTS trace_corpus (
        trace_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL,
        user_id TEXT,
        source_problem TEXT NOT NULL,
        raw TEXT NOT NULL,
        struct TEXT,
        semantic TEXT,
        reflect TEXT,
        embedding TEXT NOT NULL,
        created_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_traces_tenant ON trace_corpus(workspace_id);
    `);
    this.db = db;
    return db;
  }

  public async insert(record: TraceInsert): Promise<void> {
    const db = await this.ensure();
    db.prepare(
      `INSERT OR REPLACE INTO trace_corpus
       (trace_id, workspace_id, user_id, source_problem, raw, struct, semantic, reflect, embedding, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      record.id,
      record.workspaceId,
      record.userId,
      record.sourceProblem,
      record.raw,
      record.struct ?? null,
      record.semantic ?? null,
      record.reflect ?? null,
      JSON.stringify([...record.embedding]),
      Date.now(),
    );
  }

  public async insertBatch(records: readonly TraceInsert[]): Promise<void> {
    for (const r of records) await this.insert(r);
  }

  public async search(query: TraceQuery): Promise<readonly TraceHit[]> {
    const db = await this.ensure();
    const col = query.representation;
    if (col !== 'struct' && col !== 'semantic' && col !== 'reflect') {
      return [];
    }
    const rows = db
      .prepare(
        `SELECT trace_id, ${col} AS text, source_problem, embedding
         FROM trace_corpus
         WHERE workspace_id = ? AND ${col} IS NOT NULL`,
      )
      .all(query.workspaceId) as Record<string, unknown>[];
    const qvec = new Float32Array(query.vector);
    const scored: TraceHit[] = [];
    for (const row of rows) {
      const embRaw = row['embedding'];
      if (typeof embRaw !== 'string') continue;
      let vec: number[];
      try {
        vec = JSON.parse(embRaw) as number[];
      } catch {
        continue;
      }
      const score = cosine(new Float32Array(vec), qvec);
      if (Number.isFinite(score)) {
        scored.push({
          id: brandId<TraceId>(String(row['trace_id'])),
          score,
          text: String(row['text'] ?? ''),
          sourceProblem: String(row['source_problem']),
        });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, query.topK);
  }

  public async count(workspaceId: WorkspaceId): Promise<number> {
    const db = await this.ensure();
    const row = db
      .prepare('SELECT COUNT(*) AS n FROM trace_corpus WHERE workspace_id = ?')
      .get(workspaceId) as Record<string, unknown>;
    return Number(row['n'] ?? 0);
  }

  public async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

const cosine = (a: Float32Array, b: Float32Array): number => {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
};

void brandId;