/**
 * T3 trace corpus — sqlite-backed, shared Database handle.
 *
 * C-03: takes the shared `Database` handle. Schema lives in
 * `Workspace.open()`. Population is a separate `@raghub/traces`
 * builder step (commit 12).
 */

import type { WorkspaceId, UserId, TraceId } from '../domain/index.js';
import { brandId } from '../domain/index.js';
import { VectorStoreError } from '../errors/index.js';
import type { Database } from '../workspace.js';

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

export interface SqliteTraceCorpusOptions {
  readonly db: Database;
}

export class SqliteTraceCorpus implements TraceCorpus {
  private readonly db: Database;

  constructor(opts: SqliteTraceCorpusOptions) {
    this.db = opts.db;
  }

  public async insert(record: TraceInsert): Promise<void> {
    this.db
      .prepare(
        `INSERT OR REPLACE INTO trace_corpus
         (trace_id, workspace_id, user_id, source_problem, raw, struct, semantic, reflect, embedding, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
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
    const col = query.representation;
    if (col !== 'struct' && col !== 'semantic' && col !== 'reflect') return [];
    const rows = this.db
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
    const row = this.db
      .prepare('SELECT COUNT(*) AS n FROM trace_corpus WHERE workspace_id = ?')
      .get(workspaceId) as Record<string, unknown>;
    return Number(row['n'] ?? 0);
  }

  public async close(): Promise<void> {
    // No-op.
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

void VectorStoreError;
