/**
 * `raghub migrate pgvector-to-sqlite --dsn <pg> --out chunks.ndjson`
 *
 * Connects to a live Postgres instance via the `pg` driver,
 * queries the legacy raghub `chunks` table, and writes the rows
 * out as newline-delimited JSON. The companion `migrate-import`
 * command (same file) reads the JSONL and loads it into a fresh
 * SqliteVecStore.
 *
 * Only the legacy raghub schema is supported; arbitrary pgvector
 * tables need a column-mapping flag (Phase 2 follow-up).
 */

import { createWriteStream } from 'node:fs';
import { writeFile } from 'node:fs/promises';

import {
  type ChunkModalityValue,
  ChunkModality,
} from '@raghub/core';

import type { Command } from '../runner.js';

interface PgClient {
  query: (sql: string, params?: unknown[]) => Promise<{ rows: Record<string, unknown>[] }>;
  end: () => Promise<void>;
}

interface PgModule {
  Client: new (opts: { connectionString: string }) => PgClient;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

const loadPg = async (): Promise<PgModule | null> => {
  try {
    const mod = (await dynamicImport('pg')) as { default?: PgModule } & PgModule;
    return mod.default ?? mod;
  } catch {
    return null;
  }
};

interface ExportedChunk {
  id: string;
  tenant_id: string;
  owner_id: string;
  collection_id: string;
  document_id: string;
  modality: ChunkModalityValue | string;
  text: string;
  token_count: number;
  metadata: Record<string, string>;
  embedding: number[];
  created_at: number;
}

export const pgvectorToSqliteCommand: Command = {
  name: 'migrate',
  description: 'Migrate data from legacy raghub (pgvector or JSONL).',
  usage: 'raghub migrate pgvector-to-sqlite --dsn <pg> --out chunks.ndjson',
  async run({ flags }) {
    const dsn = flags['dsn'];
    const out = flags['out'];
    if (typeof dsn !== 'string' || typeof out !== 'string') {
      console.error('--dsn and --out are required');
      return 2;
    }
    const pg = await loadPg();
    if (!pg) {
      console.error('[raghub] `pg` package not installed. Run: pnpm add pg');
      return 1;
    }
    const client = new pg.Client({ connectionString: dsn });
    try {
      await client.query('SELECT 1');
      const res = await client.query(
        `SELECT id, tenant_id, owner_id, collection_id, document_id, modality,
                text, token_count, metadata_json, embedding, created_at
         FROM chunks`,
      );
      const stream = createWriteStream(out, { encoding: 'utf8' });
      let count = 0;
      for (const row of res.rows) {
        const record: ExportedChunk = {
          id: String(row['id']),
          tenant_id: String(row['tenant_id']),
          owner_id: String(row['owner_id']),
          collection_id: String(row['collection_id']),
          document_id: String(row['document_id']),
          modality: String(row['modality'] ?? 'text'),
          text: String(row['text']),
          token_count: Number(row['token_count'] ?? 0),
          metadata: parseJson(String(row['metadata_json'] ?? '{}')),
          embedding: parseEmbedding(row['embedding']),
          created_at: Number(row['created_at'] ?? Date.now()),
        };
        stream.write(`${JSON.stringify(record)}\n`);
        count++;
      }
      await new Promise<void>((resolve, reject) => {
        stream.end((err?: Error | null) => (err ? reject(err) : resolve()));
      });
      console.log(`✓ exported ${count} chunks to ${out}`);
      return 0;
    } catch (e) {
      console.error('[raghub] migration failed:', e instanceof Error ? e.message : String(e));
      return 1;
    } finally {
      await client.end();
    }
  },
};

const parseJson = (s: string): Record<string, string> => {
  try {
    const parsed = JSON.parse(s) as unknown;
    if (parsed && typeof parsed === 'object') {
      const out: Record<string, string> = {};
      for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
        out[k] = typeof v === 'string' ? v : JSON.stringify(v);
      }
      return out;
    }
    return {};
  } catch {
    return {};
  }
};

const parseEmbedding = (raw: unknown): number[] => {
  if (Array.isArray(raw)) return raw.map((n) => Number(n));
  if (typeof raw === 'string') {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) return parsed.map((n) => Number(n));
    } catch {
      return [];
    }
  }
  return [];
};

void ChunkModality;
void writeFile;

/**
 * `raghub migrate-import --in chunks.ndjson --db ./raghub.db`
 *
 * Reads the JSONL produced by an exporter and loads it into a
 * fresh SqliteVecStore. Idempotent via the store's own
 * addById semantics (the chunk's hashed id dedupes on
 * re-import).
 */

import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

import {
  Chunk,
  ChunkModality as ChunkModalityNS,
  type ChunkId,
  type CollectionId,
  type DocumentId,
  brandId,
  type TenantId,
  type UserId,
  SqliteVecStore,
} from '@raghub/core';

const fromLine = async (line: string): Promise<Chunk | null> => {
  if (!line.trim()) return null;
  const obj = JSON.parse(line) as ExportedChunk;
  return new Chunk({
    id: brandId<ChunkId>(obj.id),
    tenantId: brandId<TenantId>(obj.tenant_id),
    ownerId: brandId<UserId>(obj.owner_id),
    collectionId: brandId<CollectionId>(obj.collection_id),
    documentId: brandId<DocumentId>(obj.document_id),
    modality: (ChunkModalityNS[obj.modality as keyof typeof ChunkModalityNS] ?? ChunkModalityNS.Text),
    text: obj.text,
    embedding: obj.embedding,
    metadata: obj.metadata,
    tokenCount: obj.token_count,
    createdAt: new Date(obj.created_at),
  });
};

export const sqliteImportCommand: Command = {
  name: 'migrate-import',
  description: 'Import chunks from a JSONL export into sqlite-vec.',
  usage: 'raghub migrate-import --in chunks.ndjson --db ./raghub.db',
  async run({ flags, cwd }) {
    const inputPath = flags['in'];
    const dbPath = flags['db'];
    if (typeof inputPath !== 'string' || typeof dbPath !== 'string') {
      console.error('--in and --db are required');
      return 2;
    }
    const absDb = dbPath.startsWith('/') ? dbPath : `${cwd}/${dbPath}`;
    const store = new SqliteVecStore({ path: absDb });
    const stream = createReadStream(inputPath);
    const rl = createInterface({ input: stream, crlfDelay: Infinity });
    let count = 0;
    try {
      for await (const line of rl) {
        const chunk = await fromLine(line);
        if (!chunk) continue;
        await store.add(chunk);
        count++;
      }
    } finally {
      await store.close();
    }
    console.log(`✓ imported ${count} chunks into ${absDb}`);
    return 0;
  },
};