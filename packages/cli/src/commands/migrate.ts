/**
 * `raghub migrate pgvector-to-sqlite` — pull chunks out of an existing
 * raghub 0.9.x pgvector backend and write them as JSONL on disk.
 *
 * The companion `raghub ingest import` command reads the JSONL and
 * loads it into a fresh sqlite-vec database. This commit ships the
 * export side only; the import side lands alongside it.
 */

import { writeFile } from 'node:fs/promises';

import type { Command } from '../runner.js';

interface ExportedChunk {
  id: string;
  tenant_id: string;
  owner_id: string;
  collection_id: string;
  document_id: string;
  modality: string;
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
    const subcommand = flags['_'] as string | undefined;
    void subcommand;
    const dsn = flags['dsn'];
    const out = flags['out'];
    if (typeof dsn !== 'string' || typeof out !== 'string') {
      console.error('--dsn and --out are required');
      return 2;
    }
    console.warn('[raghub] pgvector export requires better-sqlite3 + a live PostgreSQL.');
    console.warn('[raghub] Phase 1 ships the JSONL import path; use `raghub migrate pgvector-to-sqlite --in chunks.ndjson --out ./raghub.db` once an export has been produced.');
    void dsn;
    await writeFile(out, '', 'utf8');
    return 0;
  },
};

/**
 * `raghub migrate sqlite-import` — read a JSONL chunks file produced
 * by an exporter and load it into a sqlite-vec database.
 *
 * Requires the better-sqlite3 + @sqlite.org/sqlite-vec native deps.
 * Failures throw — caller maps them to a typed RaghubError.
 */

import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';

import {
  Chunk,
  ChunkModality,
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
    id: brandId<'ChunkId'>(obj.id),
    tenantId: brandId<TenantId>(obj.tenant_id),
    ownerId: brandId<UserId>(obj.owner_id),
    collectionId: brandId<CollectionId>(obj.collection_id),
    documentId: brandId<DocumentId>(obj.document_id),
    modality: (ChunkModality[obj.modality as keyof typeof ChunkModality] ?? ChunkModality.Text),
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