/**
 * Document version registry (JSON-backed).
 *
 * A document's history is an append-only list of versions.
 * Each new version that exceeds the previous latest's number
 * auto-archives the prior latest. The store is workspace-scoped
 * and persists to a single JSON file under the workspace home.
 *
 * This is the in-memory + JSON-on-disk complement to the SQLite
 * `SqliteDocumentStore`. The SQLite store tracks the latest
 * version; the version registry preserves the audit trail.
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';

import { brandId, type DocumentId, type WorkspaceId } from '../domain/index.js';

export interface DocumentVersionRecord {
  readonly id: string;
  readonly workspaceId: WorkspaceId;
  readonly documentId: DocumentId;
  readonly version: number;
  readonly filename: string;
  readonly checksum: string;
  readonly byteSize: number;
  readonly status: 'ready' | 'failed' | 'superseded' | 'current';
  readonly createdAt: string;
  readonly note: string | null;
}

export interface VersionRegistrySnapshot {
  readonly versions: readonly DocumentVersionRecord[];
}

export class DocumentVersionRegistry {
  private readonly file: string;
  private records: DocumentVersionRecord[] = [];
  private loaded = false;

  constructor(homeDir: string) {
    this.file = resolve(homeDir, 'document-versions.json');
  }

  async load(): Promise<void> {
    if (this.loaded) return;
    try {
      const raw = await fs.readFile(this.file, 'utf8');
      this.records = JSON.parse(raw) as DocumentVersionRecord[];
    } catch {
      this.records = [];
    }
    this.loaded = true;
  }

  async save(): Promise<void> {
    await fs.mkdir(dirname(this.file), { recursive: true });
    await fs.writeFile(
      this.file,
      JSON.stringify({ versions: this.records }, null, 2),
      'utf8',
    );
  }

  async appendVersion(input: {
    readonly workspaceId: WorkspaceId;
    readonly documentId: DocumentId;
    readonly filename: string;
    readonly checksum: string;
    readonly byteSize: number;
    readonly note?: string;
  }): Promise<DocumentVersionRecord> {
    await this.load();
    const existing = this.records.filter((r) => r.documentId === input.documentId);
    const latestVersion = existing.reduce((m, r) => Math.max(m, r.version), 0);
    for (const r of existing) {
      if (r.status === 'current') {
        this.records = this.records.map((row) =>
          row.id === r.id ? { ...row, status: 'superseded' as const } : row,
        );
      }
    }
    const rec: DocumentVersionRecord = {
      id: brandId('v'),
      workspaceId: input.workspaceId,
      documentId: input.documentId,
      version: latestVersion + 1,
      filename: input.filename,
      checksum: input.checksum,
      byteSize: input.byteSize,
      status: 'current',
      createdAt: new Date().toISOString(),
      note: input.note ?? null,
    };
    this.records.push(rec);
    await this.save();
    return rec;
  }

  async list(workspaceId: WorkspaceId, documentId: DocumentId): Promise<readonly DocumentVersionRecord[]> {
    await this.load();
    return this.records.filter(
      (r) => r.workspaceId === workspaceId && r.documentId === documentId,
    );
  }
}