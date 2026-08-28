/**
 * LocalFileStorage — a tiny key-value store backed by the local
 * filesystem, used for session snapshots, conversation history that
 * has spilled out of the prompt window, document bytes that the
 * background ingest worker reads, and any other large blob that
 * doesn't belong in `workspace.db`.
 *
 * Keys map to relative file paths under `root`; values are stored
 * verbatim (Buffer for binary, UTF-8 string for text). The store is
 * process-safe via atomic write (write to tmp + rename).
 */

import { existsSync, mkdirSync, promises as fs, renameSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

import type { DocumentId, WorkspaceId } from '../domain/index.js';

export interface LocalFileStorageOptions {
  readonly root: string;
}

export interface FileStat {
  readonly key: string;
  readonly size: number;
  readonly modifiedAt: Date;
}

export interface LocalFileStorage {
  put(key: string, value: string | Buffer): Promise<void>;
  get(key: string): Promise<string | Buffer | null>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
  list(prefix?: string): Promise<readonly FileStat[]>;
  close(): Promise<void>;
}

const safePath = (root: string, key: string): string => {
  if (key.includes('..') || key.startsWith('/') || key.startsWith('\\')) {
    throw new Error(`unsafe key: ${key}`);
  }
  return resolve(join(root, key));
};

export class FsLocalFileStorage implements LocalFileStorage {
  private readonly root: string;
  constructor(opts: LocalFileStorageOptions) {
    this.root = resolve(opts.root);
    if (!existsSync(this.root)) mkdirSync(this.root, { recursive: true });
  }

  public async put(key: string, value: string | Buffer): Promise<void> {
    const target = safePath(this.root, key);
    mkdirSync(dirname(target), { recursive: true });
    const tmp = `${target}.tmp.${process.pid}`;
    await fs.writeFile(tmp, value);
    renameSync(tmp, target);
  }

  public async get(key: string): Promise<string | Buffer | null> {
    const target = safePath(this.root, key);
    if (!existsSync(target)) return null;
    return fs.readFile(target);
  }

  public async delete(key: string): Promise<void> {
    const target = safePath(this.root, key);
    if (existsSync(target)) rmSync(target);
  }

  public async exists(key: string): Promise<boolean> {
    return existsSync(safePath(this.root, key));
  }

  public async list(prefix?: string): Promise<readonly FileStat[]> {
    const start = prefix !== undefined ? safePath(this.root, prefix) : this.root;
    const out: FileStat[] = [];
    const stack = [start];
    while (stack.length > 0) {
      const cur = stack.pop() as string;
      let entries: import('node:fs').Dirent[];
      try {
        entries = await fs.readdir(cur, { withFileTypes: true });
      } catch {
        continue;
      }
      for (const e of entries) {
        const full = join(cur, e.name);
        if (e.isDirectory()) {
          stack.push(full);
        } else if (e.isFile()) {
          const rel = full.slice(this.root.length + 1);
          const s = statSync(full);
          out.push({ key: rel, size: s.size, modifiedAt: s.mtime });
        }
      }
    }
    return out.sort((a, b) => a.key.localeCompare(b.key));
  }

  public async close(): Promise<void> {
    // No-op: the FS is shared; closing is a noop.
  }
}

/** In-memory implementation, useful for tests. */
export class InMemoryLocalFileStorage implements LocalFileStorage {
  private readonly files = new Map<string, { value: string | Buffer; modifiedAt: Date }>();
  public async put(key: string, value: string | Buffer): Promise<void> {
    this.files.set(key, { value, modifiedAt: new Date() });
  }
  public async get(key: string): Promise<string | Buffer | null> {
    return this.files.get(key)?.value ?? null;
  }
  public async delete(key: string): Promise<void> {
    this.files.delete(key);
  }
  public async exists(key: string): Promise<boolean> {
    return this.files.has(key);
  }
  public async list(prefix?: string): Promise<readonly FileStat[]> {
    const out: FileStat[] = [];
    for (const [k, v] of this.files.entries()) {
      if (prefix !== undefined && !k.startsWith(prefix)) continue;
      out.push({ key: k, size: v.value.length, modifiedAt: v.modifiedAt });
    }
    return out.sort((a, b) => a.key.localeCompare(b.key));
  }
  public async close(): Promise<void> {
    this.files.clear();
  }
}

/**
 * Stable path layout under the file storage root. Callers (api
 * server, ingest worker) use these helpers instead of hand-rolling
 * paths so the layout is auditable.
 *
 *   documents/<workspaceId>/<documentId>.bin    raw upload bytes
 *   snapshots/<workspaceId>/<sessionId>.json    session snapshots
 *   spillover/<workspaceId>/<conversationId>/  conversation spillover
 */
export const documentBytesKey = (workspaceId: WorkspaceId, documentId: DocumentId): string =>
  `documents/${workspaceId}/${documentId}.bin`;

export const sessionSnapshotKey = (workspaceId: WorkspaceId, sessionId: string): string =>
  `snapshots/${workspaceId}/${sessionId}.json`;

export const conversationSpilloverPrefix = (workspaceId: WorkspaceId): string =>
  `spillover/${workspaceId}/`;