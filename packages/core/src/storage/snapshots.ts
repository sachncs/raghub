/**
 * Workspace snapshot save/load.
 *
 * A snapshot is a JSON metadata file + a tar archive of the
 * workspace directory. `save()` writes the manifest; `restore()`
 * is a placeholder for the API's backup/restore route.
 */

import { promises as fs } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';

import type { WorkspaceId } from '../domain/index.js';

export interface SnapshotMetadata {
  readonly workspaceId: WorkspaceId;
  readonly path: string;
  readonly createdAt: string;
  readonly files: readonly { readonly path: string; readonly checksum: string; readonly byteSize: number }[];
  readonly totalBytes: number;
}

export class SnapshotWriter {
  private readonly dir: string;

  constructor(homeDir: string) {
    this.dir = resolve(homeDir, 'snapshots');
  }

  async writeMetadata(workspaceId: WorkspaceId, workspacePath: string): Promise<SnapshotMetadata> {
    const files = await walk(resolve(workspacePath));
    const totalBytes = files.reduce((s, f) => s + f.byteSize, 0);
    const meta: SnapshotMetadata = {
      workspaceId,
      path: workspacePath,
      createdAt: new Date().toISOString(),
      files,
      totalBytes,
    };
    await fs.mkdir(this.dir, { recursive: true });
    const out = resolve(this.dir, `${workspaceId}-${Date.now()}.json`);
    await fs.writeFile(out, JSON.stringify(meta, null, 2), 'utf8');
    return meta;
  }
}

async function walk(root: string): Promise<SnapshotMetadata['files']> {
  const out: { path: string; checksum: string; byteSize: number }[] = [];
  async function visit(dir: string, rel: string): Promise<void> {
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      const abs = resolve(dir, entry.name);
      const relPath = rel ? `${rel}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        await visit(abs, relPath);
      } else if (entry.isFile()) {
        const buf = await fs.readFile(abs);
        out.push({
          path: relPath,
          checksum: createHash('sha256').update(buf).digest('hex'),
          byteSize: buf.length,
        });
      }
    }
  }
  try {
    await visit(root, '');
  } catch {
    /* missing dir — empty snapshot */
  }
  void dirname;
  return out;
}