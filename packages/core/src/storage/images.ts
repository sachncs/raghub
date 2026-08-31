/**
 * Content-addressable image storage on the local filesystem.
 *
 * Images are stored by their SHA-256 content hash under
 * `<base_path>/<hash[:2]>/<hash><extension>`. The two-character
 * prefix subdirectory keeps any single directory from growing
 * unboundedly.
 */

import { createHash } from 'node:crypto';
import { promises as fs } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

export interface ImageStore {
  save(bytes: Uint8Array, extension?: string): Promise<string>;
  get(hash: string): Promise<Uint8Array | null>;
  exists(hash: string): Promise<boolean>;
  delete(hash: string): Promise<boolean>;
}

export interface ImageSaveOptions {
  readonly extension?: string;
  readonly root?: string;
}

const DEFAULT_ROOT = './data/images';

export class FsImageStore implements ImageStore {
  private readonly root: string;

  constructor(opts: ImageSaveOptions = {}) {
    this.root = resolve(opts.root ?? DEFAULT_ROOT);
  }

  async save(bytes: Uint8Array, extension: string = '.bin'): Promise<string> {
    const hash = createHash('sha256').update(bytes).digest('hex');
    const dir = join(this.root, hash.slice(0, 2));
    const path = join(dir, hash + extension);
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(path, bytes, { flag: 'wx' });
    return hash;
  }

  async get(hash: string): Promise<Uint8Array | null> {
    const file = await this.resolvePath(hash);
    if (!file) return null;
    return new Uint8Array(await fs.readFile(file));
  }

  async exists(hash: string): Promise<boolean> {
    const file = await this.resolvePath(hash);
    return file !== null;
  }

  async delete(hash: string): Promise<boolean> {
    const file = await this.resolvePath(hash);
    if (!file) return false;
    await fs.unlink(file);
    return true;
  }

  private async resolvePath(hash: string): Promise<string | null> {
    const safe = hash.replace(/[^a-f0-9]/gi, '');
    if (safe.length < 2) return null;
    const dir = join(this.root, safe.slice(0, 2));
    const entries = await fs.readdir(dir).catch(() => []);
    for (const name of entries) {
      if (name.startsWith(safe)) {
        return join(dir, name);
      }
    }
    return null;
  }
}

export class InMemoryImageStore implements ImageStore {
  private readonly entries = new Map<string, Uint8Array>();

  async save(bytes: Uint8Array, extension: string = '.bin'): Promise<string> {
    const hash = createHash('sha256').update(bytes).digest('hex');
    this.entries.set(hash + extension, bytes);
    return hash;
  }

  async get(hash: string): Promise<Uint8Array | null> {
    const safe = hash.replace(/[^a-f0-9]/gi, '');
    for (const [key, value] of this.entries) {
      if (key.startsWith(safe)) return value;
    }
    return null;
  }

  async exists(hash: string): Promise<boolean> {
    return (await this.get(hash)) !== null;
  }

  async delete(hash: string): Promise<boolean> {
    const safe = hash.replace(/[^a-f0-9]/gi, '');
    let removed = false;
    for (const key of Array.from(this.entries.keys())) {
      if (key.startsWith(safe)) {
        this.entries.delete(key);
        removed = true;
      }
    }
    return removed;
  }
}