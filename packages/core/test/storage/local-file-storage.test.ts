import { describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { FsLocalFileStorage, InMemoryLocalFileStorage } from '../../src/storage/local-file-storage.js';

describe('InMemoryLocalFileStorage', () => {
  it('round-trips a string', async () => {
    const fs_ = new InMemoryLocalFileStorage();
    await fs_.put('a/b/c.txt', 'hello');
    const v = await fs_.get('a/b/c.txt');
    expect(v?.toString()).toBe('hello');
  });

  it('round-trips a Buffer', async () => {
    const fs_ = new InMemoryLocalFileStorage();
    const buf = Buffer.from([1, 2, 3, 4]);
    await fs_.put('bin.dat', buf);
    const v = await fs_.get('bin.dat');
    expect(Buffer.isBuffer(v)).toBe(true);
    expect((v as Buffer).equals(buf)).toBe(true);
  });

  it('delete removes a key', async () => {
    const fs_ = new InMemoryLocalFileStorage();
    await fs_.put('x', 'x');
    await fs_.delete('x');
    expect(await fs_.get('x')).toBeNull();
  });

  it('list with prefix filters', async () => {
    const fs_ = new InMemoryLocalFileStorage();
    await fs_.put('snap/a.json', '{}');
    await fs_.put('snap/b.json', '{}');
    await fs_.put('other.json', '{}');
    const list = await fs_.list('snap/');
    expect(list.length).toBe(2);
  });
});

describe('FsLocalFileStorage', () => {
  it('round-trips a string on disk', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'raghub-lfs-'));
    try {
      const fs_ = new FsLocalFileStorage({ root: dir });
      await fs_.put('session_1/state.json', '{"turn":1}');
      const v = await fs_.get('session_1/state.json');
      expect(v?.toString()).toBe('{"turn":1}');
      expect(await fs_.exists('session_1/state.json')).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('rejects path traversal', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'raghub-lfs-'));
    try {
      const fs_ = new FsLocalFileStorage({ root: dir });
      await expect(fs_.put('../escape', 'x')).rejects.toThrow(/unsafe key/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('list enumerates files recursively', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'raghub-lfs-'));
    try {
      const fs_ = new FsLocalFileStorage({ root: dir });
      await fs_.put('a/b/x.txt', 'x');
      await fs_.put('a/b/y.txt', 'y');
      const list = await fs_.list();
      expect(list.length).toBe(2);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});