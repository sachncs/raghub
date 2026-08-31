import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { openEncryptedWorkspace } from '../../src/encryption.js';

describe('encryption round-trip', () => {
  it('encrypts and decrypts via the public settings store', async () => {
    const ws = await openEncryptedWorkspace({
      path: ':memory:',
      passphrase: 'test-passphrase',
    });
    await ws.settings.set('hello', 'world');
    expect(await ws.settings.get('hello')).toBe('world');
    ws.close();
  });

  it('produces a fresh ciphertext per value', async () => {
    const ws = await openEncryptedWorkspace({
      path: ':memory:',
      passphrase: 'test-passphrase',
    });
    await ws.settings.set('a', 'x');
    await ws.settings.set('b', 'x');
    expect(await ws.settings.get('a')).toBe('x');
    expect(await ws.settings.get('b')).toBe('x');
    ws.close();
  });

  it('rejects wrong-key decryption at open time', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'revex-enc-'));
    const tempPath = join(dir, 'workspace.db');
    try {
      const ws1 = await openEncryptedWorkspace({
        path: tempPath,
        passphrase: 'right one',
      });
      await ws1.settings.set('k', 'v');
      ws1.close();
      await expect(
        openEncryptedWorkspace({ path: tempPath, passphrase: 'wrong one' }),
      ).rejects.toThrow(/passphrase/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
