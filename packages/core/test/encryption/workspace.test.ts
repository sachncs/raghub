import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  openEncryptedWorkspace,
  type WorkspaceWithSettings,
} from '../../src/encryption.js';

const PATH = ':memory:';

let tempCounter = 0;
const mkTempPath = async (): Promise<string> => {
  const dir = mkdtempSync(join(tmpdir(), 'raghub-enc-'));
  return join(dir, `workspace-${++tempCounter}.db`);
};
const rmTempPath = async (path: string): Promise<void> => {
  rmSync(join(path, '..'), { recursive: true, force: true });
};

describe('openEncryptedWorkspace', () => {
  let workspace: WorkspaceWithSettings | null = null;

  afterEach(() => {
    workspace?.close();
    workspace = null;
  });

  beforeEach(() => {});

  it('opens in plaintext mode when no passphrase is given', async () => {
    workspace = await openEncryptedWorkspace({ path: PATH });
    expect(workspace.encryption).toBe('plaintext');
    await workspace.settings.set('llm', { provider: 'openai', model: 'gpt-4.1' });
    const round = await workspace.settings.get<{ provider: string; model: string }>('llm');
    expect(round?.model).toBe('gpt-4.1');
  });

  it('opens in passphrase mode, persists ciphertext, survives reload', async () => {
    const tempPath = await mkTempPath();
    try {
      workspace = await openEncryptedWorkspace({ path: tempPath, passphrase: 'correct horse battery staple' });
      expect(workspace.encryption).toBe('passphrase-aes-256-gcm');
      await workspace.settings.set('workspace.name', 'Acme');
      await workspace.settings.set('llm', { provider: 'openai', model: 'gpt-4.1' });
      const before = await workspace.settings.get<{ provider: string; model: string }>('llm');
      expect(before?.model).toBe('gpt-4.1');
      workspace.close();
      workspace = null;

      // Reopen with the same passphrase — must round-trip.
      workspace = await openEncryptedWorkspace({ path: tempPath, passphrase: 'correct horse battery staple' });
      expect(workspace.encryption).toBe('passphrase-aes-256-gcm');
      const after = await workspace.settings.get<{ provider: string; model: string }>('llm');
      expect(after?.model).toBe('gpt-4.1');
    } finally {
      await rmTempPath(tempPath);
    }
  });

  it('rejects the wrong passphrase', async () => {
    const tempPath = await mkTempPath();
    try {
      workspace = await openEncryptedWorkspace({ path: tempPath, passphrase: 'right one' });
      await workspace.settings.set('k', 'v');
      workspace.close();
      workspace = null;
      await expect(
        openEncryptedWorkspace({ path: tempPath, passphrase: 'wrong one' }),
      ).rejects.toThrow(/passphrase/);
    } finally {
      await rmTempPath(tempPath);
    }
  });

  it('returns an empty object when no settings have been written', async () => {
    workspace = await openEncryptedWorkspace({ path: PATH });
    const all = await workspace.settings.all();
    expect(Object.keys(all)).toEqual([]);
  });

  it('supports delete', async () => {
    workspace = await openEncryptedWorkspace({ path: PATH });
    await workspace.settings.set('k', 'v');
    await workspace.settings.delete('k');
    expect(await workspace.settings.get('k')).toBeNull();
  });
});
