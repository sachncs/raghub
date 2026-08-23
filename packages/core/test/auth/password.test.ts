import { describe, expect, it } from 'vitest';

import { BcryptHasher } from '../../src/auth/password.js';
import { ConfigurationError } from '../../src/errors/index.js';

describe('BcryptHasher', () => {
  it('rejects out-of-range rounds', () => {
    expect(() => new BcryptHasher(2)).toThrow(ConfigurationError);
    expect(() => new BcryptHasher(20)).toThrow(ConfigurationError);
  });

  it('hashes and verifies a password', async () => {
    const h = new BcryptHasher(4);
    const hashed = await h.hash('correct horse battery staple');
    expect(hashed).toMatch(/^\$2[aby]\$/);
    expect(await h.verify('correct horse battery staple', hashed)).toBe(true);
    expect(await h.verify('wrong', hashed)).toBe(false);
  });

  it('rejects empty plaintext', async () => {
    const h = new BcryptHasher(4);
    await expect(h.hash('')).rejects.toThrow(/password/);
  });

  it('returns false on empty verify inputs', async () => {
    const h = new BcryptHasher(4);
    expect(await h.verify('', 'hash')).toBe(false);
    expect(await h.verify('plain', '')).toBe(false);
  });
});