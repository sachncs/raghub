import { describe, expect, it } from 'vitest';

import { JwtService } from '../../src/auth/jwt.js';
import { AuthError, ConfigurationError } from '../../src/errors/index.js';

const make = (overrides: Partial<{ secret: string; ttl: number }> = {}) =>
  new JwtService({
    secret: overrides.secret ?? 'a'.repeat(32),
    algorithm: 'HS256',
    ttlSeconds: overrides.ttl ?? 60,
  });

describe('JwtService', () => {
  it('mints a token with the expected claims', async () => {
    const jwt = make();
    const tok = await jwt.mint({ subject: 'usr_1', tenantId: 'tnt_1', isAdmin: true });
    const claims = await jwt.verify(tok);
    expect(claims.sub).toBe('usr_1');
    expect(claims.tenant_id).toBe('tnt_1');
    expect(claims.is_admin).toBe(true);
    expect(claims.exp).toBeGreaterThan(Math.floor(Date.now() / 1000));
  });

  it('rejects short secrets at construction', () => {
    expect(() => new JwtService({ secret: 'short', algorithm: 'HS256', ttlSeconds: 60 })).toThrow(
      ConfigurationError,
    );
  });

  it('throws AuthError on tampered token', async () => {
    const jwt = make();
    const tok = await jwt.mint({ subject: 'usr_1', tenantId: 'tnt_1', isAdmin: false });
    await expect(jwt.verify(tok.slice(0, -2) + 'aa')).rejects.toBeInstanceOf(AuthError);
  });

  it('rejects expired tokens', async () => {
    const jwt = make({ ttl: 1 });
    const tok = await jwt.mint({ subject: 'usr_1', tenantId: 'tnt_1', isAdmin: false });
    await new Promise((r) => setTimeout(r, 1100));
    await expect(jwt.verify(tok)).rejects.toBeInstanceOf(AuthError);
  });
});