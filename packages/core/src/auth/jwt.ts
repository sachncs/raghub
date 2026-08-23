/**
 * JWT mint + verify (HS256 default).
 *
 * Uses `jose` for sign/verify. `aud` and `iss` are optional but
 * recommended; defaults are empty. `sub` carries the user id.
 */

import { jwtVerify, SignJWT } from 'jose';

import { AuthError, ConfigurationError } from '../errors/index.js';

export const JwtAlgorithm = {
  HS256: 'HS256',
  HS384: 'HS384',
  HS512: 'HS512',
} as const;
export type JwtAlgorithmValue = (typeof JwtAlgorithm)[keyof typeof JwtAlgorithm];

export interface JwtClaims {
  readonly sub: string;
  readonly tenant_id: string;
  readonly is_admin: boolean;
  readonly exp: number;
  readonly iat?: number;
}

export interface JwtConfig {
  readonly secret: string;
  readonly algorithm: JwtAlgorithmValue;
  readonly ttlSeconds: number;
  readonly issuer?: string;
  readonly audience?: string;
}

export interface MintOptions {
  readonly subject: string;
  readonly tenantId: string;
  readonly isAdmin: boolean;
}

const ALG_TO_JOSE: Record<JwtAlgorithmValue, string> = {
  HS256: 'HS256',
  HS384: 'HS384',
  HS512: 'HS512',
};

const encoder = new TextEncoder();

export class JwtService {
  private readonly secret: Uint8Array;
  private readonly cfg: JwtConfig;

  constructor(cfg: JwtConfig) {
    if (!cfg.secret || cfg.secret.length < 16) {
      throw new ConfigurationError('jwt secret must be at least 16 characters');
    }
    this.cfg = cfg;
    this.secret = encoder.encode(cfg.secret);
  }

  public async mint(opts: MintOptions): Promise<string> {
    const now = Math.floor(Date.now() / 1000);
    let builder = new SignJWT({
      tenant_id: opts.tenantId,
      is_admin: opts.isAdmin,
    }).setProtectedHeader({ alg: ALG_TO_JOSE[this.cfg.algorithm], typ: 'JWT' });
    builder = builder.setSubject(opts.subject).setIssuedAt(now).setExpirationTime(now + this.cfg.ttlSeconds);
    if (this.cfg.issuer) builder = builder.setIssuer(this.cfg.issuer);
    if (this.cfg.audience) builder = builder.setAudience(this.cfg.audience);
    return builder.sign(this.secret);
  }

  public async verify(token: string): Promise<JwtClaims> {
    try {
      const verifyOpts: { algorithms: string[]; issuer?: string; audience?: string } = {
        algorithms: [ALG_TO_JOSE[this.cfg.algorithm]],
      };
      if (this.cfg.issuer) verifyOpts.issuer = this.cfg.issuer;
      if (this.cfg.audience) verifyOpts.audience = this.cfg.audience;
      const { payload } = await jwtVerify(token, this.secret, verifyOpts);
      if (typeof payload.sub !== 'string' || typeof payload['tenant_id'] !== 'string') {
        throw new AuthError('jwt missing required claims');
      }
      return {
        sub: payload.sub,
        tenant_id: String(payload['tenant_id']),
        is_admin: Boolean(payload['is_admin']),
        exp: Number(payload.exp),
        ...(typeof payload.iat === 'number' ? { iat: payload.iat } : {}),
      };
    } catch (err) {
      throw new AuthError('invalid or expired token', { cause: err });
    }
  }
}