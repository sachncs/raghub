/**
 * Password hashing — bcrypt with a configurable cost factor.
 *
 * Pure-JS `bcryptjs` is used to avoid native compilation in the
 * sandbox; production deployments with `bcrypt` (native) drop in
 * the same `PasswordHasher` interface.
 */

import { AuthError, ConfigurationError, MissingDepError } from '../errors/index.js';

export interface PasswordHasher {
  hash(plaintext: string): Promise<string>;
  verify(plaintext: string, hashed: string): Promise<boolean>;
}

interface BcryptModule {
  hash: (s: string, rounds: number) => Promise<string>;
  compare: (s: string, hashed: string) => Promise<boolean>;
}

const dynamicImport = (spec: string): Promise<unknown> => import(spec);

let cached: BcryptModule | null = null;

const loadBcrypt = async (): Promise<BcryptModule> => {
  if (cached) return cached;
  try {
    const mod = (await dynamicImport('bcryptjs')) as { default: BcryptModule };
    cached = mod.default;
    return cached;
  } catch (cause) {
    throw new MissingDepError('bcryptjs is not installed', {
      cause,
      details: { hint: 'pnpm add bcryptjs @types/bcryptjs' },
    });
  }
};

export class BcryptHasher implements PasswordHasher {
  private readonly rounds: number;

  constructor(rounds: number = 10) {
    if (rounds < 4 || rounds > 15) {
      throw new ConfigurationError(`bcrypt rounds out of range: ${rounds}`, {
        details: { allowed: [4, 15] },
      });
    }
    this.rounds = rounds;
  }

  public async hash(plaintext: string): Promise<string> {
    if (!plaintext) throw new AuthError('password must not be empty');
    const bcrypt = await loadBcrypt();
    return bcrypt.hash(plaintext, this.rounds);
  }

  public async verify(plaintext: string, hashed: string): Promise<boolean> {
    if (!plaintext || !hashed) return false;
    const bcrypt = await loadBcrypt();
    return bcrypt.compare(plaintext, hashed);
  }
}