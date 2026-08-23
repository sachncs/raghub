/**
 * Cryptographic key generators used by `raghub init` to seed
 * `.raghub/.env` with secrets when the user has not provided any.
 *
 * Uses `node:crypto`'s `randomBytes` — no third-party dep.
 */

import { randomBytes } from 'node:crypto';

export const generateKey = (bytes: number): string => randomBytes(bytes).toString('hex');

export const generateJwtSecret = (): string =>
  randomBytes(48).toString('base64').replace(/=+$/, '').slice(0, 64);