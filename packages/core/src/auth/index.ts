/**
 * Auth barrel.
 *
 * BcryptHasher + JwtService are the only public auth surfaces; the
 * rest of the package composes them. The legacy Python `AuthService`
 * class is rewritten in @revex/api where it can talk to the user
 * store; core only provides the primitives.
 */

export { BcryptHasher } from './password.js';
export type { PasswordHasher } from './password.js';
export { JwtService } from './jwt.js';
export type { JwtAlgorithmValue, JwtClaims, JwtConfig, MintOptions } from './jwt.js';