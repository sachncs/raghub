/**
 * Domain barrel.
 *
 * Each file is the TS counterpart of a Python `@dataclass(slots=True,
 * frozen=True)`. Everything is a frozen class wrapping an internal
 * `props` bag, never a free-form `interface` exposed publicly.
 */

export * from './ids.js';
export * from './tenant.js';
export * from './user.js';
export * from './document.js';
export * from './chunk.js';
export * from './turn.js';