/**
 * Storage barrel — user store is the only Phase 1 surface. Document
 * store, feedback store, session store are added in subsequent
 * commits as the API surface grows.
 */

export type { UserStore } from './users.js';
export { SqliteUserStore } from './users.js';
export type { SqliteUserStoreOptions } from './users.js';