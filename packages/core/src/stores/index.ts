/**
 * Store barrel.
 */

export type {
  KeywordHit,
  KeywordSearchOptions,
  StoreFilter,
  VectorSearchOptions,
  VectorStore,
} from './types.js';
export { SqliteVecStore, loadSqliteVecExtension } from './sqlite-vec.js';
export type { SqliteVecStoreOptions } from './sqlite-vec.js';