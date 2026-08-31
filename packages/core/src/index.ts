/**
 * @revex/core — public surface.
 *
 * Domain types, errors, settings, stores, embedder, retrieval, auth,
 * workspaces, telemetry, storage, LLM, chunker, ingest, workspace.
 */

export * from './errors/index.js';
export * from './domain/index.js';
export * from './settings/index.js';
export * from './embedder/index.js';
export * from './stores/index.js';
export * from './retrieval/index.js';
export * from './auth/index.js';
export * from './workspaces/index.js';
export * from './telemetry/index.js';
export * from './storage/index.js';
export * from './llm/index.js';
export * from './chunker/index.js';
export * from './ingest.js';
export * from './context/index.js';
export * from './ingest-verbose.js';
export * from './ingest-agentic.js';
export * from './web/search.js';
export * from './graph/store.js';
export {
  EXTRACT_PROMPT,
  COMMUNITY_PROMPT,
  extractEntities,
  summariseCommunity,
  clusterEntities,
} from './graph/index.js';
export type {
  ExtractedEntity,
  ExtractedTriple,
  ExtractedGraph,
} from './graph/index.js';
export * from './feedback/index.js';
export * from './summary/index.js';
export * from './traces/corpus.js';
export type { WorkspaceHandle } from './workspace.js';
export { openWorkspace } from './workspace.js';
export { openEncryptedWorkspace } from './encryption.js';
export type {
  EncryptedField,
  WorkspaceSettingsStore,
  WorkspaceWithSettings,
  OpenWorkspaceOptions,
} from './encryption.js';

export type {
  WorkspaceRegistry,
  WorkspaceRegistryEntry,
  FileWorkspaceRegistryOptions,
} from './workspace-registry.js';
export {
  defaultRegistryPath,
  openFileWorkspaceRegistry,
} from './workspace-registry.js';

export { MIGRATIONS, runMigrations, lastAppliedId } from './migrations.js';
export type { Migration } from './migrations.js';

export * from './plugins/index.js';