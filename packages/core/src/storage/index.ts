/**
 * Storage barrel — every persistent store the framework owns.
 *
 * Phase 1 surface: users, documents, jobs, sessions, conversations.
 * Document store does not own chunks (the vector store does) — it
 * tracks metadata + lifecycle.
 */

export type { UserStore } from './users.js';
export { SqliteUserStore } from './users.js';
export type { SqliteUserStoreOptions } from './users.js';

export type { DocumentStore } from './documents.js';
export { SqliteDocumentStore } from './documents.js';
export type { SqliteDocumentStoreOptions } from './documents.js';

export type { JobQueue, JobRecord, JobStatusValue } from './jobs.js';
export { JobStatus, SqliteJobQueue } from './jobs.js';
export type { SqliteJobQueueOptions } from './jobs.js';

export type { SessionRecord, SessionStore } from './sessions.js';
export { SqliteSessionStore } from './sessions.js';
export type { SqliteSessionStoreOptions } from './sessions.js';

export type { ConversationStore, TurnInput } from './conversations.js';
export { SqliteConversationStore } from './conversations.js';
export type { SqliteConversationStoreOptions } from './conversations.js';

export type { WorkspaceMember, WorkspaceMemberStore } from './workspace-member.js';
export {
  SqliteWorkspaceMemberStore,
  WorkspaceMemberRole,
  canManageWorkspace,
  canIngest,
  resolveRoleFor,
} from './workspace-member.js';
export type {
  WorkspaceMemberRoleValue,
  SqliteWorkspaceMemberStoreOptions,
} from './workspace-member.js';

export type { Role, RoleAssignment, Group, GroupMembership, RoleStore, GroupStore } from './groups.js';
export { SqliteRoleStore, SqliteGroupStore } from './groups.js';
export type { SqliteRoleStoreOptions, SqliteGroupStoreOptions } from './groups.js';