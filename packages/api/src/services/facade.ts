/**
 * ApplicationFacade — top-level service entrypoint.
 *
 * Aggregates every per-handle store into one ergonomic facade.
 * Mirrors the legacy Python `services.facade.ApplicationFacade`.
 * Production code that wants a "do everything" service should
 * depend on this rather than reaching into individual stores.
 */

import type {
  AuditEventStore,
  ConversationStore,
  DocumentPrincipalStore,
  DocumentStore,
  Embedder,
  JobQueue,
  SessionStore,
  SqliteFeedbackStore,
  UserStore,
  VectorStore,
  WorkspaceId,
  WorkspaceMemberStore,
  WorkspaceMemoryStore,
} from '@revex/core';

export interface ApplicationFacade {
  readonly workspaceId: WorkspaceId;
  readonly userId: string;
  readonly users: UserStore;
  readonly documents: DocumentStore;
  readonly documentPrincipals: DocumentPrincipalStore;
  readonly members: WorkspaceMemberStore;
  readonly sessions: SessionStore;
  readonly conversations: ConversationStore;
  readonly jobs: JobQueue;
  readonly audit: AuditEventStore;
  readonly memory: WorkspaceMemoryStore;
  readonly feedback: SqliteFeedbackStore;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
  close(): Promise<void>;
}

export const createFacade = (input: {
  readonly workspaceId: WorkspaceId;
  readonly userId: string;
  readonly users: UserStore;
  readonly documents: DocumentStore;
  readonly documentPrincipals: DocumentPrincipalStore;
  readonly members: WorkspaceMemberStore;
  readonly sessions: SessionStore;
  readonly conversations: ConversationStore;
  readonly jobs: JobQueue;
  readonly audit: AuditEventStore;
  readonly memory: WorkspaceMemoryStore;
  readonly feedback: SqliteFeedbackStore;
  readonly embedder: Embedder;
  readonly vectorStore: VectorStore | null;
}): ApplicationFacade => ({
  workspaceId: input.workspaceId,
  userId: input.userId,
  users: input.users,
  documents: input.documents,
  documentPrincipals: input.documentPrincipals,
  members: input.members,
  sessions: input.sessions,
  conversations: input.conversations,
  jobs: input.jobs,
  audit: input.audit,
  memory: input.memory,
  feedback: input.feedback,
  embedder: input.embedder,
  vectorStore: input.vectorStore,
  close: async () => undefined,
});