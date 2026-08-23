/**
 * Server boot — wires every dependency the orchestrator + API
 * need and exposes a single `startServer({ settings })` entry
 * point used by `raghub dev` and the (future) embedded runtime.
 *
 * SQLite-vec is the only Phase 1 vector backend; pgvector is
 * available via the migration command but not at runtime.
 */

import {
  createEmbedder,
  createLlm,
  createTelemetry,
  Retrieval,
  SqliteVecStore,
  SqliteUserStore,
  SqliteDocumentStore,
SqliteJobQueue,
  SqliteSessionStore,
  SqliteConversationStore,
  BcryptHasher,
  JwtService,
  type Settings,
  type VectorStore,
  type Embedder,
  type Llm,
  type Telemetry,
  type UserStore,
  type DocumentStore,
  type SessionStore,
  type ConversationStore,
  type BcryptHasher as BcryptHasherType,
  type WorkspaceId,
  type UserId,
  brandId,
} from '@raghub/core';

import { AgentRegistry, Orchestrator, ToolRegistry, registerBuiltInTools } from '@raghub/orchestrator';
import { createApp } from '@raghub/api';

export interface ServerHandle {
  readonly app: ReturnType<typeof createApp>;
  readonly close: () => Promise<void>;
  readonly settings: Settings;
}

export interface StartServerOptions {
  readonly settings: Settings;
}

const deriveTenantId = (secret: string): WorkspaceId =>
  brandId<WorkspaceId>(`tnt_${Buffer.from(secret).toString('hex').slice(0, 12)}`);

const deriveOwnerId = (secret: string): UserId =>
  brandId<UserId>(`usr_${Buffer.from(secret).toString('hex').slice(12, 24)}`);

const swapDbPath = (vectorPath: string, suffix: string): string =>
  vectorPath.replace(/\/[^/]+$/, `/${suffix}`);

export const startServer = async (opts: StartServerOptions): Promise<ServerHandle> => {
  const settings = opts.settings;
  const telemetry: Telemetry = await createTelemetry(settings);
  const embedder: Embedder = createEmbedder(settings);
  const llm: Llm = createLlm(settings);
  const vectorStore: VectorStore = new SqliteVecStore({
    path: settings.vectorStore.path,
    embeddingDim: settings.vectorStore.embeddingDim,
  });
  const userStore: UserStore = new SqliteUserStore({
    path: swapDbPath(settings.vectorStore.path, 'users.db'),
  });
  const documentStore: DocumentStore = new SqliteDocumentStore({
    path: swapDbPath(settings.vectorStore.path, 'documents.db'),
  });
  const sessionStore: SessionStore = new SqliteSessionStore({
    path: swapDbPath(settings.vectorStore.path, 'sessions.db'),
  });
  const conversationStore: ConversationStore = new SqliteConversationStore({
    path: swapDbPath(settings.vectorStore.path, 'conversations.db'),
  });
  const jobQueue: SqliteJobQueue = new SqliteJobQueue({
    path: swapDbPath(settings.vectorStore.path, 'jobs.db'),
  });
  const hasher: BcryptHasherType = new BcryptHasher(settings.auth.bcryptRounds);
  const jwt = new JwtService({
    secret: settings.auth.jwtSecret,
    algorithm: settings.auth.jwtAlgorithm,
    ttlSeconds: settings.auth.tokenTtlSeconds,
  });

  const retrieval = new Retrieval(embedder, vectorStore, {
    topK: settings.orchestrator.topK,
    denseWeight: settings.hybrid.denseWeight,
    sparseWeight: settings.hybrid.sparseWeight,
    rrfK: settings.hybrid.rrfK,
  });

  const agents = new AgentRegistry();
  const tools = new ToolRegistry();
  registerBuiltInTools(tools, { retrieval, embedder, store: vectorStore });

  const orchestrator = new Orchestrator({
    telemetry,
    workspaceId: deriveTenantId(settings.auth.jwtSecret),
    sessionOverrides: {},
    agents,
    tools,
    llm,
    retrieval,
    model: settings.llm.model,
  });
  
  const app = createApp({
    userStore,
    documentStore,
    sessionStore,
    conversationStore,
    jobQueue,
    embedder,
    vectorStore,
    hasher,
    jwt,
    orchestrator,
  });

  return {
    app,
    settings,
    async close(): Promise<void> {
      await vectorStore.close();
      await userStore.close();
      await documentStore.close();
      await sessionStore.close();
      await conversationStore.close();
      await jobQueue.close();
    },
  };
};