# Architecture overview

Revex is a pnpm + Turbo TypeScript monorepo. Four packages live under
`packages/` and two apps under `apps/`.

## Packages

### `@revex/core`

The engine. Dependencies-free of the framework's own packages; everything else
imports it.

- **Domain** — frozen value objects over an internal `props` bag: `Workspace`,
  `User`, `Document`, `Chunk`, `Turn`. Branded IDs throughout (`WorkspaceId`,
  `UserId`, `DocumentId`, `ChunkId`, `SessionId`, `TraceId`, `JobId`,
  `CollectionId`).
- **Errors** — `RevexError` base with stable string `code`s and concrete
  subclasses (`AuthError`, `AuthorizationError`, `ConfigurationError`,
  `GenerationError`, `IngestionError`, `MissingDepError`, `PipelineError`,
  `RetrievalError`, `VectorStoreError`, `VerificationError`).
- **Settings** — a Zod-validated tree, loaded from env via `loadSettings()`.
- **Embedders** — `OpenAIEmbedder`, `FeatureHashingEmbedder`, `createEmbedder`.
- **Retrieval** — the `Retrieval` pipeline (dense + sparse + RRF + RBAC),
  fusion helpers, rerankers, and retrieval transformers.
- **Stores** — the `Sqlite*Store` surface (users, documents, jobs, sessions,
  conversations, workspace members, groups, document principals, memory, audit,
  feedback, local file storage, images, document versions, snapshots).
- **Vector store** — `SqliteVecStore` (sqlite-vec + FTS5).
- **LLM** — `OpenAILlm`, `FeatureHashingLlm`, `StubLlm`, `LlmManager`,
  `createLlm`.
- **Auth primitives** — `BcryptHasher`, `JwtService`.
- **Encryption** — `openEncryptedWorkspace`, `EncryptedField`,
  `WorkspaceSettingsStore`.
- **Storage** — users, documents, sessions, conversations, file storage.
- **Workspaces** — `openWorkspace`, `WorkspaceRegistry`,
  `openFileWorkspaceRegistry`, tenant context helpers.
- **Chunker** — `chunkText`, `chunkMarkdown`, `chunkStructured`, `chunkPdf`.
- **Ingest** — `ingest`, `ingestVerbose`, `agenticIngest`.
- **Context** — `buildContext`, `defaultBudget`.
- **Jobs** — `JobQueue`, `SqliteJobQueue`, `MemoryQueue`.
- **Lifecycle** — `assertTransition`, `DocumentState`.
- **Graph** — `extractEntities`, `summariseCommunity`, `clusterEntities`,
  `SqliteGraphStore`.
- **Summary** — `SummaryIndex`, `createLlmSummaryIndex`,
  `createExtractiveSummaryIndex`, `buildRaptorTree`.
- **Feedback** — `Feedback`, scorers (`Bm25BoostScorer`,
  `VectorDownWeightScorer`).
- **Traces** — `TraceCorpus`, `SqliteTraceCorpus`.
- **Telemetry** — `NoOpTelemetry`, `LangfuseTelemetry`, `OtelTelemetry`,
  `createTelemetry`.
- **Web search** — `WebSearch`, `DuckDuckGoSearch`.
- **Migrations** — `runMigrations`, `MIGRATIONS`.

### `@revex/api`

The Hono HTTP server. Exposes `boot()` and `start()`.

- **app** — `createApp` wires middleware + routes; `errorMiddleware`,
  `jwtAuthMiddleware`, `rateLimitMiddleware`, `securityHeadersMiddleware`.
- **routes/** — one file per resource (`auth`, `documents`, `query`, `me`,
  `workspaces`, `feedback`, `audit`, `memory`, `tenants`, `webhooks`, ...).
- **workspace-context** — resolves JWT + passphrase cookie into fresh
  `Sqlite*Store` instances per request.
- **workspace-pool** — caches unlocked workspace database handles.
- **job-worker** — drains `document.ingest` jobs; `WorkspaceWorkerSupervisor`
  manages one worker per workspace.
- **services** — `ApplicationFacade`, `probeHealth`, `ShutdownCoordinator`.

### `@revex/orchestrator`

The Strands-shaped execution layer.

- **Orchestrator** — `run()`, `stream()`, `resolveInvocationState()`.
- **Patterns** — `buildGraph`, `buildSwarm`, `buildWorkflow`, `buildDeepResearch`.
- **Agents** — `AgentRegistry`, `RagAgent`, sub-agent builders, `createReActAgent`.
- **Hooks** — `HookRegistry`, `AgentHookBus`.
- **Tools** — `ToolRegistry` + 8 built-in tool creators.
- **Pipeline** — `QueryCache`, `PipelineRouter`, `shapeContext`.
- **Adapters** — `InProcessAdapter` (Phase 1) with `StrandsAdapter` boundary.

### `@revex/eval`

Retrieval benchmarking.

- Metrics: `recallAtK`, `precisionAtK`, `mrr`, contextual + faithfulness +
  correctness aggregates.
- `judgeCare` (CARE), `lostInMiddleProbe`, `judge`, `runSamples`,
  `generateSynthetic`, `evaluateGate`, harnesses for Finance and Frames.

## Apps

### `apps/web`

Next.js 16 App Router console. Marketing pages (`/`, `/sign-in`, `/privacy`,
`/terms`) and the authenticated app shell (`/chat`, `/documents`, `/members`,
`/memory`, `/onboarding`, `/settings`). A catch-all `/api/proxy` route forwards
requests to the Hono API, reusing the `revex_session` / `revex_workspace_key`
cookies.

### `apps/cli`

Commander-based `revex` binary registering 11 subcommands (see
[CLI reference](../reference/cli.md)).

## Data flow

```
upload ─► POST /v1/documents ─► LocalFileStorage ─► JobQueue
                                              └─► ingest() ─► chunk ─► embed ─► SqliteVecStore

query  ─► POST /v1/query ─► Orchestrator ─► agents/sub-agents ─► Retrieval (dense+BM25+RRF)
              ▲                                ▲
              └────────── strategy ────────────┘
```

## Tenancy & encryption model

Each workspace is one SQLite file. Encryption is per-workspace: the passphrase
derives a 32-byte AES-256-GCM key via scrypt (N=2¹⁵, r=8, p=1), and
`workspace_settings` rows are encrypted at rest. See
[Workspace model](workspace.md).

## Error handling

All errors flow through the `RevexError` hierarchy. The API `errorMiddleware`
maps `error.code` to an HTTP status:

| Code | Status |
|---|---|
| `auth_error` | 401 |
| `authorization_error` | 403 |
| `configuration_error` | 500 (503 for `StoreUnavailableError`) |
| `generation_error` | 502 |
| `ingestion_error` | 400 |
| `missing_dependency` | 500 |
| `pipeline_error` | 500 |
| `retrieval_error` | 502 |
| `vector_store_error` | 500 |
| `verification_error` | 400 |
| `revex_error` (default) | 500 |