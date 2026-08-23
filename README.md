<p align="center">
  <h1 align="center">raghub</h1>
  <p align="center">Multi-user, user-controlled RAG on Strands Agents.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/node-%E2%89%A522-green" alt="Node"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
</p>

Multi-user RAG on [Strands Agents](https://strandsagents.com). One
orchestrator façade, three patterns (Graph / Swarm / Workflow), one
configurable per-user strategy. Local-first; no cloud required.

| Concern | Library |
|---|---|
| Multi-agent | [Strands Agents SDK](https://strandsagents.com) (TypeScript) |
| Vector store | [sqlite-vec](https://github.com/asg017/sqlite-vec) |
| Embeddings + LLM | [openai](https://github.com/openai/openai-node) |
| HTTP | [Hono](https://hono.dev) |
| UI | Next.js 15 + shadcn/ui |
| Auth | bcrypt + JWT (HS256) + workspace passphrase (scrypt + AES-256-GCM) |
| Tests | Vitest |

## Layout

```text
packages/
├── core/           @raghub/core — domain, stores, retrieval, auth, telemetry
├── orchestrator/   @raghub/orchestrator — Strands-shaped Orchestrator + tools + RagAgent
├── api/            @raghub/api — Hono HTTP server
└── eval/           @raghub/eval — Finance + Frames + CARE (Phase 2)

apps/
└── web/            Next.js 15 + shadcn/ui (onboarding wizard + chat + docs + ACL)

archive/            final raghub 0.9.x Python release (deprecated, read-only)
```

## Quick start

```bash
# 1. install deps
pnpm install

# 2. start the API + web together.
pnpm --filter @raghub/api dev    # http://localhost:3000
pnpm --filter @raghub/web dev    # http://localhost:3001

# 3. open http://localhost:3001/onboarding in a browser and
#    walk the 5-step wizard (workspace name, admin email/password,
#    LLM provider, workspace passphrase, confirm). The browser
#    never sees the LLM API key in plaintext — it's encrypted with
#    the workspace passphrase and stored in workspace.db.
```

## Workspace model

Each workspace is one SQLite file (`workspace.db`) plus an optional
`snapshots/` directory. Open it via `openEncryptedWorkspace({ path,
passphrase })`; the passphrase derives a 32-byte AES-256-GCM key via
scrypt (N=2¹⁵, r=8, p=1). All `workspace_settings` rows are
encrypted at rest. Without a passphrase the workspace runs in
plaintext mode for dev.

See `docs/workspace.md`, `docs/acl.md`, `docs/agents.md`,
`docs/onboarding.md`.

## What's in the current commit chain

- **Domain** — `Workspace`, `User`, `Document`, `Chunk`, `Turn`, branded
  ids, frozen classes (TS counterpart of `@dataclass(slots=True,
  frozen=True)`).
- **Workspace RBAC** — `workspace_member` rows: owner / admin /
  member / viewer. `canManageWorkspace` / `canIngest` predicates.
- **Groups + roles** — `workspace_group`, `workspace_group_member`,
  `role`, `role_member` tables; named roles can reference users or
  groups.
- **Document ACL** — `document_principal` (user / role / group) x
  (read / admin) per document; enforced at the SQLite layer via a
  parameterized subquery in `SqliteVecStore.buildAclClause`.
- **Stores** — `SqliteVecStore` (sqlite-vec + FTS5), `SqliteGraphStore`,
  `SqliteTraceCorpus`, `SqliteWorkspaceMemoryStore`, `SqliteSessionStore`,
  `SqliteConversationStore`, `SqliteDocumentStore`,
  `SqliteDocumentPrincipalStore`, `SqliteWorkspaceMemberStore`,
  `SqliteRoleStore`, `SqliteGroupStore`, `SqliteUserStore`,
  `SqliteJobQueue`. All take the shared `Database` handle from
  `Workspace.open(path)`.
- **Retrieval** — dense + BM25 + RRF (k=60). `allowedCompanyFilter(user)`
  for per-user RBAC.
- **Embedders** — `OpenAIEmbedder` (lazy-loaded SDK),
  `FeatureHashingEmbedder` (deterministic, no-network fallback).
- **LLM** — `OpenAILlm` (Bearer or Raw `Authorization` header for
  MiniMax-style providers), `FeatureHashingLlm` (no-network fallback).
  `createLlm({ provider: 'minimax' })` defaults to
  `https://api.minimax.chat/v1` (https://platform.minimax.io/docs/guides/models-intro).
- **Encryption** — `openEncryptedWorkspace` / `WorkspaceSettingsStore`
  / `EncryptedField`. Verifier ciphertext pattern.
- **LocalFileStorage** — `FsLocalFileStorage` and `InMemoryLocalFileStorage`
  for session snapshots and any other large blob that doesn't belong
  in `workspace.db`. Atomic write (write tmp + rename) and path
  traversal protection.
- **Auth** — `BcryptHasher`, `JwtService` (jose, HS256), 5-step
  onboarding wizard at `apps/web/src/app/onboarding/page.tsx`.
- **Telemetry** — `NoOpTelemetry` (default), `LangfuseTelemetry`,
  `OtelTelemetry` (lazy-loaded).
- **Orchestrator** — single `Orchestrator` class with three pattern
  builders (`buildGraph` / `buildSwarm` / `buildWorkflow`) and an
  `InProcessAdapter`. `invocation_state` is the Strands-shaped record
  propagated to every node and tool.
- **Agent runtime** — `RagAgent` (multi-agent root), `SubAgent`
  registry (vector / keyword / graph / trace / memory / web /
  summary), `HookRegistry` (6 hook points), `RetryStrategy`,
  per-session `SessionState` bag, automatic conversation
  summarization via a registered summarizer agent.
- **Tools** — 8 built-in tools: `hybrid_search`, `vector_search`,
  `keyword_search`, `today`, `web_search`, `trace_search`,
  `summary_search`, `graph_search`.
- **Ingestion** — `ingest()` (content-addressed SHA-256, idempotent)
  + `agenticIngest()` (parallel graph + memory + summary side-effects).
- **API** — Hono server with `/v1/auth/{register,login}`,
  `/v1/me`, `/v1/query`, `/v1/query/stream` (SSE),
  `/v1/documents`, `/v1/documents/:id/principals`,
  `/v1/workspaces/members`.
- **Web** — Next.js 15 App Router with shadcn primitives.
  `/sign-in`, `/onboarding` (5-step wizard), `/chat` (SSE consumer +
  sub-agent trace panel), `/documents` (upload + ACL share modal),
  `/settings`, `/members` (RBAC).
- **Archive** — the legacy Python source tree, frozen at 0.9.x.

## What's next

See `archive/README.md` for the historical Python API and the
migration path. The active roadmap lives in the issue tracker; the
locked plan is in the design docs (Phase 1 commits through Phase 4
local hardening).

## License

MIT. See [LICENSE](./LICENSE).