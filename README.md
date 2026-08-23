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
| Auth | bcrypt + JWT (HS256) |
| Tests | Vitest |

## Layout

```text
packages/
├── core/           @raghub/core — domain, stores, retrieval, auth, telemetry
├── orchestrator/   @raghub/orchestrator — Strands-shaped Orchestrator + tools
├── api/            @raghub/api — Hono HTTP server
├── cli/            @raghub/cli — `raghub` binary
└── eval/           @raghub/eval — Finance + Frames + CARE (Phase 2)

apps/
└── web/            Next.js 15 + shadcn/ui

archive/            final raghub 0.9.x Python release (deprecated, read-only)
```

## Quick start

```bash
# 1. install deps
pnpm install

# 2. init a new project (writes .raghub/.env)
npx @raghub/cli init

# 3. edit .raghub/.env — set OPENAI_API_KEY, JWT_SECRET, secrets
$EDITOR .raghub/.env

# 4. start the API + web together (Phase 1 follow-up wires
#    a `pnpm dev` at the root that boots both).
pnpm --filter @raghub/api dev    # http://localhost:3000
pnpm --filter @raghub/web dev    # http://localhost:3001
```

## What's in Phase 1 (this commit chain)

- **Domain** — `Tenant`, `User`, `Document`, `Chunk`, `Turn`, branded
  ids, frozen classes (TS counterpart of `@dataclass(slots=True,
  frozen=True)`).
- **Stores** — `SqliteVecStore` (sqlite-vec + FTS5). One backend; no
  pgvector / memory / dedicated-vector.
- **Retrieval** — dense + BM25 + RRF (k=60).
- **Embedders** — `OpenAIEmbedder` (lazy-loaded SDK), `FeatureHashingEmbedder`
  (deterministic, no-network fallback).
- **Auth** — `BcryptHasher`, `JwtService` (jose, HS256).
- **Tenants** — `AsyncLocalStorage`-backed context. `RowLevel`
  isolation only.
- **Telemetry** — `NoOpTelemetry` (default), `LangfuseTelemetry`,
  `OtelTelemetry` (lazy-loaded).
- **Orchestrator** — single `Orchestrator` class with three pattern
  builders (`buildGraph` / `buildSwarm` / `buildWorkflow`) and an
  `InProcessAdapter` for Phase 1. `invocation_state` is the
  Strands-shaped record propagated to every node and tool.
- **Tools** — 8 built-in tools: `hybrid_search`, `vector_search`,
  `keyword_search`, `today`, `web_search`, `trace_search`,
  `summary_search`, `graph_search`.
- **API** — Hono server with `/v1/auth/{register,login}`, `/v1/me`,
  `/v1/me/strategy`, `/v1/query`, `/v1/query/stream` (SSE).
- **CLI** — `raghub init`, `raghub migrate-import`, `raghub dev`.
- **Web** — Next.js 15 App Router with shadcn primitives. `/sign-in`,
  `/onboarding`, `/chat` (SSE consumer), `/documents`, `/settings`.
- **Archive** — the legacy Python source tree, frozen at 0.9.x, with
  a top-level deprecation README and a `__deprecated__.py` shim.

## What's next

See `archive/README.md` for the historical Python API and the
migration path. The active roadmap lives in the issue tracker; the
locked plan is in the design docs (Phase 1 commits through Phase 4
local hardening).

## License

MIT. See [LICENSE](./LICENSE).