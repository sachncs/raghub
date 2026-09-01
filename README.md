<p align="center">
  <h1 align="center">Revex</h1>
  <p align="center">Hybrid retrieval for teams — vector, keyword, graph, memory, web, one engine.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/node-%E2%89%A522-green" alt="Node"></a>
    <a href="https://github.com/sachncs/revex/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/revex/ci.yml?branch=main" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
</p>

---

## What is this?

Revex is a retrieval engine for teams running on [Strands Agents](https://strandsagents.com). It answers one question:

> _"What should this agent retrieve to answer the user well?"_

The answer is a **hybrid retrieval** pipeline — dense vector search, BM25 keyword search, graph search, per-session memory, and live web search all fused into one result. One orchestrator façade, three agent patterns (Graph / Swarm / Workflow), and one configurable per-user strategy. Local-first; no cloud required.

> **The problem.** RAG systems pick one retrieval strategy and call it a day. Dense embeddings miss exact keywords, BM25 misses meaning, graphs miss conversation history, and nothing adapts to the person asking.
>
> **The answer.** Revex runs multiple retrievers in parallel and fuses the results with Reciprocal Rank Fusion, scoped by per-user workspace RBAC. It ships as a TypeScript monorepo — domain, stores, retrieval, auth, and telemetry in `@revex/core`; a Strands-shaped orchestrator and agent runtime in `@revex/orchestrator`; a Hono HTTP layer in `@revex/api`; and a CLI in `@revex/cli`.

---

## Who is this for?

You, even if:

- You're new to TypeScript or Node.js.
- You've never built a RAG pipeline.
- You've never heard of Reciprocal Rank Fusion.

If you can install Node and type commands into a terminal, you can
use Revex. When the docs use a word you don't know, search for it
in the [API Reference](#api-reference).

If you've used Node.js before, you'll be productive in five minutes.

---

## What can it do?

- **Hybrid retrieval** — dense (`sqlite-vec`) + BM25 (FTS5) fused with RRF (k=60).
- **Per-user RBAC** — document, group, and role ACLs enforced at the SQLite layer.
- **Encrypted workspaces** — scrypt-derived AES-256-GCM keys, passphrase-gated.
- **Agent runtime** — one `Orchestrator` with Graph / Swarm / Workflow builders and an in-process adapter.
- **Multi-agent retrieval** — a `RagAgent` root with a registry of sub-agents (vector / keyword / graph / trace / memory / web / summary).
- **Built-in tools** — `hybrid_search`, `vector_search`, `keyword_search`, `today`, `web_search`, `trace_search`, `summary_search`, `graph_search`.
- **Agentic ingestion** — content-addressed, idempotent `ingest()` plus parallel graph + memory + summary side-effects.
- **LLM + embeddings with no-network fallbacks** — OpenAI plus deterministic `FeatureHashingEmbedder` / `FeatureHashingLlm`.
- **Auth & onboarding** — bcrypt + JWT (HS256) + a 5-step onboarding wizard.
- **Telemetry** — pluggable `NoOpTelemetry`, `LangfuseTelemetry`, `OtelTelemetry`.
- **HTTP API & web app** — a Hono server with SSE streaming and a Next.js 15 + shadcn/ui console.

---

## Before you start

You'll need **Node.js 22 or newer** and **pnpm** installed on your computer.

If you don't know what Node is or whether you have it:

1. Open a terminal (on macOS: `Cmd + Space`, type "Terminal"; on
   Windows: open "PowerShell"; on Linux: open your usual terminal).
2. Type `node --version` and press Enter.
3. If you see a version number starting with `v22`, you're set.
4. If you see "command not found" or an older version, follow the
   [official Node installer guide](https://nodejs.org/en/download).
5. Revex uses pnpm workspaces — install it with
   `corepack enable pnpm` or `npm i -g pnpm`.

---

## Installation

```bash
# 1. Download the code
git clone https://github.com/sachncs/revex.git
cd revex

# 2. Install dependencies (pnpm workspace)
pnpm install

# 3. Run the test suite to confirm everything works
pnpm test
```

---

## Your first run

Start the API and web console together:

```bash
# Terminal 1 — HTTP API
pnpm --filter @revex/api dev        # http://localhost:3000

# Terminal 2 — Web console
pnpm --filter @revex/web dev        # http://localhost:3001
```

Then open `http://localhost:3001/onboarding` in a browser and walk the
**5-step wizard**: workspace name → admin email/password → LLM provider →
workspace passphrase → confirm. The browser never sees your LLM API key
in plaintext — it's encrypted with the workspace passphrase and stored
in `workspace.db`.

For a CLI-first workflow:

```bash
pnpm --filter @revex/cli dev -- --help
```

---

## Configuration

Workspaces are configurable through the web console, the CLI, or the
`@revex/core` API. The most important knob is the per-user retrieval
strategy — which of the retrievers (vector, keyword, graph, memory,
web) are fused for a given workspace member.

| Concern       | Field                                             |
| ------------- | ------------------------------------------------- |
| Storage       | One SQLite file per workspace (`workspace.db`)    |
| Encryption    | scrypt (N=2¹⁵, r=8, p=1) → AES-256-GCM            |
| Retrieval     | dense + BM25 + RRF (k=60)                          |
| Embeddings    | `OpenAIEmbedder`, `FeatureHashingEmbedder`        |
| LLM           | `OpenAILlm`, `FeatureHashingLlm`                  |
| LLM provider  | `createLlm({ provider: 'minimax' })` → `https://api.minimax.chat/v1` |
| Auth          | bcrypt + JWT (HS256) + workspace passphrase       |

---

## API Reference

| Symbol                                                        | What it does                                        |
| ------------------------------------------------------------- | --------------------------------------------------- |
| `Workspace.open(path)`                                        | Open a workspace database handle.                   |
| `openEncryptedWorkspace({ path, passphrase })`                | Open an encrypted workspace with a passphrase.      |
| `SqliteVecStore`                                              | Dense vector + FTS5 keyword store (sqlite-vec).     |
| `SqliteGraphStore` / `SqliteDocumentStore`                    | Graph + document stores.                            |
| `SqliteWorkspaceMemoryStore` / `SqliteConversationStore`      | Session memory + conversation history.              |
| `SqliteJobQueue`                                              | Persistent job queue.                               |
| `hybridSearch` / RRF                                         | Dense + BM25 fusion.                                |
| `OpenAIEmbedder` / `FeatureHashingEmbedder`                   | Embedding providers (network + offline).            |
| `OpenAILlm` / `FeatureHashingLlm` / `createLlm`               | LLM providers + factory.                            |
| `Orchestrator#buildGraph` / `#buildSwarm` / `#buildWorkflow`  | Three Strands-shaped pattern builders.              |
| `InProcessAdapter`                                            | Run the orchestrator in-process.                    |
| `RagAgent` / `SubAgent` / `HookRegistry`                      | Multi-agent retrieval runtime + hooks.              |
| `ingest()` / `agenticIngest()`                                | Content-addressed, idempotent ingestion.            |
| `BcryptHasher` / `JwtService`                                 | Auth primitives.                                    |
| `LangfuseTelemetry` / `OtelTelemetry` / `NoOpTelemetry`        | Telemetry backends.                                 |
| `/v1/auth` / `/v1/query` / `/v1/documents` / `/v1/workspaces` | Hono HTTP routes.                                   |

---

## Workspace model

Each workspace is one SQLite file (`workspace.db`) plus an optional
`snapshots/` directory. Open it via
`openEncryptedWorkspace({ path, passphrase })`; the passphrase derives
a 32-byte AES-256-GCM key via scrypt. All `workspace_settings` rows are
encrypted at rest. Without a passphrase the workspace runs in plaintext
mode for dev.

RBAC is enforced at the storage layer: `workspace_member` rows grant
owner / admin / member / viewer, and `document_principal` rows scope
read / admin per document. Retrieval filters results by
`allowedCompanyFilter(user)`.

---

## Where to go next

- **[API Reference](#api-reference)** — Full list of symbols and
  methods. Bookmark this once you start writing real code.
- **[Workspace Model](#workspace-model)** — Encryption and RBAC, the
  data layer behind everything.
- **[Project Structure](#project-structure)** — How the monorepo is
  laid out, for the curious.
- **[Tech Stack](#tech-stack)** — Build tools, test runners, and
  framework choices.
- **[Documentation](#documentation)** — The docs site.
- **[Roadmap](#roadmap)** — Where the project is heading.

For operators / maintainers:

- **[Development](#development)** — Run lint, tests, and builds.
- **[License](#license)** — MIT.

---

## Project Structure

```
revex/
├── packages/
│   ├── core/           # @revex/core — domain, stores, retrieval, auth, telemetry
│   ├── orchestrator/   # @revex/orchestrator — Strands-shaped Orchestrator + tools
│   ├── api/            # @revex/api — Hono HTTP server
│   └── eval/           # @revex/eval — Finance + Frames + CARE
├── apps/
│   ├── web/            # Next.js 15 + shadcn/ui console (onboarding, chat, ACL)
│   └── cli/            # @revex/cli — `revex` binary
├── docs/               # Guides, references, architecture
├── tsconfig.base.json  # strict, exactOptionalPropertyTypes
├── turbo.json
└── package.json        # pnpm workspace root
```

---

## Development

```bash
git clone https://github.com/sachncs/revex.git
cd revex
pnpm install
pnpm typecheck      # TS strict typecheck across all packages
pnpm lint           # ESLint + Prettier (oxc fast path, eslint slow path)
pnpm test           # Vitest unit + integration suites
pnpm build          # Turbo build across packages
pnpm --filter @revex/web dev    # Next.js console
pnpm --filter @revex/api dev    # Hono API
pnpm --filter @revex/cli dev    # CLI
```

---

## Documentation

Guides, architecture notes, and reference live in [`docs/`](./docs):

| Doc                            | Covers                                        |
| ------------------------------ | --------------------------------------------- |
| [`docs/quickstart.md`](./docs/quickstart.md)       | First-run setup and onboarding wizard.        |
| [`docs/workspace.md`](./docs/workspace.md)         | Encryption, storage, snapshots.               |
| [`docs/acl.md`](./docs/acl.md)                 | Document ACLs, groups, roles.                 |
| [`docs/agents.md`](./docs/agents.md)             | Orchestrator patterns and agent runtime.      |
| [`docs/onboarding.md`](./docs/onboarding.md)       | The 5-step onboarding flow.                   |
| [`docs/operations.md`](./docs/operations.md)       | Deployment and operations.                    |

---

## Tech Stack

| Layer       | Choice                                          |
| ----------- | ----------------------------------------------- |
| Language    | TypeScript 5.6+ (strict, ESM-only)              |
| Workspaces  | pnpm + Turbo                                    |
| Multi-agent | [Strands Agents SDK](https://strandsagents.com) |
| Vector store| [sqlite-vec](https://github.com/asg017/sqlite-vec) |
| Embeddings + LLM | [openai](https://github.com/openai/openai-node) |
| HTTP        | [Hono](https://hono.dev)                        |
| UI          | Next.js 15 + shadcn/ui                          |
| Auth        | bcrypt + JWT (HS256) + scrypt/AES-256-GCM       |
| Test        | Vitest                                          |
| Lint        | ESLint + Prettier                               |

---

## Roadmap

- **Current** — Hybrid retrieval, workspace RBAC, encrypted workspaces,
  Strands orchestrator with Graph / Swarm / Workflow, Hono API, web console.
- **Next** — `@revex/eval` (Finance + Frames + CARE evaluation), more
  vector adapters (PGVector first), persistent queue hardening.
- **Planned** — Feedback loop on retrieval quality, multi-tenant
  isolation improvements, backup format for snapshots.

---

## Contributing

Want to improve Revex? See the [coding standards](./AGENTS.md) and
[`CONTRIBUTING.md`](./CONTRIBUTING.md) for how to set up a development
environment and submit changes.

## Code of Conduct

We expect everyone to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Security

Found a security issue? See [`SECURITY.md`](./SECURITY.md) — please don't
open a public GitHub issue for security problems.

## License

[MIT](LICENSE) © 2026 Sachin.
