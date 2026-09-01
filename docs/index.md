# Revex — Documentation

Revex is hybrid retrieval for teams running on [Strands Agents](https://strandsagents.com).
It fuses dense vector search, BM25 keyword search, graph search, per-session memory,
and live web search into one engine behind a single orchestrator façade.

This documentation describes the **TypeScript monorepo** (v1.1.0). The earlier Python
`raghub` 0.9.x line and its ADR records have been removed; the TypeScript codebase is
canonical.

## What is inside

| Area | Doc |
|---|---|
| Getting started | [Getting started](guide/getting-started.md) |
| Architecture | [Architecture overview](architecture/overview.md) |
| Workspaces | [Workspace model](guide/workspace.md) + [Onboarding](guide/onboarding.md) |
| Access control | [RBAC & ACL](guide/rbac-acl.md) |
| Ingestion | [Ingestion pipeline](guide/ingestion.md) |
| Retrieval | [Retrieval & reranking](guide/retrieval.md) |
| Orchestration | [Orchestrator & agents](guide/orchestration.md) |
| Embeddings & LLM | [Embedders & LLM](guide/llm-embedding.md) |
| Jobs | [Workers & jobs](guide/workers-jobs.md) |
| HTTP API | [API reference](reference/api.md) |
| CLI | [CLI reference](reference/cli.md) |
| Evaluation | [Eval harness](guide/eval.md) |
| Telemetry | [Telemetry](guide/telemetry.md) |
| Web app | [Web console](guide/web.md) |
| Development | [Contributing & dev](guide/development.md) |

## Quick start

```bash
pnpm install
pnpm --filter @revex/api dev    # http://localhost:3000
pnpm --filter @revex/web dev    # http://localhost:3001
```

Then open `http://localhost:3001/onboarding` and walk the 5-step wizard.

## Layout

```
revex/
├── packages/
│   ├── core/          @revex/core          — domain, stores, retrieval, auth, telemetry, LLM, ingest
│   ├── api/           @revex/api           — Hono HTTP server
│   ├── orchestrator/  @revex/orchestrator  — Strands-shaped Orchestrator + agents + tools
│   └── eval/          @revex/eval          — retrieval metrics + benchmarks
├── apps/
│   ├── web/           Next.js 16 console
│   └── cli/           @revex/cli — `revex` binary
└── docs/              this documentation
```

## Technology

| Layer | Choice |
|---|---|
| Language | TypeScript 5.6+ (strict, ESM) |
| Workspaces | pnpm + Turbo |
| Vector store | [sqlite-vec](https://github.com/asg017/sqlite-vec) + FTS5 |
| Embeddings + LLM | [openai](https://github.com/openai/openai-node) + deterministic fallbacks |
| HTTP | [Hono](https://hono.dev) |
| Multi-agent | [Strands Agents](https://strandsagents.com) surface |
| UI | Next.js 16 + shadcn/ui |
| Auth | bcrypt + JWT (jose) + scrypt/AES-256-GCM |
| Test | Vitest + fast-check |