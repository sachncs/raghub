<p align="center">
  <h1 align="center">RAGHub</h1>
  <p align="center">Production-grade multi-user retrieval-augmented generation platform.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/raghub/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/raghub/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/raghub/"><img src="https://img.shields.io/pypi/v/raghub" alt="PyPI"></a>
  </p>
</p>

**Production-grade multi-user RAG platform.**

RAGHub is a layered retrieval-augmented generation stack with a single
replace-everything facade (`raghub.RAG`), multi-tenant RBAC, conversational
memory, persistent ingestion, real streaming, and the Finance and
FRAMES evaluators. Every collaborator (converter, chunker, vector
store, embedder, retriever, generator, telemetry, evaluator) is
replaceable through the registry; the default wiring installs the
spec-mandated libraries (Chonkie, LiteLLM, Langfuse, rank-bm25) and
falls back to deterministic in-process providers when no API keys are
present, so `pip install` and `import` is enough to be productive.

The default vector store is in-process SQLite (`raghub.stores`).
Production deployments that need to scale past a single writer
can switch to the PostgreSQL + pgvector backend
(`PgVectorStore` in `raghub.stores.pgvector`, initialised by
`raghub migrate pgvector --dsn <dsn>`) and run the API behind a load
balancer.

Every persisted entity is a frozen `@dataclass(slots=True)` that
inherits a shared `Snap` mixin providing `dump` / `validate` /
`copy` / `verify`. Settings is a frozen dataclass tree; the
`Settings.load()` reader loads env + profile and produces a
fully-typed instance. Production defaults: fail-closed CORS
(wildcard + credentials is rejected), non-zero `JWT_SECRET`
required, opaque session tokens, and a single canonical ingestion
pipeline that calls the configured chunker and persists the bundle
only after the vector store has indexed every chunk.

| Concern | Library |
|---|---|
| Chunking | [Chonkie](https://github.com/chonkie-inc/chonkie) |
| LLM + Embeddings | [LiteLLM](https://github.com/BerriAI/litellm) |
| Document Conversion | [Marker](https://github.com/datalab-to/marker) (optional `[pdf]`) |
| Structured Outputs | [Instructor](https://github.com/567-labs/instructor) (optional `[structured]`) |
| Vector Store | pgvector (recommended for production) / SQLite in-process (default) |
| Observability | Langfuse (core dep) / OpenTelemetry (optional `[otel]`) |
| Ranking | rank-bm25, optional Cohere / ColBERT / LLM-as-judge |
| Evaluation | Finance / FRAMES |

## Features

- **Single replace-everything facade** — `raghub.RAG` lazy-imports
  every collaborator and is the only construction site a caller
  needs. Sub-accessors (`RAG.queue()`, `RAG.feedback_store()`,
  `RAG.rate_limiter()`, `RAG.archive()`, `RAG.tenant_resolver()`,
  `RAG.isolation_strategy()`) cover the opt-in integrations.
- **Multi-tenant RBAC** — Query results are scoped to each user's
  `allowed_companies`; admins see everything, unauthorised users see
  nothing. The LLM is given only the authorised context.
- **Conversational history** — Session-scoped turn memory
  namespaced by both `session_id` and the caller. Two callers who
  share or guess a `session_id` cannot read each other's history.
- **Incremental indexing** — Content-addressed by SHA-256 hash;
  unchanged files are skipped on re-ingest.
- **Persistent ingestion queue** — `SqliteQueue` survives process
  restarts; saturation raises `QueueSaturatedError`.
- **Real streaming** — `rag.astream` yields tokens as they arrive;
  `POST /v1/query/stream` ships them as Server-Sent Events.
- **Plugin system** — `Plugins` registers converters, chunkers,
  embedders, vector stores, retrievers, rerankers, generators,
  telemetry providers, and evaluators. Discovery via the
  `raghub.plugins` entry-point group.
- **Telemetry** — Langfuse scores for latency / quality / token
  usage; OpenTelemetry spans for trace export. Silent no-op when
  not configured.
- **Evaluators** — `Finance` (Recall@K, Precision@K, MRR,
  Faithfulness, Context Recall / Precision, Answer Correctness)
  and `Frames` (multi-modal grounded QA).
- **Query transforms** — HyDE, multi-query, step-back,
  decomposition. Composable per-query or via
  `settings.query_transforms.enabled`.
- **Hybrid retrieval** — Dense + BM25 fused by Reciprocal Rank
  Fusion (k=60). Optional ColBERT late-interaction channel.
- **Rerankers** — Cohere, BGE, LLM-as-judge, cascade. Wired via
  `settings.reranker.provider`.
- **Long-context second pass** — Reorders the top-K with a
  long-context LLM. Optional, configured per request.
- **RAPTOR** — Recursive summary tree at ingest time; flat-tree
  search across every level.
- **GraphRAG** — Entity / community graph with LLM-driven triple
  extraction + summarisation.
- **Agentic retrieval** — ReAct planner with streaming
  `PlannerEvent` (thought / tool_call / tool_result /
  answer_chunk / final). Built-in tools cover Today, GraphSearch,
  HybridSearch, Keyword, SummarySearch, VectorSearch, WebSearch.
- **Per-user tool preferences** — Three layers of override
  (request > session > user > global); persisted via API and CLI.
- **Streaming endpoints** — `POST /v1/query/stream` (SSE),
  `POST /v1/agent/run`, and `RAG.astream_agent`.

See [`docs/ADVANCED_RAG.md`](docs/ADVANCED_RAG.md) for the full
reference.

## Installation

### From PyPI

```bash
pip install raghub
pip install "raghub[api]"               # FastAPI / uvicorn (server)
pip install "raghub[pdf]"               # marker-pdf + pypdf
```

### From source

```bash
git clone https://github.com/sachncs/raghub.git
cd raghub
pip install -e ".[dev,api]"
```

| Extra | Includes |
|---|---|
| `api` | FastAPI, uvicorn, python-multipart |
| `pdf` | marker-pdf, pypdf |
| `graph` | python-igraph, leidenalg, scikit-learn |
| `rerank` | cohere, ragatouille, rank-bm25 |
| `web` | duckduckgo-search |
| `docs` | beautifulsoup4, Pillow, openpyxl, python-docx, python-pptx |
| `auth` | bcrypt, aiosqlite |
| `otel` | opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi |
| `tiktoken` | tiktoken |
| `structured` | instructor |
| `eval` | datasets |
| `all` | every optional above |

Core deps (always installed): `numpy`, `PyYAML`, `chonkie`,
`litellm`, `langfuse`, `loguru`, `tqdm`, `typer`, `rank-bm25`.
Optional extras add the rest.

## Quick Start

### Python API

```python
from raghub import RAG

rag = RAG()
rag.ingest(b"Revenue grew 12% in Q3 2024.")
print(rag.query("revenue").answer)
```

Async / streaming:

```python
import asyncio
from raghub import RAG

async def main():
    rag = RAG()
    rag.ingest(b"Revenue grew 12% in Q3 2024.")
    async for chunk in rag.astream("revenue"):
        print(chunk, end="", flush=True)
    print()
    response = await rag.aquery("revenue")
    print(response.answer)

asyncio.run(main())
```

No API keys required — RAGHub uses a deterministic in-process
provider when none are configured. Set one of the LLM env vars
(`RAG_LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`NVIDIA_API_KEY`, `GROQ_API_KEY`, `LITELLM_API_KEY`) for real
completions.

### CLI

```bash
raghub init -o raghub.yaml
raghub ingest ./documents
raghub query "What was the revenue guidance?"
raghub backup create --output snapshot.tar.zst
raghub backup verify --input snapshot.tar.zst
raghub health
raghub version
raghub run --host 0.0.0.0 --port 8000   # start the FastAPI server
```

The CLI registers one Typer sub-app per command family:
`init`, `ingest`, `query`, `server`, `feedback`, `queue`,
`tenant`, `migrate`, `backup`, plus the `version` / `health`
utilities. A separate `raghub-financebench` console script
exposes the Finance / FRAMES eval harnesses.

The CLI emits status events through the loguru logger
(`cli.ingest`, `cli.version`, `cli.rate_limit_exceeded`, etc.);
the default sink is stderr at the configured log level.

### FastAPI

The canonical entry is the `App.create(Settings.load())` factory:

```python
from raghub.api import App
from raghub.config import Settings

app = App.create(Settings.load())
```

The same factory is the `raghub.api:create_app(application)`
helper, and `App.create` is what `uvicorn` should call
(factory-style):

```bash
uvicorn "raghub.api:create_app" \
    --factory \
    --host 0.0.0.0 --port 8000
```

The `RAG` facade is the recommended integration for new
code. Use `RAG.from_config("raghub.yaml")` to construct from a
profile, or `RAG(components=...)` to override individual
collaborators in code.

## Configuration

Settings live in `raghub.config` and are loaded by
`Settings.load()` from environment variables plus an optional
profile. Every config value is read at process start, and
missing required values (e.g. `JWT_SECRET` in production) raise
immediately.

Profile search order (first match wins):

1. `RAG_PROFILE` env var (explicit override)
2. `./config` relative to CWD
3. `~/.config/raghub` (XDG-style user config)
4. Bundled `config/` shipped with the package

Real credentials belong in `.env` (gitignored). See
`.env.example` for the template.

### Environment Variables

| Env Variable | Default | Description |
|---|---|---|
| `JWT_SECRET` | unset (required in production) | Opaque session-token signing secret (≥32 bytes) |
| `RAG_PROFILE` | unset | Profile name to load (e.g. `production`) |
| `RAG_DATA_DIR` | `./data` | Documents / sessions / registry base directory |
| `RAG_REGISTRY_PATH` | `./data/registry.json` | JSON plugin registry path |
| `RAG_SESSIONS_PATH` | `./data/sessions.json` | JSON sessions store path |
| `RAG_RERANKER_TOP_K` | unset | Optional override for the reranker top-K |
| `RAG_LLM_TIMEOUT_SECONDS` | unset | LiteLLM request timeout (seconds) |
| `RAG_TENANT_DSNS` | unset | Comma-separated `tenant=dsn,dim;...` mapping |
| `RAG_VECTORSTORE_PATH` | unset | On-disk location of the default SQLite vector store |
| `RAGHUB_USERS` | inline demo users | JSON path or inline JSON for the user directory |
| `RAGHUB_ARCHIVE_SIGNING_KEY` | unset | Required for `raghub backup` in production |
| `RAGHUB_TENANT_SECRETS_KEY` | unset | Required for per-tenant secret encryption |
| `CORS_ORIGINS` | unset | Comma-separated allow-list; wildcard+credentials is rejected at startup |
| `RAG_LLM_API_KEY` | unset | LLM provider key (highest-precedence fallback) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `NVIDIA_API_KEY` / `GROQ_API_KEY` / `LITELLM_API_KEY` | unset | Per-provider LLM keys (LiteLLM consults them in this order) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | unset | Langfuse credentials (core telemetry) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `COHERE_API_KEY` / `VOYAGE_API_KEY` / `AZURE_API_KEY` | unset | Per-provider keys consumed by the corresponding SDK |
| `RAG_ALLOW_PASSWORDLESS` | unset | Permit users without a password in non-production environments |
| Constructor kwargs | — | Passed to `RAG(...)` (highest precedence) |

Precedence (highest first): constructor arguments → env vars →
built-in defaults. `Settings.override(**changes)` returns a new
instance with the given fields changed (the original is not
mutated, since `Settings` is frozen).

## Multi-User & RBAC

Every public query method accepts a `User`. The retrieval layer is
filtered to the user's `allowed_companies`; admins see everything.
The LLM is given only the authorised context — there is no path by
which unauthorised content can leak into the prompt.

```python
from raghub.models import User

alice = User(id="alice", email="alice@x", allowed_companies=["Apple"])
bob   = User(id="bob",   email="bob@x",   allowed_companies=["Microsoft"])
admin = User(id="admin", email="admin@x", is_admin=True)

rag.query("revenue", user=alice)   # Apple-only chunks
rag.query("revenue", user=bob)     # Microsoft-only chunks
rag.query("revenue", user=admin)   # all chunks
```

A user with no `allowed_companies` and no `is_admin` sees no
documents (the filter resolves to `{"company": []}` which matches
nothing). Unauthorised retrieval attempts return an empty result
set. Session tokens are opaque; the JWT is verified at every
request boundary.

## Conversational RAG

Every public query method accepts a `session_id`. The pipeline
loads the most recent turns from the conversation store and
prepends them to the prompt so the LLM can answer follow-up
questions.

```python
await rag.aquery("revenue", user=alice, session_id="alice-s1")
# Bob's session is isolated; Alice's session has its own history.
await rag.aquery("and growth?", user=alice, session_id="alice-s1")
```

## API

| Symbol | Type | Description |
|---|---|---|
| `raghub.RAG` | class | Single replace-everything facade; lazy-imports every collaborator. Built via `RAG()`, `RAG.from_config(path)`, or `RAG(components=...)`. |
| `raghub.config.Settings` | class | Frozen dataclass; loaded by `Settings.load()`. |
| `raghub.models.User` | model | Per-user identity with `allowed_companies` and `is_admin`. |
| `raghub.services.ApplicationFacade` | class | Application service facade (login, upload, query, history, etc.). |
| `raghub.api.App.create` | classmethod | FastAPI app factory. |
| `raghub.plugins.Plugins` | class | Register converters, chunkers, vector stores, etc. via `Plugins.register(PluginKind.X, name, obj)`. |
| `raghub.eval.Finance` / `raghub.eval.Frames` | class | Evaluation harnesses. |
| `raghub.cli:main` | CLI | `raghub init / ingest / query / health / version / …` |
| `raghub-financebench` | CLI | `raghub-financebench --examples N` (via `raghub.evaluation:app`) |

The previous alias `Facade = ApplicationFacade` is deprecated and
emits a `DeprecationWarning` on first use; the canonical name is
`ApplicationFacade`. The alias is being removed in the next minor
release.

## Examples

Register a custom converter on the `Plugins` registry:

```python
from raghub.plugins import PluginKind, Plugins
from raghub.lifecycle import Marker

registry = Plugins()
registry.register(PluginKind.Converter, "marker", Marker())
rag = RAG(components={"registry": registry})
```

`Memory` (the in-process conversation store) is the default; for a
SQLite-backed session store, pass a `JsonSessions` to
`Settings.sessions` or wire one through `RAG(components=...)`.

## Project Structure

```
raghub/
├── raghub/                  # The library
│   ├── __init__.py          # Public surface (Settings, RAG, Snap-based models, errors, …)
│   ├── agent.py             # ReAct planner + tool protocol
│   ├── api.py               # FastAPI App factory
│   ├── archive/             # Snapshot / restore / verify
│   ├── auth/                # AuthService + SqliteUsers
│   ├── authhelpers/         # Inject / Auth / Bearer dependency helpers
│   ├── cli.py               # Console-script entry
│   ├── commands/            # Typer sub-apps (cli_config, feedback)
│   ├── config/              # Settings + env/profile loaders
│   ├── constants.py         # Named constants (HTTP codes, ENV_*, MAX_INFLIGHT_DEFAULT, …)
│   ├── conv.py              # ConversationStore / Memory / Tokenizer / SlidingWindow
│   ├── core.py              # Tenant / UserRecord helpers
│   ├── domain.py            # Snapshot / Contract types
│   ├── embedder.py          # LiteLLMEmbedder / FeatureHashingEmbedder
│   ├── errors.py            # Typed error hierarchy
│   ├── eval/                # Finance, Frames, Judge, Gate, Metrics, Scoring
│   ├── evaluation.py        # raghub-financebench entry
│   ├── feedback/            # SqliteFeedbackStore / Bm25BoostScorer / VectorDownWeightScorer
│   ├── gen.py               # DefaultGenerator
│   ├── ids.py               # Deterministic-id helpers
│   ├── ingest/              # Chunker, Ingestor, Jobs
│   ├── io.py                # capture() helper
│   ├── jobs/                # PersistentQueue, SqliteQueue, Worker
│   ├── knowledge/           # Manifest, MemoryRepo, Graph, Raptor, OKF helpers
│   ├── lifecycle/           # chunking, converters, scanner, state machine
│   ├── llm.py               # LiteLLM (and the deterministic provider)
│   ├── migrate.py           # Manifest v2 upgrade CLI
│   ├── models/              # Snap-based domain entities (Chunk, Document, Hit, …)
│   ├── parsers.py           # MarkerConverter / PlainTextConverter
│   ├── pipeline/            # Flow builder, agent, cache, ingest, query, router, span
│   ├── plugins.py           # Plugins (register / has / kinds / names)
│   ├── prompts.py           # Prompt templates
│   ├── rag/                 # RAG class + focused mixins
│   ├── ratelimit/           # Token bucket + ASGI middleware
│   ├── registry.py          # Registry mixin (class-lookup via `lookup`)
│   ├── repos.py             # ChunkStore / DocStore / SessionStore
│   ├── rerank_result.py     # Re-rank output helper
│   ├── response/            # Response shaping + Redaction
│   ├── retrieval/           # Rerank, Transformer, Fusion, Retrieval, search, judge
│   ├── retry.py             # aretry() helper
│   ├── routes/              # FastAPI router composition
│   ├── runtime.py           # capture() re-export
│   ├── services/            # ApplicationFacade, RagContainer, Documents, Query, Health, Preference
│   ├── sse/                 # SSE framing
│   ├── stores/              # MemoryStore, SqliteStore, PgVectorStore, JsonSessions, …
│   ├── telemetry/           # Langfuse, Logger, OpenTelemetry, Redaction
│   ├── tenants/             # TenantResolver, IsolationStrategy, Isolation helpers
│   ├── tools/               # ToolRegistry + built-in tools
│   ├── typed_dicts.py       # Type-only dict aliases
│   ├── types.py             # JSONValue recursive alias
│   └── __init__.py          # Flat public surface
├── tests/                   # Test suite
├── devtools/                # Bench harness, Finance/FRAMES pipelines
└── pyproject.toml
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
```

Run the gates:

```bash
ruff check raghub/ tests/ devtools/
ruff format --check raghub/ tests/ devtools/
mypy raghub/
interrogate -c pyproject.toml
bandit -c pyproject.toml -r raghub/ -ll -i
pip-audit
```

## Testing

```bash
python -m pytest tests/ -q                                  # full suite (1688 passed, 4 skipped)
python -m pytest tests/ -q -k rbac                          # RBAC-focused subset
python -m pytest tests/ --cov=raghub --cov-report=term-missing
```

Live-Postgres tests are gated on `RAG_TEST_PG_VECTOR_DSN` and
`RAG_TENANT_DSNS` env vars; without them those four tests skip.
The collection is reported by `pytest tests/ --collect-only`
(1692 tests at the time of writing). The suite covers the
ingestion pipeline, vector store operations, LiteLLM provider
plumbing (mocked), multi-user RBAC isolation, session-scoped
conversation history, streaming, opaque session-token auth,
Finance / FRAMES evaluators, the plugin registry and entry-point
discovery, every CLI sub-command, persistence, query-cache
TTL/invalidation, OTel span guards, document lifecycle state
machines, and the lazy-import facade.

## Build

```bash
python -m build
```

## Release

Releases are tag-driven. Bump the version in `pyproject.toml`,
push a `vX.Y.Z` tag, and the release workflow builds the
sdist/wheel, re-runs the test/lint/type gates, and publishes to
PyPI via OIDC trusted publishing (no API token secret required).

```bash
# Local pre-release gates
pytest -q \
    --cov=raghub --cov-report=term-missing --cov-fail-under=85
ruff check raghub/ tests/ devtools/
ruff format --check raghub/ tests/ devtools/
mypy raghub/
interrogate -c pyproject.toml
bandit -c pyproject.toml -r raghub/ -ll -i

# Tag and push (publishing is automated).
git tag vX.Y.Z && git push origin vX.Y.Z
```

## Benchmarking

```bash
# Finance evaluation
raghub-financebench --examples 25

# Performance benchmark (startup, throughput, p50/p95 latency, peak RSS)
python -m devtools.benchmark --documents 100 --queries 200 --concurrency 8
```

Reports are written to `devtools/report.json`.

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.12+ |
| Chunking | Chonkie |
| LLM / embeddings | LiteLLM |
| Structured output | Instructor (optional `[structured]`) |
| Vector store | pgvector (recommended) / SQLite in-process (default) |
| Observability | Langfuse, OpenTelemetry (optional `[otel]`) |
| Knowledge format | Open Knowledge Format (OKF) |
| Web framework | FastAPI (optional `[api]`) |
| Evaluation | Finance / FRAMES |
| Lint / format | ruff |
| Type check | mypy (strict) |
| Tests | pytest, hypothesis |

## Roadmap

- **v0.10.0** — Released: Pydantic removal (Snap dataclass layer),
  `Registry.lookup` rename, forward-only drop of every
  backward-compat shim, and every CI gate (ruff, ruff-format,
  mypy, interrogate, bandit, pytest + coverage) green.
- **v0.11.0** — Coverage back to ≥ 85% on the unmodified
  `coverage.run.omit` set; expand `interrogate` docstring coverage
  to private helpers; bump `langfuse` and `litellm` minor versions
  once their API drifts are absorbed.
- **v0.12.0** — Public observability story: OTel exporter parity
  with the Langfuse path, plus optional `raghub.otel.runtime`
  helpers; tighten the tenant-isolation contract with
  `DatabasePerTenant` schema introspection; add a streaming-aware
  eval harness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Vulnerability reporting, supported versions, and the disclosure
timeline live in [SECURITY.md](SECURITY.md). The CI pipeline
runs `bandit` over `raghub/` and `pip-audit` against the declared
dependency set on every push.

## License

[MIT](LICENSE) © 2026 Sachin
