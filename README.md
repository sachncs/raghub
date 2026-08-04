<p align="center">
  <h1 align="center">RAGHub</h1>
  <p align="center">Production-grade multi-user retrieval-augmented generation platform built on the spec libraries.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/raghub/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/raghub/ci.yml?branch=master" alt="CI"></a>
    <a href="https://pypi.org/project/raghub/"><img src="https://img.shields.io/pypi/v/raghub" alt="PyPI"></a>
    <a href="https://github.com/sachncs/raghub/stargazers"><img src="https://img.shields.io/github/stars/sachncs/raghub" alt="Stars"></a>
  </p>
</p>

**Production-grade multi-user RAG platform built on the spec libraries.**

RAGHub is a layered retrieval-augmented generation stack with a single
replace-everything facade (`raghub.RAG`), multi-tenant RBAC, conversational
memory, resumable ingestion, real streaming, and a Finance
evaluator. Every collaborator (converter, chunker, vector store,
embedder, retriever, generator, telemetry, evaluator) is replaceable
through a registry; the default wiring installs all spec libraries
(Marker, Chonkie, LiteLLM, Instructor, Qdrant, Langfuse) and falls
back to deterministic in-process providers when no API keys are
present, so `pip install` and `import` is enough to be productive.

The framework is fully typed, fully documented (Google-style
docstrings on every public function), and ships with a loguru-backed
logger plus tqdm progress bars in every ingest loop. Production
defaults: fail-closed CORS (wildcard + credentials is rejected),
non-zero `JWT_SECRET` required, opaque session tokens only, and a
single canonical ingestion pipeline that calls the configured
chunker and persists the bundle only after the vector store has
indexed every chunk.

| Concern | Library |
|---|---|
| Document Conversion | [Marker](https://github.com/datalab-to/marker) (optional `[pdf]`) |
| Knowledge Format | Open Knowledge Format (OKF) |
| Chunking | [Chonkie](https://github.com/chonkie-inc/chonkie) |
| LLM + Embeddings | [LiteLLM](https://github.com/BerriAI/litellm) |
| Structured Outputs | [Instructor](https://github.com/567-labs/instructor) (optional `[structured]`) |
| Vector Store | SQLite (in-process) |
| Observability | Langfuse (optional `[langfuse]`) / Prometheus / OpenTelemetry |
| Benchmark | Finance |

## Features

- **Multi-tenant RBAC** — Query results are scoped to each user's `allowed_companies`; admins see everything, unauthorised users see nothing.
- **Conversational history** — Session-scoped turn memory enables natural follow-up questions.
- **Incremental indexing** — Content-addressed by SHA-256 hash; unchanged files are skipped on re-ingest.
- **Real streaming** — `rag.astream` yields tokens as they arrive, with parallel stream-option support.
- **Token-usage tracking** — Every `generate` and `astream` call records prompt / completion token counts.
- **Resumable ingestion** — Persistent SQLite job ledger survives process restarts.
- **Structured output** — Pass a Pydantic `response_model` to `rag.query()` to get typed results via Instructor.
- **Plugin system** — Register custom converters, chunkers, vector stores, evaluators, and telemetry providers.
- **Observability** — Langfuse, OpenTelemetry, Prometheus metrics, and structlog logging out of the box.
- **Evaluation** — Finance evaluator with Recall@K, Precision@K, MRR, Faithfulness, Context Recall, Context Precision, and Answer Correctness.
- **Production safety** — `CORS_ORIGINS` rejects wildcard+credentials at startup; oversize uploads are rejected with `413` before the body is buffered; admin endpoints redact `password_hash`; the demo-user seed is suppressed in production.
- **Query transforms** — HyDE, multi-query, step-back, decomposition. Composable per-query or via `settings.query_transforms.enabled`.
- **Hybrid retrieval** — Dense + BM25 fused by Reciprocal Rank Fusion (k=60). Optional ColBERT late-interaction channel.
- **Rerankers** — Cohere, BGE, LLM-as-judge, cascade. Wired via `settings.reranker.provider`.
- **Long-context second pass** — Reorders the top-K with a long-context LLM (Claude 3.5/3.7, Gemini 1.5/2.0, Command-R+, GPT-4.1).
- **RAPTOR** — Recursive summary tree at ingest time; flat-tree search across every level.
- **GraphRAG** — Entity / community graph with LLM-driven triple extraction + summarisation.
- **Agentic retrieval** — ReAct planner with streaming `PlannerEvent` (thought / tool_call / tool_result / answer_chunk / final). Seven built-in tools; subclass `BaseTool` for more. Per-tool RBAC, error isolation, strict budgets.
- **Per-user tool preferences** — Three layers of override (request > session > user > global); persisted via API and CLI.
- **Streaming endpoints** — `POST /v1/query/stream` (Server-Sent Events), `POST /v1/agent/run`, and `RAG.astream_agent`.

See [`docs/ADVANCED_RAG.md`](docs/ADVANCED_RAG.md) for the full reference.

## Installation

### From PyPI

```bash
pip install raghub
pip install "raghub[api,structured,langfuse,pdf]"   # optional extras
```

### From source

```bash
git clone https://github.com/sachncs/raghub.git
cd raghub
pip install -e ".[dev,api]"
```

| Extra | Includes |
|---|---|
| `dev` | pytest, ruff, mypy, hypothesis, types-PyYAML, interrogate, mkdocs, build, bandit, pip-audit |
| `api` | FastAPI, uvicorn, python-multipart |
| `pdf` | marker-pdf, pypdf |
| `graph` | python-igraph, leidenalg, scikit-learn |
| `rerank` | cohere, ragatouille, rank-bm25 |
| `web` | duckduckgo-search |
| `docs` | beautifulsoup4, Pillow, openpyxl, python-docx, python-pptx |
| `auth` | bcrypt, aiosqlite |
| `otel` | opentelemetry-api, opentelemetry-sdk, opentelemetry-instrumentation-fastapi |
| `langfuse` | langfuse |
| `tiktoken` | tiktoken |
| `structured` | instructor |
| `eval` | datasets |
| `all` | everything |

Core deps only: pydantic, numpy, PyYAML, chonkie, litellm, loguru, tqdm, typer, prometheus-client, rank-bm25. Optional extras add the rest.

## Quick Start

### Python API

```python
from raghub import RAG

rag = RAG()
rag.ingest(b"Revenue grew 12 percent in Q3 2024.")
print(rag.query("revenue").answer)
```

Or with async:

```python
import asyncio
from raghub import RAG

async def main():
    rag = RAG()
    rag.ingest(b"Revenue grew 12 percent in Q3 2024.")
    result = await rag.aquery("revenue")
    print(result.answer)

asyncio.run(main())
```

No API keys required — RAGHub falls back to `HeuristicProvider` for offline use. Set `RAG_LLM_API_KEY` (or any provider-specific env var) for real LLM completions.

### CLI

```bash
raghub init -o raghub.yaml
raghub ingest ./documents
raghub query "What was the revenue guidance?"
raghub health
raghub version
raghub run --host 0.0.0.0 --port 8000   # start the FastAPI server in the foreground
```

The CLI emits status events through the loguru logger
(`cli.ingest`, `cli.version`, `cli.rate_limit_exceeded`, etc.); the
default sink is stderr at the configured log level. The
`RAGHUB_CLI_RATE_LIMIT` and `RAGHUB_CLI_RATE_BURST` env vars
control the per-subcommand rate limit.

### FastAPI

```bash
raghub run --host 0.0.0.0 --port 8000
# or, equivalently, via uvicorn:
uvicorn raghub.api:app_factory.create_app --factory --host 0.0.0.0 --port 8000
```

The legacy `RagApplication` is still reachable at
`/auth/login`, `/documents/upload`, `/query`, etc. The new `RAG`
facade is the recommended path for new integrations. The `--factory`
flag tells Uvicorn to call `get_app()` on each worker, which is the
correct way to use the app factory without falling back to a
module-level singleton.

## Configuration

Settings live in :mod:`raghub.config` and are loaded by
:meth:`Settings.load` from environment variables plus an optional
profile. Every config value is read from `os.environ` at process
start, and missing required values (e.g. `JWT_SECRET` in
production) raise immediately. See `Settings.load()` for the full
env-var contract.

Profile search order (first match wins):

1. `RAG_CONFIG_DIR` env var (explicit override)
2. `./config` relative to CWD
3. `~/.config/raghub` (XDG-style user config)
4. Bundled `config/` shipped with the package

Real credentials belong in `.env` (gitignored). See `.env.example`
for the template used by `devtools/financebench.py`.

## Multi-User & RBAC

Every public query method accepts a `User`. The retrieval layer is filtered to the user's `allowed_companies`; admins see everything. The LLM is given only the authorised context — there is no path by which unauthorised content can leak into the prompt.

```python
from raghub.models import User

alice = User(user_id="alice", email="alice@x", allowed_companies=["Apple"])
bob   = User(user_id="bob",   email="bob@x",   allowed_companies=["Microsoft"])
admin = User(user_id="admin", email="admin@x", is_admin=True)

rag.query("revenue", user=alice)   # Apple-only chunks
rag.query("revenue", user=bob)     # Microsoft-only chunks
rag.query("revenue", user=admin)   # all chunks
```

A user with no `allowed_companies` and no `is_admin` sees no documents (the filter resolves to `{"company": []}` which matches nothing). Unauthorised retrieval attempts return an empty result set.

## Conversational RAG

Every public query method accepts a `session_id`. The pipeline loads the most recent turns from `Memory` (or a custom `ConversationStore`) and prepends them to the prompt so the LLM can answer follow-up questions.

```python
await rag.aquery("revenue", user=alice, session_id="alice-s1")
# Bob's session is isolated; Alice's session has its own history.
await rag.aquery("and growth?", user=alice, session_id="alice-s1")
```

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `RAGHUB_USERS` | yes | inline demo users | JSON path or inline JSON for the user directory |
| `RAG_LLM_API_KEY` | yes | unset | LLM provider key (preferred; falls back to `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`, `LITELLM_API_KEY`) |
| `RAG_LLM_BASE_URL` | yes | unset | OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`) |
| `RAG_LLM_MODEL` | yes | `gpt-4o-mini` | Model name passed to LiteLLM |
| `JWT_SECRET` | yes | random | Opaque session-token signing secret (≥32 bytes) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | yes | unset | Langfuse credentials |
| Constructor kwargs | no | — | Passed to `RAG(...)` (highest precedence) |

Precedence (highest first): constructor arguments → env vars → built-in defaults. `Settings.override(**changes)` returns a new instance with the given fields changed (the original is not mutated).

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `raghub.RAG` | class | Single facade; lazy-imports every collaborator |
| `raghub.config.Settings` | class | Typed configuration loaded from env + YAML |
| `raghub.models.User` | model | Per-user identity with `allowed_companies` and `is_admin` |
| `raghub.api:AppFactory.create_app` | factory | FastAPI app factory (use `uvicorn raghub.api:app_factory.create_app --factory`) |
| `raghub.plugins.Plugins` | class | Register converters, chunkers, vector stores, etc. |
| `raghub.eval.Finance` | class | Recall@K, Precision@K, MRR, Faithfulness, Context Recall/Precision, Answer Correctness |
| `raghub.cli:main` | CLI | `raghub init / ingest / query / health / version` |
| `raghub-financebench` | CLI | `raghub-financebench --examples N` |

## Examples

Plugins register converters, chunkers, embedders, vector stores, retrievers, rerankers, generators, telemetry providers, and evaluators on `Plugins`. They are discovered via entry points (`group="raghub.plugins"`) and can be registered programmatically:

```python
from raghub.plugins import Plugins
from raghub.lifecycle import Marker

registry = Plugins()
registry.register_converter("marker", Marker())
rag = RAG(registry=registry)
```

Structured output with Pydantic (requires `pip install raghub[structured]`):

```python
from pydantic import BaseModel

class Revenue(BaseModel):
    amount: float
    currency: str

result = await rag.aquery(
    "What was 2024 revenue?",
    response_model=Revenue,
)
print(result.structured.amount, result.structured.currency)
```

## Project Structure

```
raghub/
├── raghub/                 # The library
│   ├── __init__.py         # Public entry: from raghub import RAG
│   ├── rag.py              # RAG(...) facade
│   ├── config.py           # Settings (env + YAML)
│   ├── models.py           # Pydantic domain models
│   ├── errors.py           # Typed error hierarchy
│   ├── llm.py              # LiteLLMProvider, HeuristicProvider
│   ├── embedder.py         # LiteLLMEmbedder
│   ├── store.py            # MemoryStore, SqliteStore
│   ├── parsers.py          # MarkerConverter, PlainTextConverter
│   ├── pipeline.py         # IngestPipeline, QueryPipeline
│   ├── ingest.py           # Chunker, Ingestor, Resumable
│   ├── gen.py              # DefaultGenerator, Instructor
│   ├── retrieval/          # Rerankers, transformers, fusion
│   ├── stores/             # Database, Sessions, ImageStore
│   ├── services/           # Facade, container wiring
│   ├── tools/              # ToolRegistry + built-in tools
│   ├── lifecycle/          # Document lifecycle state machines
│   ├── authhelpers/        # Auth helpers (App, Auth, Bearer)
│   ├── response/           # ResponseBuilder + Redaction
│   ├── sse/                # SSE framing
│   ├── ratelimit/          # Token bucket + ASGI middleware
│   ├── commands/           # CLI sub-commands
│   ├── knowledge.py        # RAPTOR, GraphRAG
│   ├── conv.py             # Conversation memory
│   ├── telemetry.py        # Telemetry providers, Prometheus metrics
│   ├── agent.py            # ReAct planner + tools
│   ├── auth.py             # UserStore, AuthService, Authz
│   ├── repos.py            # ChunkStore, DocStore, SessionStore
│   ├── api.py              # FastAPI app
│   ├── cli.py              # raghub CLI entry
│   └── evaluation.py       # Finance / FRAMES eval harnesses
├── tests/                  # Test suite
├── devtools/               # Bench harness, Finance/FRAMES pipelines
└── pyproject.toml
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Linting and formatting:

```bash
ruff check raghub/ tests/ devtools/
ruff format raghub/ tests/ devtools/
mypy raghub/
interrogate -v raghub/ -f 80
bandit -r raghub/ -q -ll -i
pip-audit
```

## Testing

```bash
python -m pytest tests/ -q                       # full suite
python -m pytest tests/ -q -k rbac               # just the RBAC suite
python -m pytest tests/ --cov=raghub --cov-report=term-missing
```

Platform and dynamic-application tests run as part of the normal test
run. The current collection size is reported by
`pytest tests/ --collect-only` (no hard-coded count). The suite
covers ingestion pipelines, vector store operations, LiteLLM
providers (mocked), multi-user RBAC isolation (10 concurrent
users), session-scoped conversation history, streaming and
token-usage tracking, opaque session-token auth and
unauthorised-access isolation, Finance evaluation metrics, the plugin registry
and entry-point discovery, all CLI commands, persistence (JSON
registry, SQLite stores), query-cache TTL/invalidation, tracing
exporters and OTel span guards, document lifecycle state
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
mypy raghub/

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
|----------|------------|
| Language | Python 3.12+ |
| Document conversion | Marker (optional `[pdf]`) |
| Chunking | Chonkie |
| LLM / embeddings | LiteLLM |
| Structured output | Instructor (optional `[structured]`) |
| Vector store | SQLite (in-process) |
| Observability | Langfuse, OpenTelemetry, Prometheus, loguru |
| Knowledge format | Open Knowledge Format (OKF) |
| Web framework | FastAPI (optional `[api]`) |
| Evaluation | Finance |
| Lint / format | ruff |
| Type check | mypy (strict optional) |
| Tests | pytest, hypothesis |

## Roadmap

- **v0.5.0** — Released: comprehensive renaming refactor (modules,
  classes, functions, constants), single-word names, themed
  subpackages for retrieval/services/stores/tools/eval/lifecycle,
  offline `RAG()` via `HeuristicProvider` (no API key required),
  PDF fallback to `PlainTextConverter` when marker-pdf is missing,
  `__all__` declared on every public module, 325 tests passing.
- **v0.6.0** — Planned: per-tenant rate limits, expanded reranker
  registry, streaming-first ingestion UI, group ABAC.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).

## Security

Vulnerability reporting, supported versions, and the disclosure
timeline live in [SECURITY.md](SECURITY.md). The CI pipeline
runs `bandit` over `raghub/` and `pip-audit --strict` against the
declared dependency set on every push.

## License

[MIT](LICENSE) © 2026 Sachin
