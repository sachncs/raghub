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

RAGHub is a layered retrieval-augmented generation stack with a single replace-everything facade (`raghub.RAG`), multi-tenant RBAC, conversational memory, resumable ingestion, real streaming, and a FinanceBench evaluator. Every collaborator (converter, chunker, vector store, embedder, retriever, generator, telemetry, evaluator) is replaceable through a registry; the default wiring installs all spec libraries (Marker, Chonkie, LiteLLM, Instructor, Qdrant, Langfuse) and falls back to deterministic in-process providers when no API keys are present, so `pip install` and `import` is enough to be productive.

| Concern | Library |
|---|---|
| Document Conversion | [Marker](https://github.com/datalab-to/marker) |
| Knowledge Format | Open Knowledge Format (OKF) |
| Chunking | [Chonkie](https://github.com/chonkie-inc/chonkie) |
| LLM + Embeddings | [LiteLLM](https://github.com/BerriAI/litellm) |
| Structured Outputs | [Instructor](https://github.com/567-labs/instructor) |
| Vector Store | [Qdrant](https://github.com/qdrant/qdrant) (ZVec optional) |
| Observability | [Langfuse](https://github.com/langfuse/langfuse) |
| Benchmark | [FinanceBench](https://github.com/patronus-ai/financebench) |

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
- **Evaluation** — FinanceBench evaluator with Recall@K, Precision@K, MRR, Faithfulness, Context Recall, Context Precision, and Answer Correctness.
- **Production safety** — `CORS_ORIGINS` rejects wildcard+credentials at startup; oversize uploads are rejected with `413` before the body is buffered; admin endpoints redact `password_hash`; the demo-user seed is suppressed in production.

## Installation

### From PyPI

```bash
pip install raghub
pip install "raghub[api,ui,zvec]"   # optional extras
```

### From source

```bash
git clone https://github.com/sachncs/raghub.git
cd raghub
pip install -e ".[dev,api,ui]"
```

| Extra | Includes |
|---|---|
| `dev` | pytest, ruff, mypy, hypothesis, types-PyYAML, interrogate, mkdocs, build, bandit, pip-audit |
| `api` | FastAPI, uvicorn, python-multipart |
| `ui` | Streamlit |
| `zvec` | [ZVec](https://github.com/zilliztech/zvec) vector store (opt-in) |

All spec libraries (Marker, Chonkie, LiteLLM, Instructor, Qdrant, Langfuse, datasets) are installed by default. For a minimal environment use `pip install -e ".[dev]"`. The `zvec` extra pulls a native extension and is no longer part of the default install.

## Quick Start

### Python API

```python
from raghub import RAG
from raghub.models import UserPrincipal

rag = RAG()
alice = UserPrincipal(user_id="alice", email="alice@x", allowed_companies=["Apple"])
rag.ingest(open("report.pdf", "rb").read(), source_uri="file://report.pdf", user=alice)
response = await rag.aquery("What was the revenue?", user=alice, session_id="alice-s1")
print(response.answer)
print(response.citations)
```

No API keys required — RAGHub falls back to deterministic in-process providers for all spec libraries.

### CLI

```bash
raghub init -o raghub.yaml
raghub ingest ./documents
raghub query "What was the revenue guidance?"
raghub health
raghub version
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI pre-seeds five demo users with different `allowed_companies`: `alice@acme.com` (Apple), `bob@acme.com` (Microsoft), `charlie@acme.com` (Amazon, Tesla), `diana@acme.com` (Google), `admin@acme.com` (admin). The default password is `password`. Override the user directory with the `RAGHUB_USERS` environment variable.

### FastAPI

```bash
uvicorn raghub.api.app:get_app --factory --host 0.0.0.0 --port 8000
```

The legacy `DynamicRagApplication` is still reachable at `/auth/login`, `/documents/upload`, `/query`, etc. The new `RAG` facade is the recommended path for new integrations. The `--factory` flag tells Uvicorn to call `get_app()` on each worker, which is the correct way to use the app factory without falling back to a module-level singleton.

## Deployment (Docker)

```bash
cp .env.example .env
# Edit .env: set JWT_SECRET, an LLM API key, and CORS_ORIGINS.
openssl rand -base64 48   # generate JWT_SECRET

docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml --profile production up -d

# Verify health
docker compose -f docker-compose.yml --profile production ps
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:6333/healthz
```

The production stack runs three services (`api`, `ui`, `qdrant`)
with hardened defaults: read-only root, `cap_drop: [ALL]`,
`no-new-privileges`, JSON-file log rotation, named volumes for
SQLite + Qdrant, service-aware healthchecks, and
`depends_on: condition: service_healthy`. The `env_file` is
declared with `required: true`, so the stack fails closed if
`.env` is missing.

For local development:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml \
    --profile dev up
```

`docker-compose.dev.yml` is an explicit override; the dev
mounts, `--reload`, and relaxed security live there so the
production target stays clean.

See [`docs/guide/deployment.md`](docs/guide/deployment.md),
[`docs/operations/backup.md`](docs/operations/backup.md),
[`docs/operations/runbook.md`](docs/operations/runbook.md), and
[`docs/operations/scaling.md`](docs/operations/scaling.md) for
the full operations story.

## Multi-User & RBAC

Every public query method accepts a `UserPrincipal`. The retrieval layer is filtered to the user's `allowed_companies`; admins see everything. The LLM is given only the authorised context — there is no path by which unauthorised content can leak into the prompt.

```python
alice = UserPrincipal(user_id="alice", email="alice@x", allowed_companies=["Apple"])
bob   = UserPrincipal(user_id="bob",   email="bob@x",   allowed_companies=["Microsoft"])
admin = UserPrincipal(user_id="admin", email="admin@x", is_admin=True)

rag.query("revenue", user=alice)   # Apple-only chunks
rag.query("revenue", user=bob)     # Microsoft-only chunks
rag.query("revenue", user=admin)   # all chunks
```

A user with no `allowed_companies` and no `is_admin` sees no documents (the filter resolves to `{"company": []}` which matches nothing). Unauthorised retrieval attempts return an empty result set.

## Conversational RAG

Every public query method accepts a `session_id`. The pipeline loads the most recent turns from `InMemoryConversationStore` (or a custom `ConversationStore`) and prepends them to the prompt so the LLM can answer follow-up questions.

```python
await rag.aquery("revenue", user=alice, session_id="alice-s1")
# Bob's session is isolated; Alice's session has its own history.
await rag.aquery("and growth?", user=alice, session_id="alice-s1")
```

## Configuration

| Setting | Env Variable | Default | Description |
|---------|--------------|---------|-------------|
| `RAGHUB_USERS` | yes | inline demo users | JSON path or inline JSON for the user directory (Streamlit UI) |
| `RAGHUB_STORE_BACKEND` | yes | `memory` | `memory` / `file` / `qdrant` / `zvec` |
| LLM provider keys | yes | unset | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `NVIDIA_API_KEY` |
| `JWT_SECRET` | yes | random | JWT signing secret |
| `QDRANT_URL` | yes | unset | Qdrant server URL |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | yes | unset | Langfuse credentials |
| TOML/YAML profile | no | — | `config/<profile>.toml` or `.yaml` |
| Constructor kwargs | no | — | Passed to `RAG(...)` (highest precedence) |

Precedence (highest first): constructor arguments → env vars → TOML config → YAML config → built-in defaults. `AppSettings.override(**changes)` returns a new instance with the given fields changed (the original is not mutated).

## API

| Symbol | Type | Description |
|--------|------|-------------|
| `raghub.RAG` | class | Single facade; lazy-imports every collaborator |
| `raghub.build_application` | function | Legacy application builder |
| `raghub.models.UserPrincipal` | model | Per-user identity with `allowed_companies` and `is_admin` |
| `raghub.api.app:get_app` | factory | FastAPI app factory (use `uvicorn raghub.api.app:get_app --factory`) |
| `raghub.plugins.registry.PluginRegistry` | class | Register converters, chunkers, vector stores, etc. |
| `raghub.evaluation.FinanceBenchEvaluator` | class | Recall@K, Precision@K, MRR, Faithfulness, Context Recall/Precision, Answer Correctness |
| `raghub.cli.main` | CLI | `raghub init / ingest / query / health / version` |
| `raghub.cli.eval_cmd.main` | CLI | `raghub-financebench --examples N` |

## Examples

Plugins register converters, chunkers, embedders, vector stores, retrievers, rerankers, generators, telemetry providers, and evaluators on `PluginRegistry`. They are discovered via entry points (`group="raghub.plugins"`) and can be registered programmatically:

```python
from raghub.plugins.registry import PluginRegistry
from raghub.converters.marker import MarkerConverter

registry = PluginRegistry()
registry.register_converter("marker", MarkerConverter())
rag = RAG(registry=registry)
```

Structured output with Pydantic:

```python
from pydantic import BaseModel

class Revenue(BaseModel):
    amount: float
    currency: str

result = await rag.aquery(
    "What was 2024 revenue?",
    user=alice,
    response_model=Revenue,
)
print(result.amount, result.currency)
```

## Project Structure

```
raghub/
├── src/raghub/
│   ├── api/                # RAG facade, FastAPI, Streamlit
│   ├── config/             # YAML / TOML configuration
│   ├── models/             # Typed Pydantic domain models (Document, Chunk, Citation)
│   ├── interfaces/         # Protocol contracts (DocumentConverter, VectorStore)
│   ├── converters/         # Marker, plain-text, OKF normaliser
│   ├── knowledge/          # OKF serialisation + InMemoryKnowledgeRepository
│   ├── ingestion/          # IngestPipeline (convert → chunk → embed → upsert)
│   ├── embeddings/         # LiteLLM, SentenceTransformers, hashing
│   ├── vectorstore/        # Qdrant, ZVec, InMemory
│   ├── llm/                # LiteLLM, Heuristic
│   ├── structured/         # Instructor
│   ├── retrieval/          # Pipeline, IdentityReranker
│   ├── generation/         # DefaultGenerator (citations, astream, token usage)
│   ├── pipelines/          # IngestPipeline, QueryPipeline
│   ├── observability/      # NoOpTelemetry, RedactingTelemetry, StructlogTelemetryProvider
│   ├── telemetry/          # LangfuseTelemetryProvider, NoopSpan, LangfuseSpan
│   ├── evaluation/         # FinanceBenchEvaluator, retrieval metrics
│   ├── plugins/            # PluginRegistry, entry-point discovery
│   └── cli/                # CLI commands (health, ingest, query, init, eval)
├── tests/                  # Run `pytest --collect-only` for the current count
├── bench/                  # Performance benchmark harness (startup, QPS, RSS)
├── streamlit_app.py        # Demo Streamlit UI
├── pyproject.toml
└── setup.sh                # venv + dev-deps bootstrap
```

## Development

```bash
./setup.sh
# or:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,ui]"
```

Linting and formatting:

```bash
ruff check raghub/
ruff format raghub/
mypy raghub/
interrogate -v \
    raghub/api/rag.py raghub/api/defaults.py raghub/api/response.py \
    raghub/evaluation/ raghub/knowledge/ \
    raghub/conversation/ raghub/cli/ -f 80
bandit -r raghub/ -q -ll -i
pip-audit
```

## Testing

```bash
python -m pytest tests/ -q                       # full suite
python -m pytest tests/ -q -k rbac               # just the RBAC suite
python -m pytest tests/ --cov=raghub --cov-report=term-missing
RAGHUB_RUN_PLATFORM_TESTS=1 python -m pytest tests/test_platform.py
```

The current collection size is reported by
`pytest tests/ --collect-only` (no hard-coded count). The suite
covers ingestion pipelines, vector store operations, LiteLLM
providers (mocked), multi-user RBAC isolation (10 concurrent
users), session-scoped conversation history, streaming and
token-usage tracking, JWT auth and unauthorised-access
isolation, FinanceBench evaluation metrics, the plugin registry
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
pytest -q --ignore=tests/test_financebench.py \
    --cov=raghub --cov-report=term-missing --cov-fail-under=90
ruff check raghub/ tests/ examples/ bench/
mypy raghub/

# Tag and push (publishing is automated).
git tag vX.Y.Z && git push origin vX.Y.Z
```

## Benchmarking

```bash
# FinanceBench evaluation
raghub-financebench --examples 25

# Performance benchmark (startup, throughput, p50/p95 latency, peak RSS)
python -m bench.benchmark --documents 100 --queries 200 --concurrency 8
```

Reports are written to `bench/report.json`.

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.12+ |
| Document conversion | Marker |
| Chunking | Chonkie |
| LLM / embeddings | LiteLLM |
| Structured output | Instructor |
| Vector store | Qdrant (ZVec optional) |
| Observability | Langfuse, OpenTelemetry, Prometheus, structlog |
| Knowledge format | Open Knowledge Format (OKF) |
| Web framework | FastAPI |
| Demo UI | Streamlit |
| Evaluation | FinanceBench |
| Lint / format | ruff |
| Type check | mypy (strict optional) |
| Tests | pytest, hypothesis |

## Roadmap

- **v0.3.x** — Current: RAG facade, multi-tenant RBAC, conversational memory, streaming, plugins, FinanceBench evaluator.
- **v0.4.0** — Planned: expanded plugin entry-point zoo, advanced rerankers (cross-encoder, Cohere), per-tenant rate limits.
- **v0.5.0** — Planned: streaming-first ingestion UI, query-cache topology, ZVec-backed config presets.

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
