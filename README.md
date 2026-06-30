# RAGHub

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Test Status](https://img.shields.io/badge/tests-351%20passed-brightgreen.svg)](tests/)
[![Ruff](https://img.shields.io/badge/ruff-0%20errors-brightgreen.svg)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/mypy-0%20errors-brightgreen.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Production-grade multi-user RAG platform built on the **spec libraries**:

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

Every collaborator is replaceable through the public
[`raghub.RAG`](raghub/api/rag.py) facade.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [CLI](#cli)
- [Streamlit UI](#streamlit-ui)
- [FastAPI](#fastapi)
- [Multi-User & RBAC](#multi-user--rbac)
- [Conversational RAG](#conversational-rag)
- [Configuration](#configuration)
- [Development](#development)
- [Project Structure](#project-structure)
- [Plugins](#plugins)
- [Benchmarking](#benchmarking)
- [Changelog](#changelog)
- [License](#license)

---

## Quick Start

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

No API keys required — RAGHub falls back to deterministic in-process
providers for all spec libraries. Just `pip install` and go.

---

## Features

- **Multi-tenant RBAC** — query results are scoped to each user's `allowed_companies`.
  Admins see everything; unauthorised users see nothing.
- **Conversational history** — session-scoped turn memory enables
  natural follow-up questions.
- **Incremental indexing** — content-addressed by SHA-256 hash;
  unchanged files are skipped on re-ingest.
- **Real streaming** — `rag.astream` yields tokens as they arrive,
  with parallel stream-option support.
- **Token-usage tracking** — every `generate` and `astream` call
  records prompt/completion token counts.
- **Resumable ingestion** — persistent SQLite job ledger survives
  process restarts.
- **Structured output** — pass a Pydantic `response_model` to
  `rag.query()` to get typed results via Instructor.
- **Plugin system** — register custom converters, chunkers, vector
  stores, evaluators, and telemetry providers.
- **Observability** — Langfuse, OpenTelemetry, Prometheus metrics,
  and structlog logging out of the box.
- **Evaluation** — FinanceBench evaluator with Recall@K,
  Precision@K, MRR, Faithfulness, Context Recall, Context
  Precision, and Answer Correctness.

---

## Installation

```bash
git clone https://github.com/sachn-cs/raghub.git
cd raghub
pip install -e ".[dev,api,ui,zvec]"
```

| Extra | Includes |
|---|---|
| `dev` | pytest, ruff, mypy, hypothesis, types-PyYAML |
| `api` | FastAPI, uvicorn, python-multipart |
| `ui` | Streamlit |
| `zvec` | [ZVec](https://github.com/zilliztech/zvec) vector store |

All spec libraries (Marker, Chonkie, LiteLLM, Instructor, Qdrant,
Langfuse, datasets) are installed by default.

For a minimal environment:
```bash
pip install -e .           # core only
pip install -e ".[dev]"    # core + dev tooling
```

---

## CLI

```bash
# Emit a starter YAML config
raghub init -o raghub.yaml

# Ingest a file or directory
raghub ingest ./documents

# Ask a question
raghub query "What was the revenue guidance?"

# Liveness probe
raghub health

# Print the version
raghub version
```

---

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

The UI pre-seeds five demo users with different `allowed_companies`:

| Email | Companies | Admin? |
|---|---|---|
| alice@acme.com | Apple | No |
| bob@acme.com | Microsoft | No |
| charlie@acme.com | Amazon, Tesla | No |
| diana@acme.com | Google | No |
| admin@acme.com | (all) | Yes |

The default password is `password`. Override the user directory
by setting `RAGHUB_USERS` to a JSON mapping.

The UI uses `st.chat_message` + `st.chat_input` for a real chat
experience with follow-up questions, conversation history, and
per-turn citation rendering.

---

## FastAPI

```bash
uvicorn raghub.api.app:app --host 0.0.0.0 --port 8000
```

The legacy `DynamicRagApplication` is still reachable at
`/auth/login`, `/documents/upload`, `/query`, etc. The new
`RAG` facade is the recommended path for new integrations.

---

## Multi-User & RBAC

Every public query method accepts a `UserPrincipal`. The retrieval
layer is filtered to the user's `allowed_companies`; admins see
everything. The LLM never receives unauthorised context.

```python
alice = UserPrincipal(user_id="alice", email="alice@x", allowed_companies=["Apple"])
bob   = UserPrincipal(user_id="bob",   email="bob@x",   allowed_companies=["Microsoft"])
admin = UserPrincipal(user_id="admin", email="admin@x", is_admin=True)

# Alice sees only Apple chunks; Bob only Microsoft; admin sees all.
rag.query("revenue", user=alice)
rag.query("revenue", user=bob)
rag.query("revenue", user=admin)
```

A user with no `allowed_companies` and no `is_admin` sees **no
documents** (the filter resolves to `{"company": []}` which
matches nothing). An unauthorised user attempting to retrieve
another tenant's content receives an empty result set.

---

## Conversational RAG

Every public query method accepts a `session_id`. The pipeline
loads the most recent turns from the
[`InMemoryConversationStore`](raghub/conversation/memory.py) (or a
custom `ConversationStore`) and prepends them to the prompt so the
LLM can answer follow-up questions.

```python
await rag.aquery("revenue", user=alice, session_id="alice-s1")
# Bob's session is isolated; Alice's session has its own history.
await rag.aquery("and growth?", user=alice, session_id="alice-s1")
```

---

## Configuration

Configuration precedence (highest first):

1. Constructor arguments to `RAG(...)`.
2. Environment variables (`RAG_*`, `JWT_SECRET`, `NVIDIA_API_KEY`,
   `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY`, `QDRANT_URL`).
3. TOML config (`config/<profile>.toml`).
4. YAML config (`config/<profile>.yaml`).
5. Built-in defaults.

`AppSettings.override(**changes)` returns a new instance with the
given fields changed (the original is not mutated). This is the
runtime-override mechanism.

---

## Development

```bash
# Automated setup (venv + install)
./setup.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,ui,zvec]"

# Run tests
python -m pytest tests/ -q

# Run linter
python -m ruff check raghub/

# Run type checker
python -m mypy raghub/
```

### Test suite

351 tests covering:

- Ingestion pipelines (plain-text, PDF through Marker)
- Vector store operations (Qdrant, in-memory, ZVec)
- LiteLLM embedding and LLM providers (with mocked responses)
- Multi-user RBAC isolation (10 concurrent users)
- Session-scoped conversation history
- Streaming and token-usage tracking
- Security — JWT auth, unauthorised access isolation
- FinanceBench evaluation metrics
- Plugin registry and entry-point discovery
- CLI commands (health, ingest, query, init)
- Persistence (JSON document registry, SQLite stores)

---

## Project Structure

```
raghub/
  api/                RAG facade (single entry point), FastAPI, Streamlit
  config/             YAML / TOML configuration
  models/             Typed Pydantic domain models (Document, Chunk, Citation, ...)
  interfaces/         Protocol contracts (DocumentConverter, VectorStore, ...)
  converters/         Marker, plain-text, OKF normaliser
  knowledge/          OKF serialisation + InMemoryKnowledgeRepository
  ingestion/          IngestPipeline (convert → chunk → embed → upsert)
  embeddings/         LiteLLM, SentenceTransformers, hashing
  vectorstore/        Qdrant, ZVec, InMemory
  llm/                LiteLLM, Heuristic
  structured/         Instructor
  retrieval/          Pipeline, IdentityReranker
  generation/         DefaultGenerator (citations, astream, token usage)
  pipelines/          IngestPipeline, QueryPipeline
  observability/      NoOpTelemetry, RedactingTelemetry, StructlogTelemetryProvider
  telemetry/          LangfuseTelemetryProvider, NoopSpan, LangfuseSpan
  evaluation/         FinanceBenchEvaluator, retrieval metrics
  plugins/            PluginRegistry, entry-point discovery
  cli/                CLI commands (health, ingest, query, init, eval)
```

---

## Plugins

Plugins can register converters, chunkers, embedders, vector stores,
retrievers, rerankers, generators, telemetry providers, or
evaluators on a `raghub.plugins.registry.PluginRegistry`. They are
discovered via entry points (`group="raghub.plugins"`) and can be
registered programmatically:

```python
from raghub.plugins.registry import PluginRegistry
from raghub.converters.marker import MarkerConverter

registry = PluginRegistry()
registry.register_converter("marker", MarkerConverter())
rag = RAG(registry=registry)
```

---

## Benchmarking

```bash
# FinanceBench evaluation
raghub-financebench --examples 25

# Performance benchmark
python -m bench.benchmark --documents 100 --queries 200 --concurrency 8
```

The performance benchmark measures startup time, ingestion throughput,
query latency (p50/p95), queries-per-second under concurrency, and
peak RSS. The report is written to `bench/report.json`.

The FinanceBench evaluator reports Recall@K, Precision@K, MRR,
Faithfulness, Context Recall, Context Precision, and Answer
Correctness.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## License

MIT.
