# RAGHub

Production-grade multi-user RAG platform built on the spec libraries:

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

## Installation

```bash
git clone https://github.com/quantiphi/raghub.git
cd raghub
pip install -e ".[api,ui,dev]"
```

The `api` extra installs FastAPI and uvicorn; `ui` installs Streamlit;
`dev` installs pytest, ruff, mypy, and the type stubs.

To install the spec libraries (Marker, Chonkie, LiteLLM,
Instructor, Qdrant, Langfuse, datasets), they are already in
`install_requires`. The RAG facade works offline without API keys —
it falls back to a deterministic in-process provider.

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

## FastAPI

```bash
uvicorn raghub.api.app:app --host 0.0.0.0 --port 8000
```

The legacy `DynamicRagApplication` is still reachable at
`/auth/login`, `/documents/upload`, `/query`, etc. The new
`RAG` facade is the recommended path for new integrations.

## FinanceBench

```bash
raghub-financebench --examples 25
# or
python -m bench.benchmark
```

The evaluator now reports Recall@K, Precision@K, MRR, Faithfulness,
Context Recall, Context Precision, and Answer Correctness in
addition to the pass-rate.

## Performance Benchmark

```bash
python -m bench.benchmark --documents 100 --queries 200 --concurrency 8
```

The benchmark measures startup time, ingestion throughput, query
latency (p50/p95), queries-per-second under concurrency, and peak
RSS. The report is written to `bench/report.json`.

## Architecture

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
  bench/              Performance benchmark script
```

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

## License

MIT.
