# RAGHub

Production-grade multi-user retrieval-augmented generation platform built on
the spec libraries. The single recommended entry point is
[`raghub.RAG`](https://github.com/sachncs/raghub); every collaborator
behind it — Marker, Chonkie, LiteLLM, Qdrant, Langfuse, Instructor — is
replaceable.

## Package layout

```text
raghub/
  api/                RAG facade, FastAPI server, Streamlit UI
  auth/               JWT authentication, RBAC (legacy service paths)
  cli/                Console scripts (``raghub``, ``raghub-financebench``)
  config/             AppSettings dataclass, YAML/TOML loader
  models/             Typed Pydantic domain + canonical models
  interfaces/         Protocol contracts for every plugin point
  converters/         Marker, plain-text, Markdown, OKF normaliser
  knowledge/          Open Knowledge Format (OKF) bundles + repository
  ingestion/          Ingest + Query pipelines, background jobs
  embeddings/         LiteLLM, SentenceTransformers, hashing
  vectorstore/        Qdrant, ZVec (legacy), InMemory
  llm/                LiteLLM, Heuristic (offline) providers
  structured/         Instructor (typed Pydantic outputs)
  retrieval/          Pipeline, IdentityReranker, search
  generation/         DefaultGenerator (citations, astream, tokens)
  pipelines/          IngestPipeline, QueryPipeline
  observability/      NoOpTelemetry, RedactingTelemetry, StructlogTelemetryProvider
  telemetry/          LangfuseTelemetryProvider (v3+ SDK)
  evaluation/         FinanceBenchEvaluator + retrieval metrics
  plugins/            PluginRegistry + entry-point discovery
  services/           DynamicRagApplication (legacy use-case facade)
  repositories/       SQLite repositories (legacy)
  domain/             Active-record models (legacy)
  storage/            SQLite persistence, image store (legacy)
  core/               DI container, RBAC, document state (legacy)
  documents/          Parsers, chunkers, validation (legacy)
```

The top half of the tree (above the *legacy* split) is the new
spec-mandated surface. The bottom half is the legacy multi-tenant
service stack, retained for backward compatibility. See
[migration.md](migration.md) for the bridge between the two.

## Quick Start

```python
from raghub import RAG

rag = RAG()
rag.ingest(b"Revenue grew 12% YoY in Q3 2024.")
print(rag.query("revenue").answer)
```

```bash
raghub init -o raghub.yaml
raghub ingest ./documents
raghub query "What was the revenue guidance?"
```

## Documentation

- [Getting started](guide/getting-started.md)
- [Development guide](guide/development.md)
- [Deployment guide](guide/deployment.md)
- [API reference](reference/api.md)
- [Configuration reference](reference/configuration.md)
- [Architecture overview](architecture/overview.md)
- [Design decisions](architecture/decisions.md)
- [Plugin authoring](plugins.md)
- [Migration guide (legacy → RAG facade)](migration.md)
- [Monitoring & observability](operations/monitoring.md)
- [Troubleshooting](troubleshooting.md)
- [Future extensions](future.md)

## Library dependencies (defaults)

| Concern | Library | Default behaviour |
|---|---|---|
| Document conversion | Marker | Falls back to `PlainTextConverter` when Marker is missing |
| Chunking | Chonkie | Falls back to `WordWindowChunker` when Chonkie is missing |
| LLM | LiteLLM | Falls back to `HeuristicLLMProvider` (offline, deterministic) |
| Embeddings | LiteLLM | Falls back to `HashingEmbeddingProvider` (offline) |
| Structured output | Instructor | Returns `None` when Instructor is missing or no API key |
| Vector store | Qdrant | Falls back to `InMemoryVectorStore` when `QDRANT_URL` is unset |
| Telemetry | Langfuse v3+ | Falls back to `NoOpTelemetry` when Langfuse is missing or unconfigured |
| Knowledge format | OKF (Open Knowledge Format) | Canonical persisted representation |

Every default can be replaced through the `RAG(...)` constructor or via
the [plugin registry](plugins.md).
