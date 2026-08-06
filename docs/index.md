# RAGHub

Production-grade multi-user retrieval-augmented generation platform built on
the spec libraries. The single recommended entry point is
[`raghub.RAG`](https://github.com/sachncs/raghub); every collaborator
behind it — Marker, Chonkie, LiteLLM, Qdrant, Langfuse, Instructor — is
replaceable.

## Package layout

```text
raghub/
  rag.py              RAG facade, FastAPI server
  api.py              FastAPI app (App.create)
  cli.py              Console scripts (``raghub``, ``raghub-financebench``)
  config.py           Settings dataclass, YAML/TOML loader
  models.py           Typed Pydantic domain + canonical models
  errors.py           Typed error hierarchy
  llm.py              LiteLLM, Heuristic (offline) providers
  embedder.py         LiteLLM, hashing embedders
  store.py            Qdrant, SQLite, InMemory vector stores
  parsers.py          Marker, plain-text converters
  pipeline.py         Ingest + Query pipelines
  gen.py              DefaultGenerator (citations, astream, tokens)
  ingest.py           Chunker, Ingestor, Resumable
  knowledge.py        Open Knowledge Format (OKF) bundles + repository
  retrieval/          Rerankers, transformers, fusion
  stores/             SQLite persistence, image store
  services/           Facade, container wiring
  tools/              ToolRegistry + built-in tools
  lifecycle/          Document lifecycle state machines
  helper/             Internal collaborators (auth, cli, sse, …)
  telemetry.py        NoOp, Redacting, Langfuse, Prometheus telemetry
  evaluation.py       Finance + FRAMES evaluators
  plugins.py          PluginRegistry + entry-point discovery
  auth.py             UserStore, AuthService, Authz
  repos.py            ChunkStore, DocStore, SessionStore
```

The `RAG` facade is the recommended entry point; the FastAPI
surface in `api.py` is the multi-tenant HTTP wrapper. See
[migration.md](migration.md) for the path from the legacy API.

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
- [Backup & restore](operations/backup.md)
- [Runbook](operations/runbook.md)
- [Scaling](operations/scaling.md)
- [Troubleshooting](troubleshooting.md)
- [Future extensions](future.md)

## Library dependencies (defaults)

| Concern | Library | Default behaviour |
|---|---|---|
| Document conversion | Marker | Falls back to `PlainTextConverter` when Marker is missing |
| Chunking | Chonkie | Falls back to `WordChunker` when Chonkie is missing |
| LLM | LiteLLM | Falls back to `HeuristicLLMProvider` (offline, deterministic) |
| Embeddings | LiteLLM | Falls back to `Hasher` (offline) |
| Structured output | Instructor | Returns `None` when Instructor is missing or no API key |
| Vector store | Qdrant | Falls back to `MemoryStore` when `QDRANT_URL` is unset |
| Telemetry | Langfuse v3+ | Falls back to `NoOpTelemetry` when Langfuse is missing or unconfigured |
| Knowledge format | OKF (Open Knowledge Format) | Canonical persisted representation |

Every default can be replaced through the `RAG(...)` constructor or via
the [plugin registry](plugins.md).
