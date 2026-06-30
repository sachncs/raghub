# Architecture Overview

RAGHub is a layered platform. The spec-mandated entry point is
[`raghub.RAG`](../..//raghub/api/rag.py); below it sit ingestion and
query pipelines, a knowledge layer in OKF, plugin-replaceable
adapters, and the legacy multi-tenant service stack retained for
backward compatibility.

## Two coexisting surfaces

The package exposes two parallel APIs:

```text
                      ┌─────────────────────────────────────────────┐
                      │                  raghub.RAG                │
                      │  (single entry point, replaceable parts)   │
                      └────────┬────────────────────────┬──────────┘
                               │                        │
                               ▼                        ▼
                  ┌───────────────────────┐    ┌─────────────────────────┐
                  │   IngestPipeline      │    │     QueryPipeline       │
                  │ convert→chunk→embed   │    │ embed→search→rerank→    │
                  │ →upsert               │    │ generate→stream         │
                  └────────────┬──────────┘    └────────────┬────────────┘
                               │                             │
                               ▼                             ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Knowledge layer (OKF), embedding provider, vector store,    │
        │  LLM, structured-output provider, telemetry, conversation   │
        │  store, reranker                                             │
        └──────────────────────────────────────────────────────────────┘
```

The second surface is the legacy multi-tenant service, still
mounted at `raghub.api.app:app` (the FastAPI routers):

```text
                      ┌─────────────────────────────────────────────┐
                      │          DynamicRagApplication              │
                      │  (use-case facade: DocumentService,         │
                      │   QueryService, AuthService, HealthService) │
                      └────────┬────────────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │   Active-record models     │
                  │   Document → Chunk → …     │
                  │   + SQLite repositories    │
                  └────────────────────────────┘
```

Both surfaces share **plugin-replaceable adapters** (converters,
chunkers, embedders, vector stores, LLMs, structured-output
providers, telemetry providers) wired through the
[`PluginRegistry`](../plugins.md).

## The `RAG` facade

`raghub.api.rag.RAG` is the spec-mandated entry point. It is a thin
DI container that wires:

| Collaborator | Type | Default |
|---|---|---|
| `settings` | `AppSettings` | from `load_settings()` |
| `registry` | `PluginRegistry` | empty |
| `converter` | `DocumentConverter` | `MarkerConverter` → `PlainTextConverter` fallback |
| `chunker` | `Chunker` | `ChonkieChunker` → `WordWindowChunker` fallback |
| `embedder` | `EmbeddingProvider` | `LiteLLMEmbeddingProvider` → `HashingEmbeddingProvider` fallback |
| `llm` | `LLMProvider` | `LiteLLMProvider` → `HeuristicLLMProvider` fallback |
| `vector_store` | `VectorStore` | `QdrantVectorStore` (when `QDRANT_URL` set) → `InMemoryVectorStore` fallback |
| `generator` | `Generator` | `DefaultGenerator` wrapping `llm` |
| `structured` | `StructuredOutputProvider` | `InstructorStructuredOutputProvider` (when key + Instructor present) else `None` |
| `telemetry` | `TelemetryProvider` | `RedactingTelemetry(LangfuseTelemetryProvider)` → `RedactingTelemetry(NoOpTelemetry)` |
| `reranker` | `Reranker` | `IdentityReranker` |
| `knowledge_repo` | `KnowledgeRepository` | `InMemoryKnowledgeRepository` |
| `conversation_store` | `ConversationStore` | `InMemoryConversationStore` |
| `manifest` | `SourceManifest` | `data/manifest.json` |

Replace any of these through the constructor or via the registry.

## Ingestion

`IngestPipeline.run` performs, in order:

```text
file_bytes
  └─► converter.convert()      ── creates KnowledgeBundle (OKF)
        └─► knowledge_repo.save(bundle)
              └─► chunks_from_knowledge_bundle()
                    └─► embedder.embed_texts(texts)
                          └─► vector_store.upsert(chunks, vectors)
```

Incremental indexing is on by default: the pipeline computes
`sha256(file_bytes)` and asks `knowledge_repo.get(bundle_id)` for a
prior checksum. If the prior checksum matches, no re-embedding
happens — the existing chunks are returned with
`outputs["incremental"] = True`.

Multi-user tenancy: when a `UserPrincipal` is supplied, the chunk
`owner` is set to the user's email and the primary
`allowed_companies` entry becomes the document tenant.

## Query

`QueryPipeline.run` performs, in order:

```text
question
  └─► embedder.embed_text(question)
        └─► vector_store.search(top_k=k, metadata_filter=...)
              └─► metadata_filter_for_user(user)   # RBAC
                    └─► reranker.rerank() (optional)
                          └─► conversation_store.load(session_id) (optional)
                                └─► generator.generate()
                                      └─► optional structured.generate()
                                            └─► conversation_store.append()
```

`QueryPipeline.stream` is the streaming variant. It uses
`Generator.astream` when available, falls back to word-by-word
yields from the synchronous generator otherwise.

## RBAC

`QueryPipeline.metadata_filter_for_user`:

- `user=None` → no filter (returns `""`).
- `user.is_admin == True` → no filter.
- `user.allowed_companies == ["Apple"]` → `{"company": ["Apple"]}`
  (Qdrant / in-memory store) or the equivalent SQL fragment.
- `user.allowed_companies == []` → `{"company": []}` — matches
  nothing. The LLM never sees unauthorised context.

See [`reference/configuration.md`](../reference/configuration.md) for
how tenant companies are attached to chunks during ingestion.

## Knowledge layer

The canonical persisted representation is **OKF (Open Knowledge
Format)**, modelled by `KnowledgeBundle`. Round-trip:

```python
from raghub.knowledge.okf import to_okf, from_okf
okf = to_okf(bundle)   # dict[str, Any]
restored = from_okf(okf)
```

`InMemoryKnowledgeRepository` keeps bundles keyed by
`bundle_id = deterministic_id("bundle", source_uri, checksum)`.

## Vector stores

| Store | Where | When it is used |
|---|---|---|
| `QdrantVectorStore` | `raghub.vectorstore.qdrant` | When `QDRANT_URL` is set and `qdrant-client` is installed |
| `InMemoryVectorStore` | `raghub.vectorstore.memory` | When no `QDRANT_URL` (test, local dev) |
| `ZVecVectorStore`     | `raghub.vectorstore.zvec`   | Retained for the legacy surface when `require_zvec` is `True` (production profile) |

All backends share the `VectorStore` interface defined in
[`raghub.interfaces.vectorstore`](../..//raghub/interfaces/vectorstore.py):
`upsert`, `search`, `delete_document`, optional `create_collection`.

## LLM providers

| Provider | Where | When |
|---|---|---|
| `LiteLLMProvider` | `raghub.llm.litellm` | Any LiteLLM model (OpenAI, Anthropic, NVIDIA, Bedrock, Cohere, Voyage, …) |
| `HeuristicLLMProvider` | `raghub.llm.heuristic` | Offline / test; deterministic |

Both expose `generate`, `astream`, and `last_usage`; the
`DefaultGenerator` forwards token counters to telemetry.

## Embedding providers

| Provider | When |
|---|---|
| `LiteLLMEmbeddingProvider` | API key set (`OPENAI_API_KEY`, `NVIDIA_API_KEY`, etc.) |
| `HashingEmbeddingProvider` | No API key (offline; deterministic) |
| `SentenceTransformerEmbeddingProvider` | Optional explicit choice |

## Telemetry

| Provider | When |
|---|---|
| `LangfuseTelemetryProvider` | `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` set, langfuse installed |
| `NoOpTelemetry` | otherwise |
| `RedactingTelemetry` | wraps *any* of the above to scrub secret-looking kwargs |
| `StructlogTelemetryProvider` | combined logger + Prometheus sink + OTel tracer (legacy surface only) |

## Security and multi-tenancy

- `UserPrincipal` carries `allowed_companies` and `is_admin`.
- The retrieval layer enforces RBAC; `AuthorizationError` is raised
  only by the legacy FastAPI surface when JWT auth fails.
- `JWT_SECRET` must be ≥ 32 bytes in production (PyJWT's
  `InsecureKeyLengthWarning` is treated as fatal in CI).
- `RedactingTelemetry` removes secret kwargs before forwarding to
  Langfuse.
- `allow_passwordless_login: true` is **forbidden in production**
  (`load_settings` raises `RuntimeError`).

## Package layout

The new surface lives under:

```
raghub/
  api/                RAG facade, FastAPI app, Streamlit helper
  cli/                Console scripts
  config/             AppSettings + load_settings
  models/             Pydantic DTOs (UserPrincipal, Chunk, Citation, …)
  interfaces/         Protocol contracts (converter, embedder, llm, …)
  converters/         Marker, plaintext, markdown, OKF normaliser
  knowledge/          OKF bundles, InMemoryKnowledgeRepository
  ingestion/          IngestPipeline, QueryPipeline, background jobs
  embeddings/         LiteLLM, SentenceTransformers, hashing
  vectorstore/        Qdrant, ZVec, InMemory
  llm/                LiteLLM, Heuristic
  structured/         Instructor
  retrieval/          Pipeline, IdentityReranker
  generation/         DefaultGenerator
  pipelines/          IngestPipeline, QueryPipeline (re-exports)
  observability/      NoOpTelemetry, RedactingTelemetry, structlog
  telemetry/          Langfuse (v3+) provider
  evaluation/         FinanceBench evaluator + retrieval metrics
  plugins/            PluginRegistry
```

The legacy service stack (retained for backward compatibility) lives
under:

```
  auth/               bcrypt + JWT (DynamicRagApplication)
  services/           DocumentService, QueryService, AuthService, HealthService
  repositories/       SQLite repositories + UnitOfWork
  storage/            SQLite persistence, JSON registry, image store
  domain/             Active-record models (Document, Chunk, Session)
  core/               DI container, RBAC, document state
  documents/          Parsers, validation, lifecycle
  prompts/            PromptBuilder (used by the legacy surface)
  api/admin.py        Admin routes mounted under /admin
```

See [`migration.md`](../migration.md) for the bridge.
