# RAGHub Architecture (v0.6)

This document captures the class-boundary decisions introduced in
RAGHub 0.6.0: the structural split of every god-class, the new
exception taxonomy, and the clean-up of the dependency graph.

## Class boundaries

The single ``RAG`` facade still exists for backward compatibility,
but its public methods now delegate to focused helper classes:

```
RAGCore
├── RAGQuery     — sync + async query dispatch (RAG.query / RAG.aquery)
├── RAGStream    — SSE streaming (RAG.astream / RAG.astream_agent)
├── RAGIngest    — ingest / aingest / delete (RAG.ingest / RAG.aingest)
└── RAGAgent     — agentic loop (RAG.run_agent / RAG.arun_agent)
```

The ``api/app.py`` FastAPI surface composes focused classes
rather than monolitihic route registration:

```
create_app
├── Lifespan                 — startup + shutdown coordinator
├── RouteGroup               — admin / rag / preferences routers
├── DependencyProviders       — current_user / settings / unit_of_work
├── StreamingResponseBuilder — SSE event marshalling
└── SseFormatter             — serialise PlannerEvent → Server-Sent Event
```

The ``pipelines/rag.py`` god-class is replaced by a 4-module
package:

```
pipelines/rag/
├── ingest.py        — IngestPipeline class
├── query.py         — QueryPipeline class
├── conversation.py  — ConversationRouter class
└── result.py        — PipelineResultBuilder helpers
```

The ``services/application.py`` 741-line facade becomes:

```
services/application/
├── facade.py         — ApplicationFacade (public surface)
├── shutdown.py       — ShutdownCoordinator
├── auth.py           — AuthCoordinator
└── preferences.py    — PreferenceCoordinator
```

## Dependency graph

Edges flow in one direction only:
**interfaces → core → storage → vectorstore → retrieval →
generation → pipelines → services → api/cli**.

The ``raghub.interfaces`` package is the leaf: pure Protocols
with zero runtime dependencies on other ``raghub.*`` modules.
Concrete implementations (e.g. ``Raptor``, ``CohereReranker``)
import the relevant Protocol and the dependency-injection seam
that uses them.

## Exception taxonomy

`raghub/exceptions/__init__.py` provides a single
``RagHubError`` base and a typed subhierarchy:

| Exception | Raised when |
|---|---|
| `ConfigurationError` | Bad configuration / missing secrets |
| `ConversionError` | Marker / parser / converter failure |
| `KnowledgeError` | OKF / knowledge repository failure |
| `IngestionError` | Chunking or pipeline failure |
| `EmbeddingError` | Model / dimension problem |
| `VectorStoreError` | Backend search / persistence failure |
| `RetrievalError` | RBAC / filter / retriever failure |
| `GenerationError` | LLM provider failure |
| `PipelineError` | Orchestration / lifecycle failure |
| `EvaluationError` | Benchmark / scoring failure |
| `RerankerError` | Reranker scoring failure |
| `WebSearchError` | Web search tool failure |
| `ToolError` | Agent tool failure |
| `AgentBudgetExceeded` | Agent loop budget exhausted |
| `GraphUnavailableError` | Graph-backed feature (RAPTOR / GraphRAG) requested but missing dep |
| `TransformError` | Query transform (HyDE / multi-query / decompose) failure |
| `TelemetryError` | Telemetry provider failure (non-fatal at boundaries) |
| `MissingDep` | Optional runtime dep not installed |
| `PipelineFailed` | Orchestration pipeline step failure |
| `StreamingFormatError` | SSE event formatting failure |
| `TokenBudgetExceeded` | Operation exceeded its token budget |
| `CacheMiss` | ``cache.get_or_raise()`` miss |

Legacy names (``AuthenticationError``, ``AuthorizationError``,
``DocumentError``, ``IndexingError``, ``PromptError``, ``LLMError``,
``StorageError``, ``ValidationError``, ``RateLimitError``) are
preserved as backward-compatible aliases.

## Naming policy

The codebase enforces: **no leading-underscore semi-private names**.

Renamed in v0.6.0:

| Old | New |
|---|---|
| `_active_metrics` (module global) | `MetricsRegistry.set / .current` |
| `_env_bool` | `env_bool` |
| `_csv_to_transforms` | `csv_to_transforms` |
| `_pdf_mod` / `_models_mod` / `_output_mod` | `pdf_module` / `models_module` / `output_module` |
| `self._device` | `self.device` |
| `self._llm` | `self.llm` |
| `self._ttl` / `self._store` | `self.ttl` / `self.store` |
| `_native_filter` | `native_filter` |
| `_build_refinery` / `_apply_refinery` | `build_refinery` / `apply_refinery` |
| `_drive_extraction` / `_drive_summarisation` | `drive_extraction` / `drive_summarisation` |

Python-language dunders (`__init__`, `__repr__`, `__hash__`,
`__iter__`, `__next__`, `__aiter__`, `__anext__`, `__aenter__`,
`__aexit__`, `__enter__`, `__exit__`, `__getattr__`,
`__setattr__`, `__contains__`, `__len__`, `__getitem__`,
`__setitem__`, `__delitem__`, `__call__`, `__await__`,
`__bool__`, comparator dunders, module-level `__all__`) are kept
— they are the Python data-model protocol, not semi-private
conventions.
