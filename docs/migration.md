# Migration Guide — v0.4 → v0.5

The v0.5 release renames almost every public symbol to a
single-word form. There are **no backward-compat aliases** — the
old names simply do not exist anymore. Code that imported them
must be updated.

## Module renames

| Old | New |
|---|---|
| `raghub.exceptions` | `raghub.errors` |
| `raghub.embeddings` | `raghub.embedder` |
| `raghub.repositories` | `raghub.repos` |
| `raghub.vectorstore` | `raghub.store` |
| `raghub.observability` | `raghub.telemetry` |
| `raghub.generation` | `raghub.gen` |
| `raghub.conversation` | `raghub.conv` |
| `raghub.ingestion` | `raghub.ingest` |
| `raghub.documents` (top-level parser classes) | `raghub.parsers` |
| `raghub.helper.evaluation` | `raghub.eval` |
| `raghub.helper.documents` (lifecycle / chunking) | `raghub.lifecycle` |
| `raghub.helper.retrieval` | `raghub.retrieval` |
| `raghub.helper.services` | `raghub.services` |
| `raghub.helper.storage` | `raghub.stores` |
| `raghub.helper.tools` | `raghub.tools` |

The four re-export wrappers (`raghub.retrieval`,
`raghub.services`, `raghub.storage`, `raghub.tools`) were
deleted — import directly from the canonical paths above.

## Class renames

| Old | New |
|---|---|
| `OptionalDependencyMissing` | `MissingDep` |
| `UserPrincipal` | `User` |
| `PipelineContext` | `PipelineCtx` |
| `BaseEmbeddingProvider` | `Embedder` |
| `HashingEmbeddingProvider` | `Hasher` |
| `LiteLLMEmbeddingProvider` | `LiteLLMEmbedder` |
| `LiteLLMProvider` | `LiteLLM` |
| `BaseLLMProvider` | `Generator` |
| `BaseVectorStore` | `Store` |
| `InMemoryVectorStore` | `MemoryStore` |
| `SqliteVectorStore` | `SqliteStore` |
| `InMemoryKnowledgeRepository` | `MemoryRepo` |
| `InMemoryConversationStore` | `MemoryConversations` |
| `InMemoryQueue` | `MemoryQueue` |
| `SqliteUserStore` | `SqliteUsers` |
| `SqliteChunkRepository` | `ChunkStore` |
| `SqliteDocumentRepository` | `DocStore` |
| `SqliteSessionRepository` | `SessionStore` |
| `DocumentLifecycleManager` | `Lifecycle` |
| `MarkerConverter` | `Marker` |
| `DocumentIngestionService` | `Ingestor` |
| `BackgroundIngestionService` | `Batch` |
| `ResumableBackgroundIngestionService` | `Resumable` |
| `RBACAuthorizationService` | `Authz` |
| `ChonkieChunker` | `Chonkie` |
| `WordWindowChunker` | `WordChunker` |
| `RaptorIndex` | `Raptor` |
| `GraphRagIndex` | `GraphIndex` |
| `SourceManifest` | `Manifest` |
| `InstructorStructuredOutputProvider` | `Instructor` |
| `AgenticQueryPipeline` | `AgentPipeline` |
| `Section` dataclass in `raghub.documents` | `ParsedSection` |
| `MarkdownSection` | `Section` |
| `BaseTool` | `Tool` |

The `*Protocol` classes with the same short name were renamed
with a `Protocol` suffix to make room for the concrete classes
above: `GeneratorProtocol`, `ToolProtocol`,
`SessionStoreProtocol`.

## Field rename

`ChunkRecord.hash` is now `ChunkRecord.checksum` (also required;
no default empty string).

## Function and constant renames

Function renames: `assert_production_invariants → production_check`,
`build_embedding_provider → build_embedder`,
`build_vector_store → build_store`, `build_llm_provider → build_llm`,
`ingest_directory_concurrent → ingest_dir`,
`chunks_from_knowledge_bundle → get_chunks`, plus 25+ more.
See `git log --oneline` for the full list.

Constant renames: `SUMMARISE_COMMUNITY_PROMPT → COMMUNITY_PROMPT`,
`MULTI_QUERY_SYSTEM_PROMPT → MULTI_QUERY`,
`HYDE_SYSTEM_PROMPT → HYDE`, `MARKER_AVAILABLE → MARKER`, plus 5
more.

## Behavioural changes

- `RAG()` no longer requires an LLM API key. When no key is
  configured, `default_llm()` returns `HeuristicProvider` (sentence
  extraction from context).
- `default_converter()` returns `Marker` when `marker-pdf` is
  installed; otherwise falls back to `PlainTextConverter` with a
  `UserWarning`. Install `raghub[pdf]` for PDF support.
- `vectorstore.insert()` and `upsert()` now return `int` (rows
  written). Mismatched dimensions raise `VectorStoreError`.
- `ChunkRecord.checksum` is required; the field is computed as
  `sha256(chunk.text).hexdigest()` at construction.
- Legacy exception aliases (`AuthenticationError`,
  `AuthorizationError`, `ValidationError`, `LLMError`, …) emit a
  `DeprecationWarning` when instantiated. New code should catch
  `RagHubError` (or the canonical subclass).

## Removed dependencies

`qdrant-client`, `zvec`, `sentence-transformers`, and `bge-reranker`
are no longer installed. The vector store is now SQLite-backed
(`SqliteStore`); reranking is local-only.