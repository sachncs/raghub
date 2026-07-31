# Migration Guide — v0 → v1

The v1 release renames every public symbol to a single-word form
with **no backward-compat aliases**. Code written against the prior
public API must be updated. This document lists every rename with
a one-line upgrade hint.

## Hard rule (no compatibility period)

There is no `DeprecationWarning` period. Old names simply do not
exist any more. If you imported `LLMError`, your import fails at
collection time, and you must rename it to `GenerationError`.

```python
# Old code:
from raghub.errors import LLMError  # ❌ ImportError

# New code:
from raghub.errors import GenerationError  # ✓
```

## Module renames

| Old module | New module | Notes |
| --- | --- | --- |
| `raghub.helper.auth` | `raghub.api_auth` | FastAPI dependency classes (`App`, `Auth`, `Bearer`) |
| `raghub.helper.cli` | `raghub.cli_commands` | Typer command classes |
| `raghub.helper.rate_limit` | `raghub.api_ratelimit` | `RateLimiterMiddleware`, `Token` |
| `raghub.helper.response` | `raghub.api_response` | `ResponseBuilder`, `Redaction` |
| `raghub.helper.sse` | `raghub.api_sse` | `Sse` |
| `raghub.helper.search` | *(deleted)* | Search wrappers never imported; use `Tool.call` directly |

## Class renames

| Old name | New name | Notes |
| --- | --- | --- |
| `AgentBudgetExceeded` | `AgentBudgetError` | exception suffix |
| `CacheMiss` | `CacheMissError` | exception suffix |
| `CacheMissError` | `CacheMissError` | unchanged |
| `ChunkRecord` | `Chunk` | universal schema; field `chunk_id` → `id` |
| `ConfigurationError` | `ConfigurationError` | unchanged |
| `ConversationManager` | `Conversations` | public aggregate |
| `ConversationStore` | `Store` | drop redundant prefix |
| `ConversationTurn` | `Turn` | universal schema |
| `DynamicRagError` | *(deleted)* | `RagHubError` is the canonical base |
| `DocumentLifecycleStatus` | `DocumentLifecycleStatus` | unchanged enum |
| `DocumentRecord` | `Document` | field `document_id` → `id` |
| `DocumentBlock` | `Block` | no prefix |
| `DocumentSection` | `Section` | no prefix |
| `Domain.Document` | `DocumentRef` | renamed to break shadow with model `Document` |
| `Domain.Chunk` | `ChunkRef` | renamed to break shadow with model `Chunk` |
| `Domain.Session` | `SessionWrap` | renamed to break shadow with model `Session` |
| `Embedding` | `Embedding` | unchanged |
| `EmbeddingProvider` | `EmbeddingProvider` | protocol, unchanged |
| `Evaluator` | `Evaluator` | protocol, unchanged |
| `FactoryMixin` | *(removed)* | use plain inheritance |
| `FinancialBench` | `Finance` | single-word |
| `FramesBenchmark` | `Frames` | single-word |
| `GeneratorProtocol` | `GeneratorProtocol` | protocol, unchanged |
| `HybridConfigShim` | *(removed)* | legacy hybrid-config shim |
| `LLMError` | `GenerationError` | exception semantics |
| `LlmJudge` | `LlmJudge` | single-word (renamed from `LLMJudge`) |
| `LongContextRerankPass` | `Context` | rerank pass |
| `MetricsShim` | `Metrics` | drop shim suffix |
| `MissingDep` | `MissingDepError` | exception suffix |
| `ParseScore` | `parse` | function is a single action |
| `PipelineResultBuilder` | `Pipeline.from(...)` | use the factory on `Pipeline` |
| `PipelineState` | `State` | drop redundant prefix |
| `PersistentJobStore` | `JobStore` | single-word |
| `PreProcessor` | *(removed)* | drop the stage splitter |
| `QueryCache` | `Cache` | drop redundant prefix |
| `Query` | `Query` | unchanged |
| `RetrievedChunk` | `Chunk` | was a duplicate-of `Chunk` |
| `SearchRequest` | `Request` (model) / kept under `Request` | renamed to drop `Search` prefix |
| `SearchResponse` | `Response` (model) | renamed to drop `Search` prefix |
| `SearchResult` | `Result` (or `Hit`) | drop redundant prefix |
| `SlidingWindowManager` | `SlidingWindow` | drop `Manager` |
| `SourceManifest` | `Manifest` | drop `Source` |
| `TokenBudgetExceeded` | `TokenBudgetError` | exception suffix |
| `Tokenizer` (eval-side clone) | *(removed)* | use `raghub.conv.Tokenizer` |
| `UserRecord` | `User` | universal schema |
| `SessionRecord` | `Session` | universal schema |
| `Variant` | `Variant` | model, unchanged |
| `Verified` | *(mixed-in via `entity.verify()`)* | pattern, not class |

Exception renames follow one rule: **every concrete exception class
ends in `Error`**. If you see an exception type in `raghub.errors`
that does not end in `Error`, it is a typo from a previous release.

## Field renames (universal entity schema)

The universal schema is: **every canonical entity carries `id`,
`type`, an internal source/target reference, direct child
collections, and a public `verify()` method.** Concretely:

| Model | Old field | New field | Notes |
| --- | --- | --- | --- |
| `Chunk` | `chunk_id` | `id` | new canonical primary key |
| `Document` | `document_id` | `id` | new canonical primary key |
| `Document` | `chunk_ids` | `chunks` | direct child list (`list[Chunk]`) |
| `User` | `user_id` | `id` | new canonical primary key |
| `PipelineResult` | `success` | *(removed)* | use `verify()` instead; check `result.error is None` |
| `Session` | `session_id` | `id` | **deferred**: Session's FK columns (`user_id`) make a wide rename invasive; schedule as Phase 1.7.x subphase |
| `Result` (eval) | `passed` | `passed` | unchanged |

`Session.id` rename is the only remaining field rename; subscribe
to release notes for the patch.

## Helpers and methods

- `try_load_gigatoken()` → `Tokenizer.load()`
- `MemoryConversations` → `Memory` (already canonical, rename was earlier)
- `ConversationRouter` → `Router`
- `PipelineResultBuilder` → `PipelineResult.from(...)` classmethod on the result class
- `IngestionJob` → `Job`
- `PersistentJobStore` → `JobStore`

## Imports cheat sheet

```python
# Old:
from raghub.errors import LLMError, MissingDep
from raghub.models import ChunkRecord, DocumentRecord, SessionRecord
from raghub.rag import RAG
from ragub.helper.auth import App  # typo-prone module name

# New:
from raghub.errors import GenerationError, MissingDepError
from raghub.models import Chunk, Document, Session
from raghub.rag import RAG
from raghub.api_auth import App
```

## Storage migration

The on-disk JSON manifest format gained a `version` field in v1.
Old `manifest.json` files (no `version` key) are auto-migrated on
read. To migrate explicitly:

```bash
python -m raghub.migrate --root ./data
```

The CLI walks `./data` (or any directory) and rewrites every
`manifest.json` in-place to the versioned format. Idempotent:
re-running against an already-v1 manifest does nothing.

## What is **not** deprecated

- `from raghub.rag import RAG` — the high-level facade is unchanged.
- `from raghub.store import SqliteStore, MemoryStore` — storage classes unchanged.
- `from raghub.models import Hit` — Hit still wraps `chunk` (the change is that
  `Hit.chunk_id` is now a *property* on top of `chunk.id`).
- `from raghub.config import Settings` — `Settings` is the same class.

## Rename table — exhaustive `RAG` instance members

```python
# Old API (`rag.RAG().ingest(...)`):
rag.rag_facade = rag                       # the main facade, kept
rag.kb = KnowledgeBundle                  # was a class; now `rag.bundle` is `Bundle`
rag.chunker = WordChunker()                 # still WordChunker; renamed `WordChunker`

# New API:
rag = RAG(converter=..., embedder=..., generator=...)
rag.chunks = ... # explicit list of Chunk records, retrievable via `rag.vector_store`
rag.bundle = Bundle(...)  # Open Knowledge Format bundle
```

## How to migrate your code

Three mechanical replacements will get you most of the way:

```sh
sed -i 's/raghub\.helper/raghub.api_/g; s/\.helper/auth/g'   # helper module split
sed -i 's/ChunkRecord/Chunk/g; s/DocumentRecord/Document/g; s/SessionRecord/Session/g; s/ConversationTurn/Turn/g'
sed -i 's/UserRecord/User/g; s/Domain\./from raghub.domain import /g'
```

After that, refresh the imports for exception renames:

```sh
sed -i 's/LLMError/GenerationError/g'                  # exception rename
sed -i 's/MissingDep/MissingDepError/g'
sed -i 's/AgentBudgetExceeded/AgentBudgetError/g; s/TokenBudgetExceeded/TokenBudgetError/g; s/CacheMiss/CacheMissError/g; s/PipelineFailed/PipelineFailedError/g'
```

The rest (`Chunk.chunk_id`, `Document.document_id`, `User.user_id`) catches the script-level edits made the entity canonical.

## When something breaks

Open an issue. The maintainers will tell you the canonical name.
We do **not** keep aliases around, so the answer is always "the
new name is X, rename your code."
