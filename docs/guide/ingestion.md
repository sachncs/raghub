# Ingestion pipeline

Revex ingests documents through a content-addressed, idempotent pipeline. A
document is identified by its SHA-256 `hash`; re-ingesting the same bytes is a
no-op (existing chunks win).

## Lifecycle

A `Document` moves through `DocumentLifecycleStatus`:
`pending → indexing → ready` (or `failed`). Transitions are guarded by
`assertTransition` / `LifecycleTransitionError`; `isTerminal` /
`isInFlight` describe a state.

## `ingest`

```ts
import { ingest } from '@revex/core';

const output = await ingest(input, deps);
```

- **Input** — raw bytes/text plus workspace/owner/collection scoping.
- **Deps** — an embedder, a vector store, and a chunker.
- **Output** — the resulting chunks and a `deduplicated: boolean` flag.

`hashDocument(input)` computes the SHA-256 content address.

## `ingestVerbose`

```ts
import { ingestVerbose, IngestEmitter } from '@revex/core';

const emitter = new IngestEmitter();
emitter.on('chunked', (e) => console.log(e));
const output = await ingestVerbose({ ...input, emitter }, deps);
```

Emits an `IngestEvent` stream of phases: `start`, `parsed`, `chunked`,
`embedding`, `persisting`, `indexed`, `skipped`, `failed`. The API exposes this
via the SSE endpoint `POST /v1/documents/ingest-stream`.

## `agenticIngest`

```ts
import { agenticIngest } from '@revex/core';

const output = await agenticIngest(input, deps);
```

Runs ingestion with parallel side-effects through a graph store, a memory
store, and a summary index. Returns `AgenticIngestSideEffect` records
(`graph` | `memory` | `summary`).

## Chunking

`@revex/core` chunker:

- `chunkText(text, opts)` → `TextChunk[]`
- `chunkMarkdown(text, opts)`
- `chunkStructured(input, opts)`
- `chunkPdf(buf, opts)` → `PdfChunkResult`

`ChunkOptions` controls window/chunk size. Each `Chunk` carries a `modality`
(`text` | `image` | `table` | `equation` | `layout` | `summary`).

## MIME detection

The lifecycle scanner (`packages/core/src/lifecycle/scanner.ts`) exposes
`detectMimeType`, `MIME_TYPES`, and `MAGIC_BYTES` for sniffing file types.

## Via the API

Documents are uploaded as multipart form data:

- `POST /v1/documents` — returns `202 {documentId, status:'pending'}`; the
  `JobWorker` drains the queue, runs `ingest()`, flips the row to
  `ready`/`failed`, and writes an `ingest.complete`/`ingest.failure` audit event.
- `POST /v1/documents/ingest-stream` — same, but streams SSE progress events.

The `JobWorker` (started by `start()`) polls the queue (500ms interval, batch
size 4) and `resetStuckJobs()` re-queues stale `running` rows. See
[Workers & jobs](workers-jobs.md).