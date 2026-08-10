"""Document ingestion workflows.

This package exposes four ingestion concerns in separate files:

* :mod:`raghub.ingest.ingestor` — :class:`Ingestor`, the synchronous
  ingestion service over the canonical ingest pipeline (the public API /
  CLI callers both hit this), plus :class:`IngestionResult`.
* :mod:`raghub.ingest.jobs` — :class:`Batch` / :class:`Job`
  thread-pool-backed fire-and-forget ingestion with status tracking,
  :class:`Resumable` (a persistent SQLite ledger so jobs survive
  restarts), and :class:`Jobs`, the ledger itself.
* :mod:`raghub.ingest.chunker` — :class:`Words`, the built-in
  overlap-aware chunker, :class:`Chonkie` with its supported
  strategies, and :func:`build_chonkie_chunker` strategy dispatch.
"""

from raghub.ingest.chunker import (
    Chonkie,
    Words,
    apply_refinery,
    build_chonkie_chunker,
    build_refinery,
)
from raghub.ingest.ingestor import (
    IngestionResult,
    Ingestor,
    record_from_pipeline,
)
from raghub.ingest.jobs import (
    Batch,
    Job,
    Jobs,
    Resumable,
)

__all__ = [
    "Batch",
    "Chonkie",
    "IngestionResult",
    "Ingestor",
    "Job",
    "Jobs",
    "Resumable",
    "Words",
    "apply_refinery",
    "build_chonkie_chunker",
    "build_refinery",
    "record_from_pipeline",
]
