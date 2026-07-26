"""Default RAG pipelines — ingest, query, and shared helpers.

Each pipeline is a focused module:

* :mod:`raghub.pipelines.rag.ingest` — :class:`IngestPipeline` plus
  the bundle / checksum / tenant helpers.
* :mod:`raghub.pipelines.rag.query` — :class:`QueryPipeline` (the
  ``embed → retrieve → rerank → generate`` orchestrator).
* :mod:`raghub.pipelines.rag.conversation` — :class:`ConversationRouter`
  facade over the conversation store.
* :mod:`raghub.pipelines.rag.result` — :class:`PipelineResultBuilder`
  for constructing pipeline results in a consistent shape.
"""

from raghub.pipelines.rag.conversation import ConversationRouter
from raghub.pipelines.rag.ingest import (
    IngestPipeline,
    chunks_from_knowledge_bundle,
    primary_company,
    sha256_checksum,
)
from raghub.pipelines.rag.query import QueryPipeline
from raghub.pipelines.rag.result import PipelineResultBuilder

__all__ = [
    "ConversationRouter",
    "IngestPipeline",
    "PipelineResultBuilder",
    "QueryPipeline",
    "chunks_from_knowledge_bundle",
    "primary_company",
    "sha256_checksum",
]