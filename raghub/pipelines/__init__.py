"""Pipeline orchestrations.

End-to-end pipelines that combine converters, chunkers, embedders,
vector stores, retrievers, and generators into a single composable
unit.

Public re-exports:

* :class:`IngestPipeline` — convert → chunk → embed → index.
* :class:`QueryPipeline` — embed → retrieve → rerank → generate.
* :class:`AgenticQueryPipeline` — ReAct agent loop with optional
  long-context pass (Phase 7).
"""

from raghub.pipelines.agentic import AgenticQueryPipeline
from raghub.pipelines.rag import IngestPipeline, QueryPipeline

__all__ = ["AgenticQueryPipeline", "IngestPipeline", "QueryPipeline"]