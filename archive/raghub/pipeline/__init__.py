"""Pipeline package — ingest, query, agentic, cache, router, builder.

This package is a structural split of the former monolithic
``raghub/pipeline.py``. Each submodule owns one concern:

* :mod:`raghub.pipeline.span_support` — small dependency-light utilities
  (timer, awaitable bridge, filter canonicalisation, checksum,
  chunk materialiser, per-request metadata dataclasses).
* :mod:`raghub.pipeline.cache` — :class:`Cache`, the TTL-based
  in-memory query cache.
* :mod:`raghub.pipeline.router` — :class:`Router`, the conversation
  store facade.
* :mod:`raghub.pipeline.pipeline_assembly` — :class:`Flow`, the fluent
  :class:`Pipeline` builder.
* :mod:`raghub.pipeline.ingest` — :class:`Ingest`, the convert →
  chunk → embed → index pipeline.
* :mod:`raghub.pipeline.query` — :class:`QueryPipeline`, the embed
  → retrieve → rerank → generate pipeline.
* :mod:`raghub.pipeline.agent` — :class:`AgentPipeline`, the ReAct
  agent-driven query pipeline.
"""

from __future__ import annotations

from raghub.pipeline.agent import AgentPipeline
from raghub.pipeline.cache import Cache
from raghub.pipeline.ingest import Ingest
from raghub.pipeline.pipeline_assembly import Flow
from raghub.pipeline.query import QueryPipeline
from raghub.pipeline.router import Router
from raghub.pipeline.span_support import (
    DurationTimer,
    IngestResolvedMetadata,
    QueryContext,
    canonical_filters,
    coerce_to_awaitable,
    get_chunks,
    primary_company,
    sha256_checksum,
)

__all__ = [
    "AgentPipeline",
    "Cache",
    "DurationTimer",
    "Flow",
    "Ingest",
    "IngestResolvedMetadata",
    "QueryContext",
    "QueryPipeline",
    "Router",
    "canonical_filters",
    "coerce_to_awaitable",
    "get_chunks",
    "primary_company",
    "sha256_checksum",
]
