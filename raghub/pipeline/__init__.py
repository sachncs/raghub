"""Pipeline package — ingest, query, agentic, cache, router, builder.

This package is a structural split of the former monolithic
``raghub/pipeline.py``. Each submodule owns one concern:

* :mod:`raghub.pipeline.helpers` — small dependency-light utilities
  (timer, awaitable bridge, filter canonicalisation, checksum,
  chunk materialiser, per-request metadata dataclasses).
* :mod:`raghub.pipeline.cache` — :class:`Cache`, the TTL-based
  in-memory query cache.
* :mod:`raghub.pipeline.router` — :class:`Router`, the conversation
  store facade.
* :mod:`raghub.pipeline.builder` — :class:`Flow`, the fluent
  :class:`Pipeline` builder (with the legacy :class:`PipelineBuilder`
  alias).
* :mod:`raghub.pipeline.ingest` — :class:`Ingest`, the convert →
  chunk → embed → index pipeline.
* :mod:`raghub.pipeline.query` — :class:`QueryPipeline`, the embed
  → retrieve → rerank → generate pipeline.
* :mod:`raghub.pipeline.agent` — :class:`AgentPipeline`, the ReAct
  agent-driven query pipeline.

The public names are re-exported here so existing imports —
``from raghub.pipeline import Ingest`` — continue to work unchanged.
"""

from __future__ import annotations

from raghub.pipeline.agent import AgentPipeline
from raghub.pipeline.builder import Flow, PipelineBuilder
from raghub.pipeline.cache import Cache
from raghub.pipeline.helpers import (
    DurationTimer,
    IngestResolvedMetadata,
    QueryContext,
    awaitable,
    canonical_filters,
    get_chunks,
    primary_company,
    sha256_checksum,
)
from raghub.pipeline.ingest import Ingest
from raghub.pipeline.query import QueryPipeline
from raghub.pipeline.router import Router

__all__ = [
    "AgentPipeline",
    "Cache",
    "DurationTimer",
    "Flow",
    "Ingest",
    "IngestResolvedMetadata",
    "PipelineBuilder",
    "QueryContext",
    "QueryPipeline",
    "Router",
    "awaitable",
    "canonical_filters",
    "get_chunks",
    "primary_company",
    "sha256_checksum",
]
