"""Compatibility shim — :mod:`raghub.pipelines.rag`.

The original god-class lived in this file; it has since been split
into a :mod:`raghub.pipelines.rag` **package** with focused modules
(``ingest``, ``query``, ``conversation``, ``result``). External code
that still does ``from raghub.pipelines.rag import IngestPipeline``
continues to work via this re-export shim.
"""

from raghub.pipelines.rag.ingest import (
    IngestPipeline,
    chunks_from_knowledge_bundle,
    primary_company,
    sha256_checksum,
)
from raghub.pipelines.rag.query import QueryPipeline

__all__ = [
    "IngestPipeline",
    "QueryPipeline",
    "chunks_from_knowledge_bundle",
    "primary_company",
    "sha256_checksum",
]