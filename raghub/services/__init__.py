"""Application services, container, and worker primitives.

Public surface is re-exported from focused submodules:

    helpers     - emit_log, emit_metric, probes, aggregate_status, seed_blocked,
                  parse_users, build_models, upload_record, missing_doc,
                  seed_demo_users.
    documents   - Documents service plus list_records/get_doc helpers.
    health      - Health service.
    query       - Query service.
    workers     - Synchronous, ThreadPool, MemoryQueue primitives.
    container   - RagContainer dataclass and build_container.
    shutdown    - Shutdown coordinator.
    preference  - Preference router for advanced-RAG flags.
    facade      - Facade plus RAG_FACADE_AVAILABLE.

The module-level dispatch entry points live in :mod:`raghub.api`
and the CLI surface in :mod:`raghub.cli.main`.
"""

from __future__ import annotations

from raghub.services.container import RagContainer, build_container
from raghub.services.documents import Documents, get_doc, list_records
from raghub.services.facade import RAG_FACADE_AVAILABLE, Facade
from raghub.services.health import Health
from raghub.services.diagnostics import (
    aggregate_status,
    build_models,
    emit_log,
    emit_metric,
    missing_doc,
    parse_users,
    probe_embedder,
    probe_vector_store,
    seed_blocked,
    seed_demo_users,
    upload_record,
)
from raghub.services.preference import Preference
from raghub.services.query import Query
from raghub.services.shutdown import Shutdown
from raghub.services.workers import MemoryQueue, Synchronous, ThreadPool

__all__ = [
    "RAG_FACADE_AVAILABLE",
    "Documents",
    "Facade",
    "Health",
    "MemoryQueue",
    "Preference",
    "Query",
    "RagContainer",
    "Shutdown",
    "Synchronous",
    "ThreadPool",
    "aggregate_status",
    "build_container",
    "build_models",
    "emit_log",
    "emit_metric",
    "get_doc",
    "list_records",
    "missing_doc",
    "parse_users",
    "probe_embedder",
    "probe_vector_store",
    "seed_blocked",
    "seed_demo_users",
    "upload_record",
]
