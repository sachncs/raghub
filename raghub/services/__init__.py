"""Application services, container, and worker primitives.

Public surface re-exported from this package; the implementation lives
in :mod:`raghub.services.helper`.

Class summary::

    Mixin                - structured-log + metric helpers shared by every service.
    Document             - document management.
    Health               - liveness aggregation.
    Query                - RAG hot path.
    Synchronous / ThreadPool / InMemoryQueue
                          - in-process worker + queue primitives.
    RagContainer         - composition root for every collaborator.
    Facade               - high-level facade exposing every public action;
                          ``RagApplication`` is a legacy alias of ``Facade``.

Helpers::

    build_container      - construct a fully-wired :class:`RagContainer`.
    seed_demo_users      - populate the demo user set.
    parse_seed_users_json- parse the ``RAGHUB_USERS`` env var.
    seed_blocked         - ``True`` when prod / wildcard CORS suppresses seeding.
    probe_vector_store / probe_embedder / aggregate_status
                        - low-level health probes.
"""

from __future__ import annotations

from raghub.services.helper import (
    Auth,
    Document,
    Facade,
    Health,
    InMemoryQueue,
    Mixin,
    Preference,
    Query,
    RagApplication,
    RagContainer,
    Shutdown,
    Synchronous,
    ThreadPool,
    aggregate_status,
    build_container,
    document_by_id_helper,
    list_all_records_helper,
    parse_seed_users_json,
    probe_embedder,
    probe_vector_store,
    raise_missing_document,
    seed_blocked,
    seed_demo_users,
    upload_record_helper,
)


__all__ = [
    "Auth",
    "Document",
    "Facade",
    "Health",
    "InMemoryQueue",
    "Mixin",
    "Preference",
    "Query",
    "RagApplication",
    "RagContainer",
    "Shutdown",
    "Synchronous",
    "ThreadPool",
    "aggregate_status",
    "build_container",
    "document_by_id_helper",
    "list_all_records_helper",
    "parse_seed_users_json",
    "probe_embedder",
    "probe_vector_store",
    "raise_missing_document",
    "seed_blocked",
    "seed_demo_users",
    "upload_record_helper",
]
