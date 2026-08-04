"""Service-layer helpers: logging, metrics, probes, seeding, and model wiring."""

from __future__ import annotations

import os
import time
from typing import Any

from raghub.config import Settings
from raghub.embedder import Embedder, build_embedder
from raghub.errors import IngestionError
from raghub.ingest import IngestionResult, Ingestor
from raghub.lifecycle import Lifecycle
from raghub.llm import Generator, build_llm
from raghub.models import Document
from raghub.parsers import Catalog
from raghub.prompts import Prompt
from raghub.retrieval import (
    Identity as IdentityReranker,
)
from raghub.retrieval import (
    Retrieval as RetrievalPipeline,
)
from raghub.store import Store
from raghub.stores import ImageStore
from raghub.telemetry import build_logger
from raghub.types import JSONValue


def emit_log(container: Any, message: str, **payload: JSONValue) -> None:
    """Emit a structured log event via the container's logger."""
    logger = getattr(container, "logger", None)
    log_method = getattr(logger, "info", None) if logger else None
    if callable(log_method):
        log_method(message, extra=payload)


def emit_metric(container: Any, name: str, started_at: float) -> None:
    """Record a latency metric given a ``perf_counter`` start time."""
    metrics = getattr(container, "metrics", None)
    recorder = getattr(metrics, "record_latency", None) if metrics else None
    if callable(recorder):
        recorder(name, (time.perf_counter() - started_at) * 1000.0)


def upload_record(result: IngestionResult | Any) -> Document:
    """Return the :class:`Document` carried by an ingestion result."""
    return result.document


def missing_doc(document_id: str) -> Document:
    """Raise :class:`IngestionError` for an unknown document id."""
    raise IngestionError(f"Unknown document id: {document_id}")


def probe_vector_store(store: object) -> dict[str, object]:
    """Probe a vector store for liveness.

    Calls the collaborator's ``health()`` method and translates the
    result into a canonical status.
    """
    probe = getattr(store, "health", None)
    if not callable(probe):
        return {"status": "unknown", "detail": "no health() method"}
    payload = probe()
    if not isinstance(payload, dict):
        payload = {"value": payload}
    status = str(payload.get("status", "ok")).lower()
    if status not in {"ok", "healthy", "up", "ready"}:
        payload = {**payload, "status": "degraded"}
    else:
        payload = {**payload, "status": "ok"}
    return payload


def probe_embedder(embedder: object) -> dict[str, object]:
    """Probe an embedding provider by emitting a tiny probe vector."""
    if embedder is None:
        return {"status": "unknown", "detail": "no embedder configured"}
    embed = getattr(embedder, "embed_text", None)
    if not callable(embed):
        return {"status": "unknown", "detail": "no embed_text() method"}
    vector = embed("health-check-probe")
    if not isinstance(vector, (list, tuple)) or hasattr(vector, "__aiter__"):
        return {
            "status": "ok",
            "dimension": None,
            "model": getattr(embedder, "model_name", ""),
        }
    dim = len(vector) if hasattr(vector, "__len__") else None
    if dim is None or dim == 0:
        return {"status": "down", "error": "empty embedding returned"}
    return {
        "status": "ok",
        "dimension": dim,
        "model": getattr(embedder, "model_name", ""),
    }


def aggregate_status(probes: dict[str, dict[str, object]]) -> str:
    """Combine per-component probes into a single status string."""
    statuses = [str(p.get("status", "")).lower() for p in probes.values()]
    if any(s == "down" for s in statuses):
        return "down"
    if any(s in {"degraded", "unknown"} for s in statuses):
        return "degraded"
    return "ok"


def seed_blocked(settings: Settings) -> bool:
    """Return ``True`` when the demo-user seed must be skipped."""
    if settings.environment == "production":
        return True
    return os.getenv("CORS_ORIGINS", "").strip() == "*"


def parse_users(raw: str) -> Any:
    """Parse the ``RAGHUB_USERS`` env var as JSON."""
    import json as json_import

    return json_import.loads(raw)


async def seed_demo_users(user_store: Any) -> None:
    """Seed demo users from ``RAGHUB_USERS`` or the default list."""
    users_env = os.getenv("RAGHUB_USERS", "").strip()
    if users_env:
        seed_users = parse_users(users_env)
        if isinstance(seed_users, dict):
            for email, cfg in seed_users.items():
                if not isinstance(cfg, dict):
                    continue
                existing = await user_store.get_by_email(email)
                if existing is not None:
                    continue
                await user_store.create_user(
                    email=email,
                    password=str(cfg.get("password", "password")),
                    companies=list(cfg.get("companies", []) or []),
                    is_admin=bool(cfg.get("is_admin", False)),
                )
        return

    default_seed = [
        ("alice@acme.com", "password", ["Apple"], False),
        ("bob@acme.com", "password", ["Microsoft"], False),
        ("charlie@acme.com", "password", ["Amazon", "Tesla"], False),
        ("diana@acme.com", "password", ["Google"], False),
        ("admin@acme.com", "password", [], True),
        ("alice@email.com", "test", ["Apple"], False),
        ("bob@email.com", "test", ["Microsoft", "Google"], False),
        ("charlie@email.com", "test", ["Amazon", "Tesla"], False),
        ("admin@email.com", "admin", [], True),
    ]
    for email, pwd, companies, is_admin in default_seed:
        existing = await user_store.get_by_email(email)
        if existing is not None:
            continue
        await user_store.create_user(
            email=email,
            password=pwd,
            companies=companies,
            is_admin=is_admin,
        )


def build_models(
    settings: Settings,
    vector_store: Store,
    uow: Any,
    nvidia_api_key: str,
) -> tuple[Any, ...]:
    """Build the LLM, embedding, retrieval, and document collaborators."""
    embeddings: Embedder = build_embedder(
        settings.embedding_model,
        settings.embedding_dim,
        nvidia_api_key,
    )
    llm: Generator = build_llm(settings.llm_model, nvidia_api_key)
    prompt_builder = Prompt()
    conversation = _build_conversation(uow)
    lifecycle = Lifecycle()
    ingestion = Ingestor(
        uow=uow,
        embedding_provider=embeddings,
        lifecycle_manager=lifecycle,
        max_upload_bytes=settings.max_upload_bytes,
    )
    retrieval = RetrievalPipeline(
        embedding_provider=embeddings,
        vector_store=vector_store,
        rerank=IdentityReranker(),
    )
    image_store = ImageStore(settings.data_dir / "images")
    parser_registry = Catalog()
    return (
        embeddings,
        llm,
        retrieval,
        ingestion,
        conversation,
        prompt_builder,
        image_store,
        parser_registry,
    )


def _build_conversation(uow: Any) -> Any:
    """Construct a :class:`ConversationHistory` bound to ``uow``."""
    from raghub.conv import ConversationHistory

    return ConversationHistory(uow)


__all__ = [
    "aggregate_status",
    "build_logger",
    "build_models",
    "emit_log",
    "emit_metric",
    "missing_doc",
    "parse_users",
    "probe_embedder",
    "probe_vector_store",
    "seed_blocked",
    "seed_demo_users",
    "upload_record",
]
