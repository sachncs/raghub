"""Settings field loaders.

Each :func:`load_*` function reads a coherent slice of the config
payload (from env vars + the YAML/TOML profile dict) and returns either
a partial ``Settings`` kwargs mapping or a nested configuration block.
:func:`load_from_env` orchestrates them into a complete
:class:`Settings`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import SecretStr

from raghub.config.env import (
    csv_to_transforms,
    env_bool,
    env_float,
    env_int,
    load_profile,
)
from raghub.config.settings import (
    AgentConfig,
    ArchiveConfig,
    FeedbackConfig,
    HybridConfig,
    LongContextConfig,
    QueryTransformsConfig,
    QueueConfig,
    RateLimitConfig,
    RerankerConfig,
    Settings,
    TenantsConfig,
    WebSearchConfig,
    production_check,
)
from raghub.constants import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    DEFAULT_TOP_K,
    ENV_JWT_SECRET,
    ENV_NVIDIA_API_KEY,
    ENV_RAG_ALLOW_PASSWORDLESS,
    ENV_RAG_RERANKER_TOP_K,
    GPT4O_MINI_MODEL,
    HASHING_BGE_MODEL,
)

__all__ = ["load_from_env"]


def load_env(selected_profile: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the env-driven payload for the simple ``Settings`` fields.

    Each ``int()`` / ``float()`` coercion is wrapped in
    :func:`env_int` / :func:`env_float` so an invalid value (e.g.
    ``RAG_TOP_K="abc"``) raises a clear
    :class:`ConfigurationError` instead of ``ValueError: invalid
    literal for int()``.
    """
    return {
        **load_path_settings(selected_profile, payload),
        **load_int_settings(payload),
        **load_string_settings(payload),
        **load_security_settings(payload),
    }


def load_path_settings(selected_profile: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build path-typed Settings (environment, data_dir, registry, sessions)."""
    return {
        "environment": os.getenv("RAG_ENV", selected_profile),
        "data_dir": Path(os.getenv("RAG_DATA_DIR", payload.get("data_dir", "./data"))),
        "registry_path": Path(
            os.getenv("RAG_REGISTRY_PATH", payload.get("registry_path", "./data/registry.json"))
        ),
        "sessions_path": Path(
            os.getenv("RAG_SESSIONS_PATH", payload.get("sessions_path", "./data/sessions.json"))
        ),
    }


def load_int_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Build int-typed Settings (chunk sizes, top_k, embedding_dim, etc)."""
    return {
        "chunk_size_words": env_int(
            "RAG_CHUNK_SIZE_WORDS", payload.get("chunk_size_words", DEFAULT_CHUNK_SIZE_WORDS)
        ),
        "chunk_overlap_words": env_int(
            "RAG_CHUNK_OVERLAP_WORDS",
            payload.get("chunk_overlap_words", DEFAULT_CHUNK_OVERLAP_WORDS),
        ),
        "top_k": env_int("RAG_TOP_K", payload.get("top_k", DEFAULT_TOP_K)),
        "embedding_dim": env_int(
            "RAG_EMBEDDING_DIM", payload.get("embedding_dim", DEFAULT_EMBEDDING_DIM)
        ),
        "session_timeout_seconds": env_int(
            "RAG_SESSION_TIMEOUT_SECONDS",
            payload.get("session_timeout_seconds", DEFAULT_SESSION_TIMEOUT_SECONDS),
        ),
        "max_upload_bytes": env_int(
            "RAG_MAX_UPLOAD_BYTES", payload.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES)
        ),
    }


def load_string_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Build string-typed Settings (chunkers, models, log level)."""
    return {
        "chunker_strategy": os.getenv(
            "RAG_CHUNKER_STRATEGY", payload.get("chunker_strategy", "recursive")
        ),
        "embedding_model_chunker": os.getenv(
            "RAG_EMBEDDING_MODEL_CHUNKER",
            payload.get("embedding_model_chunker", "minishlab/potion-base-8M"),
        ),
        "embedding_model": os.getenv(
            "RAG_EMBEDDING_MODEL", payload.get("embedding_model", HASHING_BGE_MODEL)
        ),
        "llm_model": os.getenv("RAG_LLM_MODEL", payload.get("llm_model", GPT4O_MINI_MODEL)),
        "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", payload.get("retrieval_mode", "sync")),
        "log_level": os.getenv("RAG_LOG_LEVEL", payload.get("log_level", "INFO")),
        "worker_backend": os.getenv(
            "RAG_WORKER_BACKEND", payload.get("worker_backend", "threadpool")
        ),
    }


def load_security_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Build security-typed Settings (jwt_secret, nvidia_api_key, allow_passwordless)."""
    return {
        "jwt_secret": SecretStr(os.getenv(ENV_JWT_SECRET, "")),
        "nvidia_api_key": os.getenv(ENV_NVIDIA_API_KEY, payload.get("nvidia_api_key", "")),
        "allow_passwordless_login": env_bool(
            ENV_RAG_ALLOW_PASSWORDLESS,
            payload.get("allow_passwordless_login", False),
        ),
    }


def load_agent(payload: dict[str, Any]) -> AgentConfig:
    """Build :class:`AgentConfig` from env + payload."""
    return AgentConfig(
        enabled=env_bool("RAG_AGENT_ENABLED", payload.get("agent", {}).get("enabled", False)),
        max_steps=env_int("RAG_AGENT_MAX_STEPS", payload.get("agent", {}).get("max_steps", 8)),
        max_tool_calls=env_int(
            "RAG_AGENT_MAX_TOOL_CALLS", payload.get("agent", {}).get("max_tool_calls", 10)
        ),
        max_wall_seconds=env_float(
            "RAG_AGENT_MAX_WALL_SECONDS", payload.get("agent", {}).get("max_wall_seconds", 30.0)
        ),
        planner_model=os.getenv(
            "RAG_AGENT_PLANNER_MODEL", payload.get("agent", {}).get("planner_model")
        )
        or None,
        enable_streaming=env_bool(
            "RAG_AGENT_STREAMING",
            payload.get("agent", {}).get("enable_streaming", True),
        ),
    )


def load_web(payload: dict[str, Any]) -> WebSearchConfig:
    """Build :class:`WebSearchConfig` from env + payload."""
    return WebSearchConfig(
        enabled=env_bool("RAG_WEB_ENABLED", payload.get("web_search", {}).get("enabled", False)),
        max_results=int(
            os.getenv(
                "RAG_WEB_MAX_RESULTS",
                str(payload.get("web_search", {}).get("max_results", 5)),
            )
        ),
        timeout_seconds=float(
            os.getenv(
                "RAG_WEB_TIMEOUT_SECONDS",
                str(payload.get("web_search", {}).get("timeout_seconds", 10.0)),
            )
        ),
        safe_search=cast(
            "Literal['strict', 'moderate', 'off']",
            os.getenv(
                "RAG_WEB_SAFE_SEARCH",
                payload.get("web_search", {}).get("safe_search", "moderate"),
            ),
        ),
    )


def load_reranker(payload: dict[str, Any]) -> RerankerConfig:
    """Build :class:`RerankerConfig` from env + payload."""
    return RerankerConfig(
        provider=cast(
            "Literal['none', 'cohere', 'llm', 'cascade']",
            os.getenv(
                "RAG_RERANKER_PROVIDER",
                payload.get("reranker", {}).get("provider", "none"),
            ),
        ),
        top_k=int(
            os.getenv(ENV_RAG_RERANKER_TOP_K, str(payload.get("reranker", {}).get("top_k", 20)))
        ),
        cascade_threshold=float(
            os.getenv(
                "RAG_RERANKER_CASCADE_THRESHOLD",
                str(payload.get("reranker", {}).get("cascade_threshold", 0.05)),
            )
        ),
    )


def load_longcontext(payload: dict[str, Any]) -> LongContextConfig:
    """Build :class:`LongContextConfig` from env + payload."""
    return LongContextConfig(
        enabled=env_bool(
            "RAG_LONG_CONTEXT_ENABLED",
            payload.get("long_context_pass", {}).get("enabled", False),
        ),
        candidate_k=int(
            os.getenv(
                "RAG_LONG_CONTEXT_CANDIDATE_K",
                str(payload.get("long_context_pass", {}).get("candidate_k", 20)),
            )
        ),
    )


def load_hybrid(payload: dict[str, Any]) -> HybridConfig:
    """Build :class:`HybridConfig` from env + payload."""
    return HybridConfig(
        fusion=cast(
            "Literal['rrf', 'linear']",
            os.getenv(
                "RAG_HYBRID_FUSION",
                payload.get("hybrid", {}).get("fusion", "rrf"),
            ),
        ),
        rrf_k=int(os.getenv("RAG_HYBRID_RRF_K", str(payload.get("hybrid", {}).get("rrf_k", 60)))),
        keyword_weight=float(
            os.getenv(
                "RAG_HYBRID_KEYWORD_WEIGHT",
                str(payload.get("hybrid", {}).get("keyword_weight", 0.3)),
            )
        ),
        vector_weight=float(
            os.getenv(
                "RAG_HYBRID_VECTOR_WEIGHT",
                str(payload.get("hybrid", {}).get("vector_weight", 0.7)),
            )
        ),
        colbert_enabled=env_bool(
            "RAG_HYBRID_COLBERT",
            payload.get("hybrid", {}).get("colbert_enabled", False),
        ),
    )


def load_transforms(payload: dict[str, Any]) -> QueryTransformsConfig:
    """Build :class:`QueryTransformsConfig` from env + payload."""
    return QueryTransformsConfig(
        enabled=csv_to_transforms(
            os.getenv("RAG_TRANSFORMS_ENABLED", ""),
            payload.get("query_transforms", {}).get("enabled", []),
        ),
        hyde_n=int(
            os.getenv(
                "RAG_TRANSFORMS_HYDE_N",
                str(payload.get("query_transforms", {}).get("hyde_n", 1)),
            )
        ),
        multi_query_n=int(
            os.getenv(
                "RAG_TRANSFORMS_MULTI_QUERY_N",
                str(payload.get("query_transforms", {}).get("multi_query_n", 4)),
            )
        ),
    )


def load_queue(payload: dict[str, Any]) -> QueueConfig:
    """Build :class:`QueueConfig` from env + payload."""
    queue_payload = payload.get("queue", {})
    return QueueConfig(
        backend=cast(
            "Literal['memory', 'sqlite']",
            os.getenv("RAG_QUEUE_BACKEND", queue_payload.get("backend", "memory")),
        ),
        max_inflight=env_int("RAG_QUEUE_MAX_INFLIGHT", queue_payload.get("max_inflight", 256)),
    )


def load_feedback(payload: dict[str, Any]) -> FeedbackConfig:
    """Build :class:`FeedbackConfig` from env + payload."""
    feedback_payload = payload.get("feedback", {})
    return FeedbackConfig(
        backend=cast(
            "Literal['sqlite', 'postgres', 'none']",
            os.getenv("RAG_FEEDBACK_BACKEND", feedback_payload.get("backend", "none")),
        ),
        dsn=os.getenv("RAG_FEEDBACK_DSN") or feedback_payload.get("dsn"),
    )


def load_rate_limit(payload: dict[str, Any]) -> RateLimitConfig:
    """Build :class:`RateLimitConfig` from env + payload."""
    rate_limit_payload = payload.get("rate_limit", {})
    exempt_raw = os.getenv("RAG_RATE_LIMIT_EXEMPT_TENANTS", "")
    if exempt_raw:
        exempt = [t.strip() for t in exempt_raw.split(",") if t.strip()]
    else:
        exempt = list(rate_limit_payload.get("exempt_tenants", []))
    return RateLimitConfig(
        backend=cast(
            "Literal['memory', 'sqlite']",
            os.getenv("RAG_RATE_LIMIT_BACKEND", rate_limit_payload.get("backend", "memory")),
        ),
        per_tenant_rps=float(
            os.getenv(
                "RAG_RATE_LIMIT_RPS",
                str(rate_limit_payload.get("per_tenant_rps", 10.0)),
            )
        ),
        per_tenant_burst=int(
            os.getenv(
                "RAG_RATE_LIMIT_BURST",
                str(rate_limit_payload.get("per_tenant_burst", 20)),
            )
        ),
        per_user_rps=float(
            os.getenv(
                "RAG_RATE_LIMIT_USER_RPS",
                str(rate_limit_payload.get("per_user_rps", 5.0)),
            )
        ),
        per_user_burst=int(
            os.getenv(
                "RAG_RATE_LIMIT_USER_BURST",
                str(rate_limit_payload.get("per_user_burst", 10)),
            )
        ),
        exempt_tenants=exempt,
    )


def load_archive(payload: dict[str, Any]) -> ArchiveConfig:
    """Build :class:`ArchiveConfig` from env + payload."""
    archive_payload = payload.get("archive", {})
    local_dir_raw = os.getenv("RAG_ARCHIVE_DIR") or archive_payload.get(
        "local_dir", "./data/archives"
    )
    return ArchiveConfig(
        backend=cast(
            "Literal['local', 'none']",
            os.getenv("RAG_ARCHIVE_BACKEND", archive_payload.get("backend", "none")),
        ),
        local_dir=Path(local_dir_raw),
    )


def load_tenants(payload: dict[str, Any]) -> TenantsConfig:
    """Build :class:`TenantsConfig` from env + payload."""
    tenants_payload = payload.get("tenants", {})
    return TenantsConfig(
        resolver=cast(
            "Literal['none', 'header', 'jwt', 'composite']",
            os.getenv("RAG_TENANTS_RESOLVER", tenants_payload.get("resolver", "none")),
        ),
        isolation=cast(
            "Literal['row_level', 'schema_per_tenant', 'database_per_tenant']",
            os.getenv(
                "RAG_TENANTS_ISOLATION",
                tenants_payload.get("isolation", "row_level"),
            ),
        ),
    )


def load_from_env(profile: str | None = None) -> Settings:
    """Read YAML/TOML profile + env vars, then return a configured :class:`Settings`.

    The function is the single orchestrator for config loading. The
    actual field-by-field reading is split into ``load_*`` helpers
    below so each block stays small.
    """
    selected_profile, profile_path, payload = load_profile(profile)
    if selected_profile is None:
        selected_profile = "development"
    env_payload = load_env(selected_profile, payload)
    env_payload["agent"] = load_agent(payload)
    env_payload["web_search"] = load_web(payload)
    env_payload["reranker"] = load_reranker(payload)
    env_payload["long_context_pass"] = load_longcontext(payload)
    env_payload["hybrid"] = load_hybrid(payload)
    env_payload["query_transforms"] = load_transforms(payload)
    env_payload["graph_search_enabled"] = env_bool(
        "RAG_GRAPH_ENABLED", payload.get("graph_search_enabled", False)
    )
    env_payload["summary_search_enabled"] = env_bool(
        "RAG_SUMMARY_ENABLED", payload.get("summary_search_enabled", False)
    )
    env_payload["queue"] = load_queue(payload)
    env_payload["feedback"] = load_feedback(payload)
    env_payload["rate_limit"] = load_rate_limit(payload)
    env_payload["archive"] = load_archive(payload)
    env_payload["tenants"] = load_tenants(payload)
    settings = Settings(
        **env_payload,
        profile_path=profile_path if profile_path.exists() else None,
        extra={k: v for k, v in payload.items() if k not in env_payload},
    )
    production_check(settings)
    settings.ensure_dirs()
    return settings
