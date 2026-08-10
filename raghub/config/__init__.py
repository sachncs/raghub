"""Configuration loading from YAML and environment variables.

Settings are loaded with the following precedence (highest wins):

1. Environment variables (``RAG_*`` / ``JWT_SECRET`` / ``NVIDIA_API_KEY``).
2. The YAML profile at ``config/<profile>.yaml``.
3. Built-in defaults declared on :class:`Settings`.

Production deployments must set ``JWT_SECRET`` and must disable
passwordless login; :meth:`Settings.load` raises :class:`RuntimeError`
when either invariant is violated.

This package exposes the :class:`Settings` model (with its nested
collaborator configuration blocks), the :func:`load_from_env`
orchestrator, and the small env-parsing helpers (:func:`env_bool`,
:func:`csv_to_transforms`).
"""

from raghub.config.env import (
    TRANSFORM_NAMES,
    TRUTHY,
    TransformName,
    csv_to_transforms,
    env_bool,
    env_float,
    env_int,
    load_profile,
    read_toml_file,
    resolve_config_dir,
)
from raghub.config.loaders import (
    load_agent,
    load_archive,
    load_env,
    load_feedback,
    load_from_env,
    load_hybrid,
    load_int_settings,
    load_longcontext,
    load_path_settings,
    load_queue,
    load_rate_limit,
    load_reranker,
    load_security_settings,
    load_string_settings,
    load_tenants,
    load_transforms,
    load_web,
)
from raghub.config.settings import (
    JWT_SECRET_MIN_BYTES,
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
)

__all__ = [
    "JWT_SECRET_MIN_BYTES",
    "TRANSFORM_NAMES",
    "TRUTHY",
    "AgentConfig",
    "ArchiveConfig",
    "FeedbackConfig",
    "HybridConfig",
    "LongContextConfig",
    "QueryTransformsConfig",
    "QueueConfig",
    "RateLimitConfig",
    "RerankerConfig",
    "Settings",
    "TenantsConfig",
    "TransformName",
    "WebSearchConfig",
    "csv_to_transforms",
    "env_bool",
    "env_float",
    "env_int",
    "load_agent",
    "load_archive",
    "load_env",
    "load_feedback",
    "load_from_env",
    "load_hybrid",
    "load_int_settings",
    "load_longcontext",
    "load_path_settings",
    "load_profile",
    "load_queue",
    "load_rate_limit",
    "load_reranker",
    "load_security_settings",
    "load_string_settings",
    "load_tenants",
    "load_transforms",
    "load_web",
    "read_toml_file",
    "resolve_config_dir",
]
