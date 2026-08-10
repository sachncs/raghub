"""Pydantic configuration models.

Defines :class:`Settings` and the nested collaborator configuration
blocks (:class:`AgentConfig`, :class:`QueueConfig`, etc.) plus
:func:`production_check`, which enforces the production-mode
invariants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from raghub.constants import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    MINISHLAB_POTION_MODEL,
)

__all__ = [
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
    "WebSearchConfig",
]

JWT_SECRET_MIN_BYTES = 32


class Settings(BaseModel):
    """Runtime configuration for the platform.

    Attributes:
        environment: Profile name (``"development"``,
            ``"staging"``, ``"production"``).
        data_dir: Root directory for derived state (registry, sessions).
        registry_path: Path to the JSON-backed document registry.
        sessions_path: Path to the JSON-backed session store.
        chunk_size_words: Default chunk size used by the chunker.
        chunk_overlap_words: Default overlap used by the chunker.
        chunker_strategy: Chunking strategy name (``"recursive"``,
            ``"token"``, ``"sentence"``, ``"semantic"``, ``"late"``,
            ``"table"``, ``"code"``, ``"slumber"``, ``"neural"``,
            ``"auto"``).
        embedding_model_chunker: Embedding model for semantic/late
            chunkers (chonkie's built-in model name).
        top_k: Default top-k for retrieval.
        embedding_dim: Embedding dimensionality.
        session_timeout_seconds: Session inactivity timeout.
        max_upload_bytes: Maximum accepted upload size.
        embedding_model: Embedding model name (``"hashing-bge"``,
            ``"nvidia/..."``).
        llm_model: LLM model name.
        retrieval_mode: ``"sync"`` or ``"background"``.
        log_level: Minimum log level (``"INFO"``, ``"DEBUG"``, …).
        worker_backend: ``"threadpool"`` or ``"asyncio"``.
        profile_path: Path to the YAML profile that was loaded.
        jwt_secret: Secret used to sign JWTs. **Required in
            production.**
        nvidia_api_key: NVIDIA API key (only consumed by the NVIDIA
            providers).
        allow_passwordless_login: Development-only convenience for
            issuing sessions without a password. **Must be ``False``
            in production.**
        extra: Free-form config dict for settings not yet on the model.

    """

    environment: str = "development"
    data_dir: Path = Path("./data")
    registry_path: Path = Path("./data/registry.json")
    sessions_path: Path = Path("./data/sessions.json")
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS
    chunker_strategy: str = "recursive"
    embedding_model_chunker: str = MINISHLAB_POTION_MODEL
    top_k: int = 5
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    session_timeout_seconds: int = DEFAULT_SESSION_TIMEOUT_SECONDS
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    embedding_model: str = "hashing-bge"
    llm_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    profile_path: Path | None = None
    retrieval_mode: str = "sync"
    worker_backend: str = "threadpool"
    jwt_secret: SecretStr = SecretStr("")
    nvidia_api_key: str = ""
    allow_passwordless_login: bool = False
    enable_query_cache: bool = False
    query_cache_ttl_seconds: int = 300
    extra: dict[str, Any] = Field(default_factory=dict)

    # -- Advanced RAG (Phase 1.6) ------------------------------------------
    agent: AgentConfig = Field(default_factory=lambda: AgentConfig())
    web_search: WebSearchConfig = Field(default_factory=lambda: WebSearchConfig())
    graph_search_enabled: bool = False
    summary_search_enabled: bool = False
    reranker: RerankerConfig = Field(default_factory=lambda: RerankerConfig())
    long_context_pass: LongContextConfig = Field(default_factory=lambda: LongContextConfig())
    hybrid: HybridConfig = Field(default_factory=lambda: HybridConfig())
    query_transforms: QueryTransformsConfig = Field(default_factory=lambda: QueryTransformsConfig())

    # -- v0.9.0 Tier 1 collaborators ---------------------------------------
    queue: QueueConfig = Field(default_factory=lambda: QueueConfig())
    feedback: FeedbackConfig = Field(default_factory=lambda: FeedbackConfig())
    rate_limit: RateLimitConfig = Field(default_factory=lambda: RateLimitConfig())
    archive: ArchiveConfig = Field(default_factory=lambda: ArchiveConfig())
    tenants: TenantsConfig = Field(default_factory=lambda: TenantsConfig())

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    def ensure_dirs(self) -> None:
        """Create the directories referenced by the settings object."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.sessions_path).parent.mkdir(parents=True, exist_ok=True)
        # Coerce path-like fields to ``Path`` so YAML/TOML strings
        # round-trip cleanly.
        self.data_dir = Path(self.data_dir)
        self.registry_path = Path(self.registry_path)
        self.sessions_path = Path(self.sessions_path)
        if self.profile_path is not None:
            self.profile_path = Path(self.profile_path)

    def override(self, **changes: Any) -> Settings:
        """Return a new :class:`Settings` with the given fields changed.

        Args:
            **changes: Field name → new value pairs. Unknown keys
                are kept on the ``extra`` mapping.

        Returns:
            A new instance; the receiver is not mutated.

        """
        merged: dict[str, Any] = self.model_dump()
        extra: dict[str, Any] = dict(merged.get("extra", {}))
        for key, value in changes.items():
            if key in Settings.model_fields:
                merged[key] = value
            else:
                extra[key] = value
        merged["extra"] = extra
        return Settings(**merged)

    @classmethod
    def load(cls: type[Settings], profile: str | None = None) -> Settings:
        """Load from ``config/<profile>.yaml`` and environment variables.

        Args:
            profile: Optional profile name. ``None`` consults the
                ``RAG_PROFILE`` env var, then defaults to
                ``"development"``.

        Returns:
            The parsed :class:`Settings`.

        Raises:
            RuntimeError: When ``environment == "production"`` and the
                operator has not set ``JWT_SECRET`` or has left
                ``allow_passwordless_login`` enabled.

        """
        from raghub.config import load_from_env

        return load_from_env(profile)


# ---------------------------------------------------------------------------
# Advanced RAG configuration blocks (Phase 1.6)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Agent loop controls.

    Attributes:
        enabled: Master switch. ``False`` keeps the fast-path query
            pipeline untouched (Phase 10.6 regression test).
        max_steps: Hard cap on planner steps before raising
            :class:`AgentBudgetError`.
        max_tool_calls: Hard cap on total tool invocations per query.
        max_wall_seconds: Wall-clock cap per query.
        planner_model: Optional override for the planner LLM. ``None``
            falls back to :attr:`Settings.llm_model`.
        enable_streaming: When ``True``, :meth:`Agent.astream` yields
            :class:`PlannerEvent` instances instead of awaiting a final
            :class:`AgentTrace`.

    """

    enabled: bool = False
    max_steps: int = 8
    max_tool_calls: int = 10
    max_wall_seconds: float = 30.0
    planner_model: str | None = None
    enable_streaming: bool = True


class WebSearchConfig(BaseModel):
    """Web search tool configuration.

    Attributes:
        enabled: Register :class:`WebSearchTool` at startup.
        max_results: Default result count for each call.
        timeout_seconds: Network timeout per call.
        safe_search: ``"strict"`` / ``"moderate"`` / ``"off"``.

    """

    enabled: bool = False
    max_results: int = 5
    timeout_seconds: float = 10.0
    safe_search: Literal["strict", "moderate", "off"] = "moderate"


class RerankerConfig(BaseModel):
    """Cross-encoder / listwise reranker selection.

    Attributes:
        provider: ``"none"`` (identity) / ``"cohere"`` /
            ``"llm"`` / ``"cascade"``.
        top_k: Maximum number of hits the reranker is asked to score.
        cascade_threshold: For ``"cascade"`` — when the cheap reranker's
            top-N score spread is below this threshold the expensive
            reranker is invoked as well.

    """

    provider: Literal["none", "cohere", "llm", "cascade"] = "none"
    top_k: int = 20
    cascade_threshold: float = 0.05


class LongContextConfig(BaseModel):
    """Second-pass rerank using a long-context LLM.

    Attributes:
        enabled: Master switch.
        candidate_k: Number of post-rerank candidates fed to the
            long-context LLM.
        allowlist_models: Model names eligible for the second pass.
            Anything not in the list triggers a graceful no-op with a
            telemetry event (Phase 5.1 invariant).

    """

    enabled: bool = False
    candidate_k: int = 20
    allowlist_models: list[str] = [
        "claude-3-5-sonnet",
        "claude-3-7-sonnet",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "command-r-plus",
        "gpt-4.1",
    ]


class HybridConfig(BaseModel):
    """Hybrid retrieval fusion.

    Attributes:
        fusion: ``"rrf"`` (default) or ``"linear"`` (older weighted-sum).
        rrf_k: Standard RRF damping constant; 60 matches the literature.
        keyword_weight: Weight for the keyword channel when ``fusion == "linear"``.
        vector_weight: Weight for the dense channel when ``fusion == "linear"``.
        colbert_enabled: When ``True``, an optional ColBERT late-interaction
            channel is added to the hybrid retrieval (Phase 3.4).

    """

    fusion: Literal["rrf", "linear"] = "rrf"
    rrf_k: int = 60
    keyword_weight: float = 0.3
    vector_weight: float = 0.7
    colbert_enabled: bool = False


class QueryTransformsConfig(BaseModel):
    """Pre-retrieval query transforms (Phase 2).

    Attributes:
        enabled: Ordered list of transform names. Empty list = no
            transforms (default behaviour preserved).
        hyde_n: Number of hypothetical paragraphs to generate when
            ``"hyde"`` is enabled.
        multi_query_n: Number of rephrasings when ``"multi_query"`` is
            enabled.

    """

    enabled: list[Literal["hyde", "multi_query", "step_back", "decompose"]] = Field(
        default_factory=list
    )
    hyde_n: int = 1
    multi_query_n: int = 4


# ---------------------------------------------------------------------------
# v0.9.0 Tier 1 collaborator config blocks
# ---------------------------------------------------------------------------


class QueueConfig(BaseModel):
    """Persistent ingestion queue configuration.

    Attributes:
        backend: ``"memory"`` (no queue; legacy Resumable path) or
            ``"sqlite"`` (durable SqliteQueue backed by ``db_path``).
        db_path: SQLite file path; defaults to
            ``{data_dir}/queue.db``.
        max_inflight: Maximum pending+running jobs before
            :meth:`SqliteQueue.submit` raises
            :class:`QueueSaturatedError`.

    """

    backend: Literal["memory", "sqlite"] = "memory"
    db_path: Path | None = None
    max_inflight: int = 256


class FeedbackConfig(BaseModel):
    """Feedback capture configuration.

    Attributes:
        backend: ``"none"`` (no feedback store), ``"sqlite"``
            (SqliteFeedbackStore), or ``"postgres"``
            (PgFeedbackStore reusing the pgvector pool).
        db_path: SQLite file path; defaults to
            ``{data_dir}/feedback.db``.
        dsn: Postgres connection string when
            ``backend == "postgres"``.

    """

    backend: Literal["sqlite", "postgres", "none"] = "none"
    db_path: Path | None = None
    dsn: str | None = None


class RateLimitConfig(BaseModel):
    """Per-tenant rate limiting configuration.

    Attributes:
        backend: ``"memory"`` (process-local Bucket) or
            ``"sqlite"`` (durable backend).
        per_tenant_rps: Sustained refill rate per tenant.
        per_tenant_burst: Maximum bucket capacity per tenant.
        per_user_rps: Sustained refill rate per user.
        per_user_burst: Maximum bucket capacity per user.
        exempt_tenants: Tenants that bypass rate limits.

    """

    backend: Literal["memory", "sqlite"] = "memory"
    per_tenant_rps: float = 10.0
    per_tenant_burst: int = 20
    per_user_rps: float = 5.0
    per_user_burst: int = 10
    exempt_tenants: list[str] = Field(default_factory=list)


class ArchiveConfig(BaseModel):
    """Backup archive configuration.

    Attributes:
        backend: ``"none"`` or ``"local"`` (LocalArchiveStore).
        local_dir: Directory under which archives are written.

    """

    backend: Literal["local", "none"] = "none"
    local_dir: Path = Path("./data/archives")


class TenantsConfig(BaseModel):
    """Multi-tenant configuration.

    Attributes:
        resolver: TenantResolver implementation. ``"none"`` skips
            tenant resolution; ``"header"`` reads ``X-Tenant-ID``;
            ``"jwt"`` reads the JWT claim; ``"composite"`` prefers
            JWT and falls back to header.
        isolation: Isolation. ``"row_level"`` (default),
            ``"schema_per_tenant"``, or ``"database_per_tenant"``.

    """

    resolver: Literal["none", "header", "jwt", "composite"] = "none"
    isolation: Literal["row_level", "schema_per_tenant", "database_per_tenant"] = "row_level"


def production_check(settings: Settings) -> None:
    """Raise ``RuntimeError`` if production-mode invariants are violated."""
    if settings.environment != "production":
        return
    secret = settings.jwt_secret.get_secret_value()
    if not secret:
        raise RuntimeError("JWT_SECRET environment variable is required in production mode")
    if len(secret.encode("utf-8")) < JWT_SECRET_MIN_BYTES:
        raise RuntimeError(
            "JWT_SECRET must be at least 32 bytes long in production mode "
            "(PyJWT rejects shorter keys for HS256)."
        )
    if settings.allow_passwordless_login:
        raise RuntimeError(
            "Passwordless login is forbidden in production mode. "
            "Set RAG_ALLOW_PASSWORDLESS=0 or 'allow_passwordless_login: false' in config."
        )
