"""Runtime configuration for the platform.

The :class:`Settings` dataclass is the single source of truth for all
runtime configuration. Environment-variable loading is split into the
small ``load_*`` helpers in :mod:`raghub.config.loaders` and stitched
together by :func:`load_from_env` (also in :mod:`raghub.config.loaders`).
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from raghub.constants import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    MINISHLAB_POTION_MODEL,
)
from raghub.models import Secret, Snap

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


@dataclass(slots=True, frozen=True)
class AgentConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class WebSearchConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class RerankerConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class LongContextConfig(Snap):
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
    allowlist_models: list[str] = field(
        default_factory=lambda: [
            "claude-3-5-sonnet",
            "claude-3-7-sonnet",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "command-r-plus",
            "gpt-4.1",
        ]
    )


@dataclass(slots=True, frozen=True)
class HybridConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class QueryTransformsConfig(Snap):
    """Pre-retrieval query transforms (Phase 2).

    Attributes:
        enabled: Ordered list of transform names. Empty list = no
            transforms (default behaviour preserved).
        hyde_n: Number of hypothetical paragraphs to generate when
            ``"hyde"`` is enabled.
        multi_query_n: Number of rephrasings when ``"multi_query"`` is
            enabled.

    """

    enabled: list[Literal["hyde", "multi_query", "step_back", "decompose"]] = field(
        default_factory=list
    )
    hyde_n: int = 1
    multi_query_n: int = 4


@dataclass(slots=True, frozen=True)
class QueueConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class FeedbackConfig(Snap):
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


@dataclass(slots=True, frozen=True)
class RateLimitConfig(Snap):
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
    exempt_tenants: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ArchiveConfig(Snap):
    """Backup archive configuration.

    Attributes:
        backend: ``"none"`` or ``"local"`` (LocalArchiveStore).
        local_dir: Directory under which archives are written.

    """

    backend: Literal["local", "none"] = "none"
    local_dir: Path = field(default_factory=lambda: Path("./data/archives"))


@dataclass(slots=True, frozen=True)
class TenantsConfig(Snap):
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


def settings_field_names() -> frozenset[str]:
    """Return the field names of :class:`Settings` as a frozen set.

    Used by callers that need to filter user-provided dicts down to
    the canonical Settings surface (keeping unknowns on ``extra``).
    """
    return frozenset(f.name for f in dataclasses.fields(Settings))


@dataclass(slots=True, frozen=True)
class Settings(Snap):
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
        agent: Advanced RAG agent loop controls.
        web_search: Web search tool configuration.
        graph_search_enabled: Enable the graph retrieval path.
        summary_search_enabled: Enable the summary retrieval path.
        reranker: Cross-encoder / listwise reranker selection.
        long_context_pass: Long-context second-pass rerank controls.
        hybrid: Hybrid retrieval fusion settings.
        query_transforms: Pre-retrieval query transforms.
        queue: Persistent ingestion queue configuration.
        feedback: Feedback capture configuration.
        rate_limit: Per-tenant rate limiting configuration.
        archive: Backup archive configuration.
        tenants: Multi-tenant configuration.

    """

    environment: str = "development"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    registry_path: Path = field(default_factory=lambda: Path("./data/registry.json"))
    sessions_path: Path = field(default_factory=lambda: Path("./data/sessions.json"))
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
    jwt_secret: Secret = field(default_factory=lambda: Secret(""))
    nvidia_api_key: str = ""
    allow_passwordless_login: bool = False
    enable_query_cache: bool = False
    query_cache_ttl_seconds: int = 300
    extra: dict[str, Any] = field(default_factory=dict)

    agent: AgentConfig = field(default_factory=AgentConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    graph_search_enabled: bool = False
    summary_search_enabled: bool = False
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    long_context_pass: LongContextConfig = field(default_factory=LongContextConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    query_transforms: QueryTransformsConfig = field(default_factory=QueryTransformsConfig)

    queue: QueueConfig = field(default_factory=QueueConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    tenants: TenantsConfig = field(default_factory=TenantsConfig)

    def __post_init__(self) -> None:
        """Coerce raw strings and dict-shaped nested configs.

        Callers (settings factories, overrides, YAML loaders) often
        pass plain ``str`` for ``jwt_secret`` and ``dict`` for the
        nested config blocks (``agent``, ``reranker``, ...). This
        hook normalises both shapes so the dataclass-typed surface
        is preserved without forcing every call site to remember
        the wrapper types.
        """
        if isinstance(self.jwt_secret, str):
            object.__setattr__(self, "jwt_secret", Secret(self.jwt_secret))
        hints = typing.get_type_hints(Settings)
        for f in dataclasses.fields(self):
            if f.name == "extra":
                continue
            current = getattr(self, f.name)
            expected = hints.get(f.name, f.type)
            if isinstance(current, dict) and dataclasses.is_dataclass(expected):
                expected_cls: Any = cast(Any, expected)
                object.__setattr__(self, f.name, expected_cls(**current))

    def known_fields(self) -> dict[str, Any]:
        """Return a ``dict`` of all canonical fields (excluding ``extra``)."""
        return {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "extra"
        }

    def ensure_dirs(self) -> Settings:
        """Create the directories referenced by this :class:`Settings`.

        Returns:
            A new :class:`Settings` with every path field normalised
            to a :class:`Path` instance (the original is left intact
            because the dataclass is frozen).

        """
        data_dir = Path(self.data_dir)
        registry_path = Path(self.registry_path)
        sessions_path = Path(self.sessions_path)
        profile_path = Path(self.profile_path) if self.profile_path is not None else None
        data_dir.mkdir(parents=True, exist_ok=True)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        sessions_path.parent.mkdir(parents=True, exist_ok=True)
        return cast(
            "Settings",
            self.copy(
                data_dir=data_dir,
                registry_path=registry_path,
                sessions_path=sessions_path,
                profile_path=profile_path,
            ),
        )

    def override(self, **changes: Any) -> Settings:
        """Return a new :class:`Settings` with the given fields changed.

        Args:
            **changes: Field name → new value pairs. Unknown keys
                are kept on the ``extra`` mapping.

        Returns:
            A new instance; the receiver is not mutated.

        """
        known = self.known_fields()
        extra: dict[str, Any] = dict(self.extra)
        for key, value in changes.items():
            if key in known:
                known[key] = value
            else:
                extra[key] = value
        known["extra"] = extra
        return Settings(**known)

    @classmethod
    def load(cls, profile: str | None = None) -> Settings:
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


def production_check(settings: Settings) -> None:
    """Raise ``RuntimeError`` if production-mode invariants are violated."""
    if settings.environment != "production":
        return
    secret = settings.jwt_secret.value
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
