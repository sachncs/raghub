"""Configuration loading from YAML and environment variables.

Settings are loaded with the following precedence (highest wins):

1. Environment variables (``RAG_*`` / ``JWT_SECRET`` / ``NVIDIA_API_KEY``).
2. The YAML profile at ``config/<profile>.yaml``.
3. Built-in defaults declared on :class:`Settings`.

Production deployments must set ``JWT_SECRET`` and must disable
passwordless login; :meth:`Settings.load` raises :class:`RuntimeError`
when either invariant is violated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, Field, SecretStr

from raghub.constants import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    DEFAULT_TOP_K,
    GPT4O_MINI_MODEL,
    HASHING_BGE_MODEL,
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


# ---------------------------------------------------------------------------
# Env helpers (Phase 1.6)
# ---------------------------------------------------------------------------


TRUTHY = {"1", "true", "yes", "on"}


def env_bool(name: str, default: bool) -> bool:
    """Return the boolean value of ``os.getenv(name, ...)``.

    Treats ``"1"`` / ``"true"`` / ``"yes"`` / ``"on"`` (case-insensitive)
    as ``True``. Any other non-empty value is ``False``. Missing var
    falls back to ``default``.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in TRUTHY


TransformName = Literal["hyde", "multi_query", "step_back", "decompose"]
TRANSFORM_NAMES: tuple[TransformName, ...] = ("hyde", "multi_query", "step_back", "decompose")


def csv_to_transforms(raw: str, default: list[str]) -> list[TransformName]:
    """Parse a comma-separated env var into a validated transform list.

    Unknown names are dropped silently — config files are validated by
    Pydantic and raise on bad values; the env path is forgiving so a
    typo doesn't prevent startup.
    """
    if not raw:
        return cast(
            list[TransformName],
            [name for name in default if name in TRANSFORM_NAMES],
        )
    out: list[TransformName] = []
    for chunk in raw.split(","):
        name = chunk.strip().lower()
        if name and name in TRANSFORM_NAMES and name not in out:
            out.append(name)
    return out


def read_toml_file(path: Path) -> dict[str, Any]:
    """Load a TOML file using :mod:`tomllib` (3.11+).

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed dict, or ``{}`` if the file is empty.

    Raises:
        FileNotFoundError: When ``path`` doesn't exist.
        tomllib.TOMLDecodeError: When the TOML is malformed.
        OSError: When the file cannot be read.

    """
    import tomllib

    return tomllib.loads(path.read_text(encoding="utf-8")) or {}


def load_profile(profile: str | None) -> tuple[str | None, Path, dict[str, Any]]:
    """Read the YAML + TOML profile files into a single payload dict.

    Search order for the profile directory:

    1. ``RAG_CONFIG_DIR`` environment variable.
    2. ``./config`` relative to the current working directory.
    3. ``~/.config/raghub`` (XDG user config dir).
    4. The bundled ``config/`` shipped with the package.

    Returns:
        Tuple of (selected_profile, profile_path, payload). The
        ``profile_path`` is the YAML path searched; it may not exist.
        Missing files simply contribute an empty payload.

    """
    base_dir = __resolve()
    selected_profile = profile or os.getenv("RAG_PROFILE", "development")
    profile_path = base_dir / f"{selected_profile}.yaml"
    toml_path = base_dir / f"{selected_profile}.toml"
    payload: dict[str, Any] = {}
    if profile_path.exists():
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    if toml_path.exists():
        toml_payload = read_toml_file(toml_path)
        if toml_payload:
            # TOML takes precedence over YAML when both are present.
            payload = {**payload, **toml_payload}
    return selected_profile, profile_path, payload


def __resolve() -> Path:
    """Return the directory to search for profile YAML/TOML files.

    Resolution order:
    1. ``RAG_CONFIG_DIR`` env var (explicit override).
    2. ``./config`` (CWD-relative).
    3. ``~/.config/raghub`` (XDG-style user config).
    4. ``config/`` shipped with the package (read-only default).
    """
    env = os.getenv("RAG_CONFIG_DIR")
    if env:
        return Path(env)
    cwd_dir = Path.cwd() / "config"
    if cwd_dir.is_dir():
        return cwd_dir
    xdg_dir = Path.home() / ".config" / "raghub"
    if xdg_dir.is_dir():
        return xdg_dir
    try:
        from importlib.resources import files

        bundled = files("raghub").joinpath("config")
        if bundled.is_dir():
            return Path(str(bundled))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # Final fallback: return the CWD-relative path even if it
    # doesn't exist yet — the caller treats missing files as no-ops.
    return cwd_dir


def load_env(selected_profile: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the env-driven payload for the simple ``Settings`` fields.

    Each ``int()`` / ``float()`` coercion is wrapped in
    :func:`__int` / :func:`__float` so an invalid value (e.g.
    ``RAG_TOP_K="abc"``) raises a clear
    :class:`ConfigurationError` instead of ``ValueError: invalid
    literal for int()``.
    """
    return {
        "environment": os.getenv("RAG_ENV", selected_profile),
        "data_dir": Path(os.getenv("RAG_DATA_DIR", payload.get("data_dir", "./data"))),
        "registry_path": Path(
            os.getenv("RAG_REGISTRY_PATH", payload.get("registry_path", "./data/registry.json"))
        ),
        "sessions_path": Path(
            os.getenv("RAG_SESSIONS_PATH", payload.get("sessions_path", "./data/sessions.json"))
        ),
        "chunk_size_words": __int("RAG_CHUNK_SIZE_WORDS", payload.get("chunk_size_words", DEFAULT_CHUNK_SIZE_WORDS)),
        "chunk_overlap_words": __int(
            "RAG_CHUNK_OVERLAP_WORDS", payload.get("chunk_overlap_words", DEFAULT_CHUNK_OVERLAP_WORDS)
        ),
        "chunker_strategy": os.getenv(
            "RAG_CHUNKER_STRATEGY", payload.get("chunker_strategy", "recursive")
        ),
        "embedding_model_chunker": os.getenv(
            "RAG_EMBEDDING_MODEL_CHUNKER",
            payload.get("embedding_model_chunker", "minishlab/potion-base-8M"),
        ),
        "top_k": __int("RAG_TOP_K", payload.get("top_k", DEFAULT_TOP_K)),
        "embedding_dim": __int("RAG_EMBEDDING_DIM", payload.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
        "session_timeout_seconds": __int(
            "RAG_SESSION_TIMEOUT_SECONDS", payload.get("session_timeout_seconds", DEFAULT_SESSION_TIMEOUT_SECONDS)
        ),
        "max_upload_bytes": __int(
            "RAG_MAX_UPLOAD_BYTES", payload.get("max_upload_bytes", DEFAULT_MAX_UPLOAD_BYTES)
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
        "jwt_secret": SecretStr(os.getenv("JWT_SECRET", "")),
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", payload.get("nvidia_api_key", "")),
        "allow_passwordless_login": env_bool(
            "RAG_ALLOW_PASSWORDLESS",
            payload.get("allow_passwordless_login", False),
        ),
    }


def __int(name: str, default: int) -> int:
    """Read ``name`` from the environment as an int.

    Args:
        name: Environment variable name.
        default: Value when the env var is unset.

    Returns:
        Parsed integer, or ``default``.

    Raises:
        ConfigurationError: When the env var is set but not parseable
            as an int.

    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        from raghub.errors import ConfigurationError

        raise ConfigurationError(f"{name}={raw!r} is not a valid integer") from exc


def __float(name: str, default: float) -> float:
    """Read ``name`` from the environment as a float.

    Args:
        name: Environment variable name.
        default: Value when the env var is unset.

    Returns:
        Parsed float, or ``default``.

    Raises:
        ConfigurationError: When the env var is set but not parseable
            as a float.

    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        from raghub.errors import ConfigurationError

        raise ConfigurationError(f"{name}={raw!r} is not a valid float") from exc


def load_agent(payload: dict[str, Any]) -> AgentConfig:
    """Build :class:`AgentConfig` from env + payload."""
    return AgentConfig(
        enabled=env_bool("RAG_AGENT_ENABLED", payload.get("agent", {}).get("enabled", False)),
        max_steps=__int("RAG_AGENT_MAX_STEPS", payload.get("agent", {}).get("max_steps", 8)),
        max_tool_calls=__int(
            "RAG_AGENT_MAX_TOOL_CALLS", payload.get("agent", {}).get("max_tool_calls", 10)
        ),
        max_wall_seconds=__float(
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
            os.getenv("RAG_RERANKER_TOP_K", str(payload.get("reranker", {}).get("top_k", 20)))
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
        max_inflight=__int(
            "RAG_QUEUE_MAX_INFLIGHT", queue_payload.get("max_inflight", 256)
        ),
    )


def load_feedback(payload: dict[str, Any]) -> FeedbackConfig:
    """Build :class:`FeedbackConfig` from env + payload."""
    feedback_payload = payload.get("feedback", {})
    return FeedbackConfig(
        backend=cast(
            "Literal['sqlite', 'postgres', 'none']",
            os.getenv(
                "RAG_FEEDBACK_BACKEND", feedback_payload.get("backend", "none")
            ),
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
            os.getenv(
                "RAG_RATE_LIMIT_BACKEND", rate_limit_payload.get("backend", "memory")
            ),
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
            os.getenv(
                "RAG_TENANTS_RESOLVER", tenants_payload.get("resolver", "none")
            ),
        ),
        isolation=cast(
            "Literal['row_level', 'schema_per_tenant', 'database_per_tenant']",
            os.getenv(
                "RAG_TENANTS_ISOLATION",
                tenants_payload.get("isolation", "row_level"),
            ),
        ),
    )


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


def load_from_env(profile: str | None = None) -> Settings:
    """Read YAML/TOML profile + env vars, then return a configured :class:`Settings`.

    The function is the single orchestrator for config loading. The
    actual field-by-field reading is split into ``_load_*`` helpers
    below so each block stays under 80 lines.
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
