"""Configuration loading from YAML and environment variables.

Settings are loaded with the following precedence (highest wins):

1. Environment variables (``RAG_*`` / ``JWT_SECRET`` / ``NVIDIA_API_KEY``).
2. The YAML profile at ``config/<profile>.yaml``.
3. Built-in defaults declared on :class:`AppSettings`.

Production deployments must set ``JWT_SECRET`` and must disable
passwordless login; :func:`load_settings` raises :class:`RuntimeError`
when either invariant is violated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr


class AppSettings(BaseModel):
    """Runtime configuration for the platform.

    Attributes:
        environment: Profile name (``"development"``,
            ``"staging"``, ``"production"``).
        data_dir: Root directory for derived state (registry, sessions).
        registry_path: Path to the JSON-backed document registry.
        sessions_path: Path to the JSON-backed session store.
        zvec_dir: Directory used by the zvec vector store.
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
            ``"nvidia/..."``, ``"sentence-transformers/..."``).
        llm_model: LLM model name.
        retrieval_mode: ``"sync"`` or ``"background"``.
        log_level: Minimum log level (``"INFO"``, ``"DEBUG"``, …).
        worker_backend: ``"threadpool"`` or ``"asyncio"``.
        profile_path: Path to the YAML profile that was loaded.
        require_zvec: Whether startup should fail when the zvec
            backend is unavailable.
        jwt_secret: Secret used to sign JWTs. **Required in
            production.**
        nvidia_api_key: NVIDIA API key (only consumed by the NVIDIA
            providers).
        allow_passwordless_login: Development-only convenience for
            issuing sessions without a password. **Must be ``False``
            in production.**
        extra: Free-form config dict for forward-compatible settings.
    """

    environment: str = "development"
    data_dir: Path = Path("./data")
    registry_path: Path = Path("./data/registry.json")
    sessions_path: Path = Path("./data/sessions.json")
    zvec_dir: Path = Path("./data/zvec")
    chunk_size_words: int = 800
    chunk_overlap_words: int = 100
    chunker_strategy: str = "recursive"
    embedding_model_chunker: str = "minishlab/potion-base-8M"
    top_k: int = 5
    embedding_dim: int = 384
    session_timeout_seconds: int = 3600
    max_upload_bytes: int = 20 * 1024 * 1024
    embedding_model: str = "hashing-bge"
    llm_model: str = "heuristic-llm"
    log_level: str = "INFO"
    profile_path: Path | None = None
    retrieval_mode: str = "sync"
    worker_backend: str = "threadpool"
    require_zvec: bool = False
    jwt_secret: SecretStr = SecretStr("")
    nvidia_api_key: str = ""
    allow_passwordless_login: bool = True
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
    query_transforms: QueryTransformsConfig = Field(
        default_factory=lambda: QueryTransformsConfig()
    )

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
        self.zvec_dir = Path(self.zvec_dir)
        if self.profile_path is not None:
            self.profile_path = Path(self.profile_path)

    def override(self, **changes: Any) -> AppSettings:
        """Return a new :class:`AppSettings` with the given fields changed.

        Args:
            **changes: Field name → new value pairs. Unknown keys
                are kept on the ``extra`` mapping.

        Returns:
            A new instance; the receiver is not mutated.
        """
        merged: dict[str, Any] = self.model_dump()
        extra: dict[str, Any] = dict(merged.get("extra", {}))
        for key, value in changes.items():
            if key in AppSettings.model_fields:
                merged[key] = value
            else:
                extra[key] = value
        merged["extra"] = extra
        return AppSettings(**merged)


# ---------------------------------------------------------------------------
# Advanced RAG configuration blocks (Phase 1.6)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Agent loop controls.

    Attributes:
        enabled: Master switch. ``False`` keeps the fast-path query
            pipeline untouched (Phase 10.6 regression test).
        max_steps: Hard cap on planner steps before raising
            :class:`AgentBudgetExceeded`.
        max_tool_calls: Hard cap on total tool invocations per query.
        max_wall_seconds: Wall-clock cap per query.
        planner_model: Optional override for the planner LLM. ``None``
            falls back to :attr:`AppSettings.llm_model`.
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
        provider: ``"none"`` (identity) / ``"cohere"`` / ``"bge"`` /
            ``"llm"`` / ``"cascade"``.
        top_k: Maximum number of hits the reranker is asked to score.
        cascade_threshold: For ``"cascade"`` — when the cheap reranker's
            top-N score spread is below this threshold the expensive
            reranker is invoked as well.
    """

    provider: Literal["none", "cohere", "bge", "llm", "cascade"] = "none"
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
        fusion: ``"rrf"`` (default) or ``"linear"`` (legacy).
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


TRANSFORM_NAMES = ("hyde", "multi_query", "step_back", "decompose")


def csv_to_transforms(raw: str, default: list[str]) -> list[str]:
    """Parse a comma-separated env var into a validated transform list.

    Unknown names are dropped silently — config files are validated by
    Pydantic and raise on bad values; the env path is forgiving so a
    typo doesn't prevent startup.
    """
    if not raw:
        return [name for name in default if name in TRANSFORM_NAMES]
    out: list[str] = []
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


def load_settings(profile: str | None = None) -> AppSettings:
    """Load settings from ``config/<profile>.yaml`` and environment variables.

    Args:
        profile: Optional profile name (``"development"``,
            ``"staging"``, ``"production"``). When ``None`` the
            ``RAG_PROFILE`` environment variable is consulted, then
            defaults to ``"development"``.

    Returns:
        The parsed :class:`AppSettings`.

    Raises:
        RuntimeError: When ``environment == "production"`` and the
            operator has not set ``JWT_SECRET`` or has left
            ``allow_passwordless_login`` enabled.
    """
    base_dir = Path.cwd() / "config"
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

    env_payload: dict[str, Any] = {
        "environment": os.getenv("RAG_ENV", selected_profile),
        "data_dir": Path(os.getenv("RAG_DATA_DIR", payload.get("data_dir", "./data"))),
        "registry_path": Path(
            os.getenv("RAG_REGISTRY_PATH", payload.get("registry_path", "./data/registry.json"))
        ),
        "sessions_path": Path(
            os.getenv("RAG_SESSIONS_PATH", payload.get("sessions_path", "./data/sessions.json"))
        ),
        "zvec_dir": Path(os.getenv("RAG_ZVEC_DIR", payload.get("zvec_dir", "./data/zvec"))),
        "chunk_size_words": int(
            os.getenv("RAG_CHUNK_SIZE_WORDS", str(payload.get("chunk_size_words", 800)))
        ),
        "chunk_overlap_words": int(
            os.getenv("RAG_CHUNK_OVERLAP_WORDS", str(payload.get("chunk_overlap_words", 100)))
        ),
        "chunker_strategy": os.getenv(
            "RAG_CHUNKER_STRATEGY", payload.get("chunker_strategy", "recursive")
        ),
        "embedding_model_chunker": os.getenv(
            "RAG_EMBEDDING_MODEL_CHUNKER",
            payload.get("embedding_model_chunker", "minishlab/potion-base-8M"),
        ),
        "top_k": int(os.getenv("RAG_TOP_K", str(payload.get("top_k", 5)))),
        "embedding_dim": int(
            os.getenv("RAG_EMBEDDING_DIM", str(payload.get("embedding_dim", 384)))
        ),
        "session_timeout_seconds": int(
            os.getenv(
                "RAG_SESSION_TIMEOUT_SECONDS",
                str(payload.get("session_timeout_seconds", 3600)),
            )
        ),
        "max_upload_bytes": int(
            os.getenv(
                "RAG_MAX_UPLOAD_BYTES", str(payload.get("max_upload_bytes", 20 * 1024 * 1024))
            )
        ),
        "embedding_model": os.getenv(
            "RAG_EMBEDDING_MODEL", payload.get("embedding_model", "hashing-bge")
        ),
        "llm_model": os.getenv("RAG_LLM_MODEL", payload.get("llm_model", "heuristic-llm")),
        "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", payload.get("retrieval_mode", "sync")),
        "log_level": os.getenv("RAG_LOG_LEVEL", payload.get("log_level", "INFO")),
        "worker_backend": os.getenv(
            "RAG_WORKER_BACKEND", payload.get("worker_backend", "threadpool")
        ),
        "require_zvec": os.getenv("RAG_REQUIRE_ZVEC", "").lower() in ("1", "true", "yes")
        or payload.get("require_zvec", False),
        "jwt_secret": SecretStr(os.getenv("JWT_SECRET", "")),
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", payload.get("nvidia_api_key", "")),
        "allow_passwordless_login": os.getenv("RAG_ALLOW_PASSWORDLESS", "").lower()
        in ("1", "true", "yes")
        or payload.get("allow_passwordless_login", True),
    }
    # Advanced RAG (Phase 1.6) — keep flat env vars; nested blocks are
    # built from them so config files can still express them as YAML.
    advanced_payload: dict[str, Any] = {
        "agent": AgentConfig(
            enabled=env_bool("RAG_AGENT_ENABLED", payload.get("agent", {}).get("enabled", False)),
            max_steps=int(
                os.getenv("RAG_AGENT_MAX_STEPS", str(payload.get("agent", {}).get("max_steps", 8)))
            ),
            max_tool_calls=int(
                os.getenv(
                    "RAG_AGENT_MAX_TOOL_CALLS",
                    str(payload.get("agent", {}).get("max_tool_calls", 10)),
                )
            ),
            max_wall_seconds=float(
                os.getenv(
                    "RAG_AGENT_MAX_WALL_SECONDS",
                    str(payload.get("agent", {}).get("max_wall_seconds", 30.0)),
                )
            ),
            planner_model=os.getenv(
                "RAG_AGENT_PLANNER_MODEL", payload.get("agent", {}).get("planner_model")
            )
            or None,
            enable_streaming=env_bool(
                "RAG_AGENT_STREAMING",
                payload.get("agent", {}).get("enable_streaming", True),
            ),
        ),
        "web_search": WebSearchConfig(
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
            safe_search=os.getenv(
                "RAG_WEB_SAFE_SEARCH",
                payload.get("web_search", {}).get("safe_search", "moderate"),
            ),
        ),
        "graph_search_enabled": env_bool(
            "RAG_GRAPH_ENABLED", payload.get("graph_search_enabled", False)
        ),
        "summary_search_enabled": env_bool(
            "RAG_SUMMARY_ENABLED", payload.get("summary_search_enabled", False)
        ),
        "reranker": RerankerConfig(
            provider=os.getenv(
                "RAG_RERANKER_PROVIDER",
                payload.get("reranker", {}).get("provider", "none"),
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
        ),
        "long_context_pass": LongContextConfig(
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
        ),
        "hybrid": HybridConfig(
            fusion=os.getenv(
                "RAG_HYBRID_FUSION",
                payload.get("hybrid", {}).get("fusion", "rrf"),
            ),
            rrf_k=int(
                os.getenv("RAG_HYBRID_RRF_K", str(payload.get("hybrid", {}).get("rrf_k", 60)))
            ),
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
        ),
        "query_transforms": QueryTransformsConfig(
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
        ),
    }
    env_payload.update(advanced_payload)
    settings = AppSettings(
        **env_payload,
        profile_path=profile_path if profile_path.exists() else None,
        extra={k: v for k, v in payload.items() if k not in env_payload},
    )
    if settings.environment == "production":
        # ``JWT_SECRET`` is mandatory in production: without it we
        # cannot sign or verify tokens, and the system would silently
        # accept forged credentials.
        secret = settings.jwt_secret.get_secret_value()
        if not secret:
            raise RuntimeError("JWT_SECRET environment variable is required in production mode")
        # ``JWT_SECRET`` must be at least 32 bytes for SHA-256
        # signing; PyJWT emits an InsecureKeyLengthWarning otherwise.
        if len(secret.encode("utf-8")) < 32:
            raise RuntimeError(
                "JWT_SECRET must be at least 32 bytes long in production mode "
                "(PyJWT rejects shorter keys for HS256)."
            )
        if settings.allow_passwordless_login:
            raise RuntimeError(
                "Passwordless login is forbidden in production mode. "
                "Set RAG_ALLOW_PASSWORDLESS=0 or 'allow_passwordless_login: false' in config."
            )
    settings.ensure_dirs()
    return settings


__all__ = [
    "AgentConfig",
    "AppSettings",
    "HybridConfig",
    "LongContextConfig",
    "QueryTransformsConfig",
    "RerankerConfig",
    "WebSearchConfig",
    "csv_to_transforms",
    "env_bool",
    "load_settings",
    "read_toml_file",
]
