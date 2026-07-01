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
import sys
from pathlib import Path
from typing import Any

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

    def override(self, **changes: Any) -> "AppSettings":
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


def read_toml_file(path: Path) -> dict[str, Any]:
    """Load a TOML file using :mod:`tomllib` (3.11+) or :mod:`tomli`.

    Args:
        path: Path to the TOML file.

    Returns:
        The parsed dict, or ``{}`` if the file is empty. Missing
        optional dependencies are non-fatal: the caller logs a
        warning and falls back to YAML.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        # Neither tomllib nor tomli is available; return empty so
        # the YAML profile is used.
        return {}
    except Exception:
        return {}


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
            os.getenv("RAG_MAX_UPLOAD_BYTES", str(payload.get("max_upload_bytes", 20 * 1024 * 1024)))
        ),
        "embedding_model": os.getenv(
            "RAG_EMBEDDING_MODEL", payload.get("embedding_model", "hashing-bge")
        ),
        "llm_model": os.getenv("RAG_LLM_MODEL", payload.get("llm_model", "heuristic-llm")),
        "retrieval_mode": os.getenv(
            "RAG_RETRIEVAL_MODE", payload.get("retrieval_mode", "sync")
        ),
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
            raise RuntimeError(
                "JWT_SECRET environment variable is required in production mode"
            )
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
