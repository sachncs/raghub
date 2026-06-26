"""Configuration loading from YAML and environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(slots=True)
class AppSettings:
    """Runtime configuration for the platform."""

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
    retrieval_mode: str = "sync"
    log_level: str = "INFO"
    worker_backend: str = "threadpool"
    profile_path: Path | None = None
    require_zvec: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def ensure_dirs(self) -> None:
        """Create required directories."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)


def _merge_mapping(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    merged.update(source)
    return merged


def load_settings(profile: str | None = None) -> AppSettings:
    """Load settings from ``config/<profile>.yaml`` and environment variables."""

    base_dir = Path.cwd() / "config"
    selected_profile = profile or os.getenv("RAG_PROFILE", "development")
    profile_path = base_dir / f"{selected_profile}.yaml"
    payload: dict[str, Any] = {}
    if profile_path.exists():
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}

    env_payload: dict[str, Any] = {
        "environment": os.getenv("RAG_ENV", selected_profile),
        "data_dir": Path(os.getenv("RAG_DATA_DIR", payload.get("data_dir", "./data"))),
        "registry_path": Path(os.getenv("RAG_REGISTRY_PATH", payload.get("registry_path", "./data/registry.json"))),
        "sessions_path": Path(os.getenv("RAG_SESSIONS_PATH", payload.get("sessions_path", "./data/sessions.json"))),
        "zvec_dir": Path(os.getenv("RAG_ZVEC_DIR", payload.get("zvec_dir", "./data/zvec"))),
        "chunk_size_words": int(os.getenv("RAG_CHUNK_SIZE_WORDS", str(payload.get("chunk_size_words", 800)))),
        "chunk_overlap_words": int(os.getenv("RAG_CHUNK_OVERLAP_WORDS", str(payload.get("chunk_overlap_words", 100)))),
        "top_k": int(os.getenv("RAG_TOP_K", str(payload.get("top_k", 5)))),
        "embedding_dim": int(os.getenv("RAG_EMBEDDING_DIM", str(payload.get("embedding_dim", 384)))),
        "session_timeout_seconds": int(
            os.getenv("RAG_SESSION_TIMEOUT_SECONDS", str(payload.get("session_timeout_seconds", 3600)))
        ),
        "max_upload_bytes": int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(payload.get("max_upload_bytes", 20 * 1024 * 1024)))),
        "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", payload.get("embedding_model", "hashing-bge")),
        "llm_model": os.getenv("RAG_LLM_MODEL", payload.get("llm_model", "heuristic-llm")),
        "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", payload.get("retrieval_mode", "sync")),
        "log_level": os.getenv("RAG_LOG_LEVEL", payload.get("log_level", "INFO")),
        "worker_backend": os.getenv("RAG_WORKER_BACKEND", payload.get("worker_backend", "threadpool")),
        "require_zvec": os.getenv("RAG_REQUIRE_ZVEC", "").lower() in ("1", "true", "yes") or payload.get("require_zvec", False),
    }
    settings = AppSettings(
        **env_payload,
        profile_path=profile_path if profile_path.exists() else None,
        extra={k: v for k, v in payload.items() if k not in env_payload},
    )
    settings.ensure_dirs()
    return settings
