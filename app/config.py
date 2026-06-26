"""Application configuration.

This module defines the runtime configuration used across the application.
It depends only on the standard library and Pydantic-friendly typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime configuration for the RAG application."""

    chunk_size: int = 800
    overlap: int = 100
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "minimaxai/minimax-m3"
    top_k: int = 5
    temperature: float = 0.1
    data_dir: Path = Path("database")
    documents_dir: Path = Path("documents")
    users_path: Path = Path("app/users.json")
    sqlite_path: Path = Path("database/rag.db")
    zvec_dir: Path = Path("database/zvec")
    max_upload_bytes: int = 20 * 1024 * 1024
    nvidia_api_key_env: str = "NVIDIA_API_KEY"

    def ensure_directories(self) -> None:
        """Create directories required by the application."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.zvec_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def load_config() -> AppConfig:
    """Load configuration from environment variables.

    Returns:
        AppConfig: Application configuration instance.
    """

    return AppConfig(
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        overlap=int(os.getenv("RAG_OVERLAP", "100")),
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("RAG_LLM_MODEL", "minimaxai/minimax-m3"),
        top_k=int(os.getenv("RAG_TOP_K", "5")),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
        data_dir=Path(os.getenv("RAG_DATA_DIR", "database")),
        documents_dir=Path(os.getenv("RAG_DOCUMENTS_DIR", "documents")),
        users_path=Path(os.getenv("RAG_USERS_PATH", "app/users.json")),
        sqlite_path=Path(os.getenv("RAG_SQLITE_PATH", "database/rag.db")),
        zvec_dir=Path(os.getenv("RAG_ZVEC_DIR", "database/zvec")),
        max_upload_bytes=int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        nvidia_api_key_env=os.getenv("RAG_NVIDIA_API_KEY_ENV", "NVIDIA_API_KEY"),
    )
