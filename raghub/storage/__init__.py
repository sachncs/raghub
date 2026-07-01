"""Durable storage adapters."""

from typing import TYPE_CHECKING, Any

from .image_store import FilesystemImageStore
from .sqlite_session_store import SqliteSessionStore

if TYPE_CHECKING:
    from .migration import migrate_from_json

__all__ = [
    "FilesystemImageStore",
    "migrate_from_json",
    "SqliteSessionStore",
]


def __getattr__(name: str) -> Any:
    """Lazily expose :func:`migrate_from_json` to avoid circular imports."""
    if name == "migrate_from_json":
        from .migration import migrate_from_json

        return migrate_from_json
    raise AttributeError(f"module 'raghub.storage' has no attribute {name!r}")
