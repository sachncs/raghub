"""Content-addressable image storage on the local filesystem.

Images are stored by their SHA-256 content hash under
``<base_path>/<hash[:2]>/<hash><extension>``. The two-character prefix
subdirectory keeps any single directory from growing unboundedly.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

__all__ = ["ImageStore"]


class ImageStore:
    """Content-addressable image storage on the local filesystem.

    Images are stored by their SHA-256 content hash under
    ``<base_path>/<hash[:2]>/<hash><extension>``. The two-character prefix
    subdirectory keeps any single directory from growing unboundedly.
    """

    def __init__(self, base_path: str | Path = "./data/images") -> None:
        """Store the root directory; created lazily on first :meth:`save`."""
        self.base_path = Path(base_path)

    def save(self, file_bytes: bytes, extension: str = ".png") -> str:
        """Persist ``file_bytes`` and return the content hash."""
        content_hash = sha256(file_bytes).hexdigest()
        subdir = content_hash[:2]
        dest_dir = self.base_path / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{content_hash}{extension}"
        if not dest_path.exists():
            dest_path.write_bytes(file_bytes)
        return content_hash

    def get_path(self, content_hash: str, extension: str = ".png") -> Path | None:
        """Resolve a content hash to its filesystem path, or ``None``."""
        path = self.base_path / content_hash[:2] / f"{content_hash}{extension}"
        return path if path.exists() else None

    def get_bytes(self, content_hash: str, extension: str = ".png") -> bytes | None:
        """Return the raw bytes for ``content_hash``, or ``None``."""
        path = self.get_path(content_hash, extension)
        return path.read_bytes() if path is not None else None

    def delete(self, content_hash: str, extension: str = ".png") -> bool:
        """Delete the file for ``content_hash``. Returns whether a file was removed."""
        path = self.get_path(content_hash, extension)
        if path is not None:
            path.unlink()
            return True
        return False
