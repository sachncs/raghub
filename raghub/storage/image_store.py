"""Content-addressable image storage on the local filesystem.

Images are stored by their SHA-256 content hash under
``<base_path>/<hash[:2]>/<hash><extension>``. The two-character prefix
subdirectory keeps any single directory from growing unboundedly (most
filesystems slow down past ~10k entries per dir).

The store is idempotent: re-saving the same bytes returns the existing
hash and does not re-write the file. This is the foundation for
content-deduplication across multimodal messages.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


class FilesystemImageStore:
    """Filesystem-backed image store keyed by content hash."""

    def __init__(self, base_path: str | Path = "./data/images") -> None:
        """Initialise the store.

        Args:
            base_path: Root directory for stored images. Created lazily
                on the first :meth:`save` call; not pre-created.
        """
        self.base_path = Path(base_path)

    def save(self, file_bytes: bytes, extension: str = ".png") -> str:
        """Persist ``file_bytes`` and return the content hash.

        Args:
            file_bytes: Raw image bytes.
            extension: File extension to append to the hash. Default
                ``.png``. No validation is performed; callers must
                supply a sensible extension for the content type.

        Returns:
            The hex SHA-256 content hash, also used as the file's stem.
        """
        content_hash = sha256(file_bytes).hexdigest()
        # Two-char prefix keeps the directory fan-out bounded.
        subdir = content_hash[:2]
        dest_dir = self.base_path / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{content_hash}{extension}"
        if not dest_path.exists():
            dest_path.write_bytes(file_bytes)
        return content_hash

    def get_path(self, content_hash: str, extension: str = ".png") -> Path | None:
        """Resolve a content hash to a filesystem path.

        Args:
            content_hash: The hex SHA-256 returned by :meth:`save`.
            extension: Extension to look up. Must match what was used
                in :meth:`save`.

        Returns:
            The :class:`Path` if the file exists, otherwise ``None``.
        """
        subdir = content_hash[:2]
        path = self.base_path / subdir / f"{content_hash}{extension}"
        if path.exists():
            return path
        return None

    def get_bytes(self, content_hash: str, extension: str = ".png") -> bytes | None:
        """Return the raw bytes for ``content_hash``.

        Args:
            content_hash: The hex SHA-256 returned by :meth:`save`.
            extension: Extension used when the file was stored.

        Returns:
            The file contents, or ``None`` if the file is missing.
        """
        path = self.get_path(content_hash, extension)
        if path is not None:
            return path.read_bytes()
        return None

    def delete(self, content_hash: str, extension: str = ".png") -> bool:
        """Delete the file for ``content_hash``.

        Args:
            content_hash: The hex SHA-256.
            extension: Extension used when the file was stored.

        Returns:
            ``True`` if a file was removed, ``False`` if no file existed.
        """
        path = self.get_path(content_hash, extension)
        if path is not None:
            path.unlink()
            return True
        return False
