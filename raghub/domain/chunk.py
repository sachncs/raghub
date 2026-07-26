"""Legacy chunk domain model.

Deprecated in favour of :class:`raghub.models.Chunk` and
:class:`raghub.models.ChunkRecord`.
"""

from __future__ import annotations

from typing import Any

from raghub.models import ChunkRecord


class Chunk:
    """Active-record wrapper around a :class:`ChunkRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the chunk as if it were the underlying Pydantic model.
    """

    def __init__(self, record: ChunkRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def chunk_id(self) -> str:
        """Return the chunk id from the wrapped record."""
        return self.record.chunk_id

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attribute reads to the wrapped record."""
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute writes to the wrapped record.

        Only ``record`` itself is stored on the wrapper; everything
        else is set on the underlying Pydantic model.
        """
        if name in ("record",):
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def update(self, **kwargs: Any) -> Chunk:
        """Bulk-set fields on the wrapped record.

        Args:
            **kwargs: Field name/value pairs to assign.

        Returns:
            ``self`` for chaining.
        """
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        return self


__all__ = ["Chunk"]