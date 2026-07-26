"""Legacy document domain model.

Deprecated in favour of the canonical models in
:mod:`raghub.models`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from raghub.models import DocumentLifecycleStatus, DocumentRecord


class Document:
    """Active-record wrapper around a :class:`DocumentRecord`.

    Attribute reads/writes forward to the wrapped record so callers
    can use the document as if it were the underlying Pydantic model.
    """

    def __init__(self, record: DocumentRecord) -> None:
        """Wrap ``record``."""
        self.record = record

    @property
    def document_id(self) -> str:
        """Return the document id from the wrapped record."""
        return self.record.document_id

    @property
    def status(self) -> DocumentLifecycleStatus:
        """Return the current lifecycle status."""
        return self.record.status

    @status.setter
    def status(self, value: DocumentLifecycleStatus) -> None:
        """Update the lifecycle status on the wrapped record."""
        self.record.status = value

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

    def update(self, **kwargs: Any) -> Document:
        """Bulk-set fields and bump ``updated_at``.

        Args:
            **kwargs: Field name/value pairs to assign.

        Returns:
            ``self`` for chaining.
        """
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        self.record.updated_at = datetime.now(UTC)
        return self

    def mark_failed(self, error: str) -> Document:
        """Mark the document as failed and record ``error``.

        Args:
            error: Human-readable failure description.

        Returns:
            ``self`` for chaining.
        """
        self.record.status = self.record.status.__class__.FAILED
        self.record.error = error
        self.record.updated_at = datetime.now(UTC)
        return self


__all__ = ["Document"]