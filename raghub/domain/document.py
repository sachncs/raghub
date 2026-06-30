"""Legacy document domain model.

Deprecated in favour of the canonical models in
:mod:`raghub.models`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from raghub.models import DocumentLifecycleStatus, DocumentRecord


class Document:
    def __init__(self, record: DocumentRecord) -> None:
        self.record = record

    @property
    def document_id(self) -> str:
        return self.record.document_id

    @property
    def status(self) -> DocumentLifecycleStatus:
        return self.record.status

    @status.setter
    def status(self, value: DocumentLifecycleStatus) -> None:
        self.record.status = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.record, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("record",):
            super().__setattr__(name, value)
        else:
            setattr(self.record, name, value)

    def update(self, **kwargs: Any) -> Document:
        for key, value in kwargs.items():
            setattr(self.record, key, value)
        self.record.updated_at = datetime.now(timezone.utc)
        return self

    def mark_failed(self, error: str) -> Document:
        self.record.status = self.record.status.__class__.FAILED
        self.record.error = error
        self.record.updated_at = datetime.now(timezone.utc)
        return self
