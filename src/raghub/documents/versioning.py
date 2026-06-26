"""Document versioning helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from raghub.models import DocumentVersion, DocumentLifecycleStatus


def new_version(previous: DocumentVersion | None, **overrides: Any) -> DocumentVersion:
    """Create a new document version.

    Args:
        previous: Previous version to clone.
        **overrides: Fields to override on the new version.

    Returns:
        The new document version.
    """

    version_number = 1 if previous is None else previous.version + 1
    payload = previous.model_dump() if previous else {}
    payload.update(overrides)
    payload["version"] = version_number
    payload["status"] = DocumentLifecycleStatus.NEW
    payload["updated_at"] = datetime.now(timezone.utc)
    if previous is not None:
        payload.setdefault("document_id", previous.document_id)
        payload.setdefault("created_at", previous.created_at)
    return DocumentVersion.model_validate(payload)
