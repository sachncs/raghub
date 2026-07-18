"""Document-version construction helper.

The only public entry point is :func:`new_version`, which clones the
previous version (when supplied), bumps the version number, resets the
status to ``NEW``, and applies any caller-supplied overrides.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from raghub.models import DocumentLifecycleStatus, DocumentVersion


def new_version(previous: DocumentVersion | None, **overrides: Any) -> DocumentVersion:
    """Build a new :class:`DocumentVersion` from a previous record.

    Algorithm:

    1. Start from the previous record's full ``model_dump`` (or an
       empty dict if there is no prior version).
    2. Apply the caller-supplied ``overrides``.
    3. Set ``version`` to ``previous.version + 1`` (or ``1`` if no
       prior version exists).
    4. Reset ``status`` to ``NEW`` and ``updated_at`` to ``now``.
    5. Carry over ``document_id`` and ``created_at`` from the prior
       record when not already overridden.

    Args:
        previous: The prior version, or ``None`` for a brand-new document.
        **overrides: Field overrides applied after the clone.

    Returns:
        A fully-typed :class:`DocumentVersion` ready for persistence.
    """
    version_number = 1 if previous is None else previous.version + 1
    payload = previous.model_dump() if previous else {}
    payload.update(overrides)
    payload["version"] = version_number
    # New versions always start in the ``NEW`` state; the ingestion
    # pipeline will drive them through the lifecycle.
    payload["status"] = DocumentLifecycleStatus.NEW
    payload["updated_at"] = datetime.now(UTC)
    if previous is not None:
        # ``setdefault`` keeps any caller overrides intact; the only
        # case these fire is when the caller did not already supply
        # the keys.
        payload.setdefault("document_id", previous.document_id)
        payload.setdefault("created_at", previous.created_at)
    return DocumentVersion.model_validate(payload)
