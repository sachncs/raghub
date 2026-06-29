"""Knowledge repository contract.

Persistence/storage layer for :class:`KnowledgeBundle` objects. The
canonical persisted representation is Open Knowledge Format (OKF);
concrete adapters include the in-memory implementation used by tests
and SQLite/Postgres adapters added by callers.
"""

from __future__ import annotations

from typing import Protocol

from raghub.models import KnowledgeBundle


class KnowledgeRepository(Protocol):
    """Persists and retrieves :class:`KnowledgeBundle` objects."""

    def save(self, bundle: KnowledgeBundle) -> KnowledgeBundle:
        """Persist ``bundle`` and return the stored record.

        Args:
            bundle: The bundle to store.

        Returns:
            The persisted bundle. May differ from the input by
            populated ``bundle_id`` or ``created_at``.
        """

    def get(self, bundle_id: str) -> KnowledgeBundle | None:
        """Look up a bundle by id.

        Args:
            bundle_id: The bundle id.

        Returns:
            The bundle, or ``None`` if not found.
        """

    def list_by_source(self, source_uri: str) -> list[KnowledgeBundle]:
        """Return all bundles derived from ``source_uri``.

        Args:
            source_uri: The source identifier.

        Returns:
            Every persisted bundle for the source, newest first.
        """

    def delete(self, bundle_id: str) -> None:
        """Remove a bundle.

        Args:
            bundle_id: The bundle id.
        """
