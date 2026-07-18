"""In-memory knowledge repository adapter.

Useful for tests and simple deployments. Production deployments swap
in a SQLite/Postgres/Redis-backed repository without changing the
public API.
"""

from __future__ import annotations

from raghub.interfaces.knowledge import KnowledgeRepository
from raghub.models import KnowledgeBundle


class InMemoryKnowledgeRepository(KnowledgeRepository):
    """Threadsafe-ish :class:`KnowledgeRepository` for tests and dev.

    The implementation keeps every bundle in a single in-process dict.
    Concurrent reads are safe; concurrent writes may race but the
    repository is intended for single-writer deployments and tests
    where race-freedom is the caller's responsibility.
    """

    def __init__(self) -> None:
        """Initialise the empty in-memory store."""
        self.bundles: dict[str, KnowledgeBundle] = {}
        self.by_source: dict[str, list[str]] = {}

    def save(self, bundle: KnowledgeBundle) -> KnowledgeBundle:
        """Persist ``bundle`` in memory."""
        self.bundles[bundle.bundle_id] = bundle
        self.by_source.setdefault(bundle.source_uri, []).insert(0, bundle.bundle_id)
        return bundle

    def get(self, bundle_id: str) -> KnowledgeBundle | None:
        """Return the bundle with id ``bundle_id`` or ``None``."""
        return self.bundles.get(bundle_id)

    def list_by_source(self, source_uri: str) -> list[KnowledgeBundle]:
        """Return every bundle for ``source_uri`` (newest first)."""
        return [
            self.bundles[bid] for bid in self.by_source.get(source_uri, []) if bid in self.bundles
        ]

    def delete(self, bundle_id: str) -> None:
        """Remove the bundle; missing ids are ignored."""
        bundle = self.bundles.pop(bundle_id, None)
        if bundle is not None:
            ids = self.by_source.get(bundle.source_uri, [])
            if bundle_id in ids:
                ids.remove(bundle_id)


__all__ = ["InMemoryKnowledgeRepository"]
