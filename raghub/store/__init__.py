"""Vector-store implementations.

This package exposes the polymorphic :class:`Store` base, an
in-process ``MemoryStore`` (cosine + BM25), and a ``SqliteStore``
that uses the ``sqlite-vector`` package from
``github.com/sqliteai/sqlite-vector`` when installed, or a SQLite +
NumPy fallback when not.

The implementation is split across four files:

* :mod:`raghub.store.base` — the :class:`Store` Registry base, the
  :class:`MemoryVectorRecord` payload, and the ``matches_metadata_*``
  pre-filter helpers.
* :mod:`raghub.store.memory` — :class:`MemoryStore`.
* :mod:`raghub.store.sqlite` — :class:`SqliteStore` and the
  ``SQLITE_VECTOR_PKG`` backend probe.
* :mod:`raghub.store.factory` — :func:`build_store`.
"""

from raghub.store.base import (
    MemoryVectorRecord,
    Store,
    matches_metadata_dict,
    matches_metadata_string,
)
from raghub.store.factory import build_store
from raghub.store.memory import MemoryStore
from raghub.store.sqlite import SqliteStore

__all__ = [
    "MemoryStore",
    "MemoryVectorRecord",
    "SqliteStore",
    "Store",
    "build_store",
    "matches_metadata_dict",
    "matches_metadata_string",
]
