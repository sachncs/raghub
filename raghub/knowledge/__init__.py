"""Knowledge representation layer.

Holds the Open Knowledge Format (OKF) serialisation, an in-memory
repository adapter, the persistent source manifest, and the helpers
used by converters to normalise their output into a
:class:`raghub.models.KnowledgeBundle`.

Public re-exports:

* :func:`to_okf` / :func:`from_okf` / :func:`dumps` / :func:`loads` — OKF
  serialisation helpers.
* :class:`InMemoryKnowledgeRepository` — the default in-process
  repository.
* :class:`SourceManifest` — checksum-indexed source manifest.
"""

from raghub.knowledge.okf import dumps, from_okf, loads, to_okf
from raghub.knowledge.repository import InMemoryKnowledgeRepository
from raghub.knowledge.manifest import SourceManifest

__all__ = [
    "InMemoryKnowledgeRepository",
    "SourceManifest",
    "dumps",
    "from_okf",
    "loads",
    "to_okf",
]

