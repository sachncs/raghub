"""Knowledge representation layer.

Public surface re-exported from focused submodules:

* :mod:`raghub.knowledge.okf` — OKF serialisation, :func:`to_okf`,
  :func:`from_okf`, :func:`dumps`, :func:`loads`, and the
  in-memory :class:`MemoryRepo`.
* :mod:`raghub.knowledge.manifest` — the on-disk :class:`Manifest`
  and the :func:`sha256_bytes` byte-hash helper.
* :mod:`raghub.knowledge.raptor` — :class:`KnowledgeIndex` base,
  :class:`Raptor` recursive-summary tree, and the pure helpers
  used by it.
* :mod:`raghub.knowledge.graph` — :class:`GraphIndex` entity /
  community graph and the JSON / tokenisation helpers used by
  extraction and search.
"""

from __future__ import annotations

from raghub.knowledge.graph import (
    COMMUNITY_PROMPT,
    EXTRACT_PROMPT,
    MIN_TOKEN_LENGTH,
    GraphIndex,
    connected_components,
    extract_json_object,
    run_in_thread,
    running_loop_present,
    tokenise,
)
from raghub.knowledge.manifest import Manifest, sha256_bytes
from raghub.knowledge.okf import (
    OKF_SCHEMA_VERSION,
    MemoryRepo,
    dumps,
    from_okf,
    loads,
    to_okf,
)
from raghub.knowledge.raptor import (
    SUMMARY_PROMPT,
    KnowledgeIndex,
    Raptor,
    chunk_to_record,
    cluster,
    cosine_similarity,
    summarise,
    summarise_sync,
    summary_id_for,
)

__all__ = [
    "COMMUNITY_PROMPT",
    "EXTRACT_PROMPT",
    "MIN_TOKEN_LENGTH",
    "OKF_SCHEMA_VERSION",
    "SUMMARY_PROMPT",
    "GraphIndex",
    "KnowledgeIndex",
    "Manifest",
    "MemoryRepo",
    "Raptor",
    "chunk_to_record",
    "cluster",
    "connected_components",
    "cosine_similarity",
    "dumps",
    "extract_json_object",
    "from_okf",
    "loads",
    "run_in_thread",
    "running_loop_present",
    "sha256_bytes",
    "summarise",
    "summarise_sync",
    "summary_id_for",
    "to_okf",
    "tokenise",
]
