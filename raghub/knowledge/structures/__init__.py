"""Knowledge-structure package (Phase 6).

Sub-package that ships the alternative retrieval structures the
planner can call: RAPTOR (recursive summaries) and GraphRAG
(entity / community graph). Both implement :class:`KnowledgeIndex`.
"""

from raghub.knowledge.structures.base import KnowledgeIndex
from raghub.knowledge.structures.graphrag import GraphRagIndex
from raghub.knowledge.structures.raptor import RaptorIndex

__all__ = ["GraphRagIndex", "KnowledgeIndex", "RaptorIndex"]