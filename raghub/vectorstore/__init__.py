"""Vector-store implementations for dense, hybrid, and metadata-filtered search.

The package exposes the abstract contract together with the in-memory,
Qdrant, and Zvec adapters.
"""

from .base import BaseVectorStore
from .memory import InMemoryVectorStore
from .qdrant import QdrantVectorStore
from .zvec import ZvecVectorStore

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "ZvecVectorStore",
]
