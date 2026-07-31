"""Retrieval-augmented generation framework.

The package exposes a high-level :class:`raghub.RAG` facade (the
spec entry point) plus :class:`raghub.services.Facade` and the
:func:`raghub.services.build_container` builder used by the
FastAPI surface. Both APIs are stable; new code should prefer
:class:`raghub.RAG`.

Public names are importable directly from their submodules —
the package re-exports its core entry points below so that
``from raghub import RAG`` (and ``import raghub`` + ``raghub.RAG``)
work without a separate ``raghub.rag`` import.
"""

from raghub.config import Settings
from raghub.errors import MissingDep, RagHubError

__all__ = ["RAG", "MissingDep", "RagHubError", "Settings"]


# ``RAG`` is the primary user-facing entry point; lazy-load it so the package
# loads without paying the full RAG stack on import.
def __getattr__(name: str) -> object:
    if name == "RAG":
        from raghub.rag import RAG

        return RAG
    raise AttributeError(f"module 'raghub' has no attribute {name!r}")

