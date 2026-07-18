"""Retrieval-augmented generation framework.

The package exposes a high-level :class:`RAG` facade (the spec entry
point) plus the legacy :class:`DynamicRagApplication` /
:func:`build_application` builders used by the FastAPI and Streamlit
surfaces. Both APIs are stable; new code should prefer
:class:`RAG`.

The :class:`RAG` facade is imported lazily so that the base package
can be imported without the optional web dependencies installed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raghub.api.rag import RAG as RAG
    from raghub.core.container import build_application
    from raghub.services.application import DynamicRagApplication, DynamicRagContainer


def __getattr__(name: str) -> Any:
    """Lazily import :class:`RAG` and the legacy builders."""
    if name == "RAG":
        from raghub.api.rag import RAG as rag_import

        return rag_import
    if name == "build_application":
        from raghub.core.container import build_application as ba_import

        return ba_import
    if name in {"DynamicRagApplication", "DynamicRagContainer"}:
        from raghub.services.application import (
            DynamicRagApplication as app_import,
        )
        from raghub.services.application import (
            DynamicRagContainer as ctr_import,
        )

        return app_import if name == "DynamicRagApplication" else ctr_import
    raise AttributeError(f"module 'raghub' has no attribute {name!r}")


__all__ = [
    "RAG",
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
]
