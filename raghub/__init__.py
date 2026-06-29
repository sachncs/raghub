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
    from raghub.services.application import DynamicRagApplication, DynamicRagContainer
    from raghub.core.container import build_application


def __getattr__(name: str) -> Any:
    """Lazily import :class:`RAG` and the legacy builders."""
    if name == "RAG":
        from raghub.api.rag import RAG as _RAG

        return _RAG
    if name == "build_application":
        from raghub.core.container import build_application as _ba

        return _ba
    if name in {"DynamicRagApplication", "DynamicRagContainer"}:
        from raghub.services.application import (
            DynamicRagApplication as _app,
            DynamicRagContainer as _ctr,
        )

        return _app if name == "DynamicRagApplication" else _ctr
    raise AttributeError(f"module 'raghub' has no attribute {name!r}")


__all__ = [
    "DynamicRagApplication",
    "DynamicRagContainer",
    "RAG",
    "build_application",
]
