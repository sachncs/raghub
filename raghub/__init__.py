"""Retrieval-augmented generation framework.

The package exposes a high-level :class:`raghub.RAG` facade (the
spec entry point) plus the legacy :class:`raghub.services.application.RagApplication`
and :func:`raghub.core.build_application` builders used by the
FastAPI and Streamlit surfaces. Both APIs are stable; new code
should prefer :class:`raghub.RAG`.

Public names are importable directly from their submodules —
this package is intentionally empty of re-exports so that every
public surface is discoverable at the source location.
"""
