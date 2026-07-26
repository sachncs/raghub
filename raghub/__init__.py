"""Retrieval-augmented generation framework.

The package exposes a high-level :class:`RAG` facade (the spec entry
point) plus the legacy :class:`DynamicRagApplication` /
:func:`build_application` builders used by the FastAPI and Streamlit
surfaces. Both APIs are stable; new code should prefer
:class:`RAG`.
"""

from raghub.core import build_application
from raghub.rag import RAG
from raghub.services.application import DynamicRagApplication, DynamicRagContainer
