"""Retrieval-augmented generation framework.

This package exposes the reusable application builder and container types.
It intentionally avoids importing optional web dependencies.
"""

from dynamic_rag.core.container import build_application
from dynamic_rag.services.application import DynamicRagApplication, DynamicRagContainer

__all__ = [
    "DynamicRagApplication",
    "DynamicRagContainer",
    "build_application",
]
