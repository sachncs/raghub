from __future__ import annotations

"""Query-transform package.

Re-exports the public surface so callers can do
``from raghub.retrieval.transforms import QueryTransformer, QueryVariant``.
Concrete transforms (``HydeTransformer`` etc.) live in their own modules.
"""

from raghub.retrieval.transforms.base import (
    QueryTransformer,
    QueryVariant,
    QueryVariantKind,
)
from raghub.retrieval.transforms.compose import ComposeTransformer
from raghub.retrieval.transforms.decompose import DecomposeTransformer
from raghub.retrieval.transforms.hyde import HydeTransformer
from raghub.retrieval.transforms.multi_query import MultiQueryTransformer
from raghub.retrieval.transforms.step_back import StepBackTransformer

__all__ = [
    "ComposeTransformer",
    "DecomposeTransformer",
    "HydeTransformer",
    "MultiQueryTransformer",
    "QueryTransformer",
    "QueryVariant",
    "QueryVariantKind",
    "StepBackTransformer",
]