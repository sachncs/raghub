"""Retrieval support: rerankers, query transformers, fusion, faceted search.

Collapses the previous 18 files (top-level + rerankers/ + transforms/)
into a single helper module. The package surface is small:

Class summary::

    Variant              - one rephrased question (Pydantic value object).
    Transformer          - async rewriter; subclasses implement transform().
    Hyde, MultiQuery, Decompose, StepBack, Compose
                         - concrete transforms; ``Compose`` chains them.
    Rerank               - protocol with rerank() / arerank() variants.
    Identity, Bge, Cohere, Cascade, LlmJudge, Colbert, Context
                         - concrete reranker implementations. ``Context`` is
                           the long-context second pass.
    Fusion               - RRF / linear combine of ranked lists.
    Pipeline             - end-to-end vector + keyword + hybrid retrieval.
    Search               - faceted chunk search with filters.
    RerankerFactory      - build rerankers from application Settings.

The package-level entry points :func:`reranker`, :func:`areranker` and
:func:`transform` dispatch to a named implementation via the
``method`` keyword.

Long-context LLM rerank is exposed as :class:`Context` (was the
``LongContextRerankPass`` module — the "long_context" name was just
noise; the class is a context-bound rerank pass).
"""

from raghub.retrieval.colbert import Colbert
from raghub.retrieval.context import Context
from raghub.retrieval.factories import (
    RerankerFactory,
    areranker,
    build_reranker,
    build_transformer,
    reranker,
    transform,
)
from raghub.retrieval.fusion import (
    Fusion,
    LinearFusion,
    ReciprocalRankFusion,
    linear_combine,
    merge_rrf,
    reciprocal_rank_fusion,
)
from raghub.retrieval.judge import (
    LlmJudge,
    context_prompt,
    extract_array,
    extract_object,
    extract_strings,
    record_latency,
    reorder_candidates,
)
from raghub.retrieval.pipeline import Retrieval
from raghub.retrieval.rerank import Cascade, Cohere, Identity, rerank_latency
from raghub.retrieval.search import Search, SearchFilters, build_filter
from raghub.retrieval.transforms import (
    Compose,
    Decompose,
    Hyde,
    MultiQuery,
    StepBack,
    decompose_prompt,
    hyde_prompt,
    query_prompt,
    step_prompt,
)
from raghub.retrieval.types import Rerank, Transformer, Variant

__all__ = [
    "Cascade",
    "Cohere",
    "Colbert",
    "Compose",
    "Context",
    "Decompose",
    "Fusion",
    "Hyde",
    "Identity",
    "LlmJudge",
    "MultiQuery",
    "LinearFusion",
    "Rerank",
    "RerankerFactory",
    "ReciprocalRankFusion",
    "Retrieval",
    "Search",
    "SearchFilters",
    "StepBack",
    "Transformer",
    "Variant",
    "areranker",
    "build_filter",
    "build_reranker",
    "build_transformer",
    "context_prompt",
    "decompose_prompt",
    "extract_array",
    "extract_object",
    "extract_strings",
    "hyde_prompt",
    "linear_combine",
    "merge_rrf",
    "query_prompt",
    "reciprocal_rank_fusion",
    "record_latency",
    "reorder_candidates",
    "rerank_latency",
    "reranker",
    "step_prompt",
    "transform",
]
