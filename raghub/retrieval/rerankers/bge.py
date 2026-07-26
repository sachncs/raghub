"""BGE / sentence-transformers cross-encoder reranker.

Local model; no API key required but downloads ~2 GB on first use.
The :mod:`sentence_transformers` package is a required runtime
dependency; the encoder is loaded on first use.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from raghub.models import RetrievalHit
from raghub.observability.metrics import record_rerank_latency


def record_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up.

    Args:
        provider: The reranker provider label (``"bge"`` here).
        seconds: Observed wall-clock latency.
    """
    record_rerank_latency(provider, seconds)


class BgeReranker:
    """Local cross-encoder reranker (default: ``BAAI/bge-reranker-v2-m3``).

    Attributes:
        name: ``"bge"``.
    """

    name = "bge"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 20,
        encoder: CrossEncoder | None = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            model: HuggingFace model id or local path.
            top_k: Maximum candidates the reranker is asked to score.
            encoder: Optional pre-built ``CrossEncoder`` instance.
                When supplied the model is not loaded — useful for
                tests and for sharing an encoder across rerankers.
        """
        self.model = model
        self.top_k = top_k
        self.encoder = encoder

    def ensure_encoder(self) -> CrossEncoder:
        """Return the underlying ``CrossEncoder``, loading it on first use.

        Returns:
            The :class:`sentence_transformers.CrossEncoder` instance.
        """
        if self.encoder is None:
            self.encoder = CrossEncoder(self.model)
        return self.encoder

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Return the hits reordered by the cross-encoder's relevance score.

        Args:
            question: User query.
            hits: Candidate hits from the retriever.

        Returns:
            The same hits reordered by descending cross-encoder score.
        """
        if not hits:
            return []
        started = time.perf_counter()
        ordered = self.do_rerank(question, hits)
        record_latency(self.name, time.perf_counter() - started)
        return ordered

    def do_rerank(
        self, question: str, hits: Sequence[RetrievalHit]
    ) -> list[RetrievalHit]:
        """Score every ``(question, chunk)`` pair and sort by the score.

        Args:
            question: The user query.
            hits: The candidates to rerank.

        Returns:
            The hits reordered to match the cross-encoder's scores.
        """
        encoder = self.ensure_encoder()
        pairs = [(question, hit.chunk.text) for hit in hits]
        scores = list(encoder.predict(pairs))
        ordered = sorted(
            zip(scores, hits, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [hit for _, hit in ordered[: self.top_k]]


__all__ = ["BgeReranker", "record_latency"]
