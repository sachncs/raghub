"""Concrete reranker implementations.

Provides the no-op :class:`Identity` reranker, the :class:`Cohere`
cross-encoder wrapper, and the two-stage :class:`Cascade` that only
invokes the expensive reranker when the cheap one didn't have an opinion.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from pydantic import SecretStr

from raghub.errors import RerankerError
from raghub.constants import ENV_COHERE_API_KEY
from raghub.models import Hit
from raghub.telemetry import record_rerank_latency

if TYPE_CHECKING:
    import cohere

    from raghub.retrieval.types import Rerank


class Identity:
    """No-op reranker.

    Attributes:
        name: ``"identity"``.

    """

    name = "identity"

    @staticmethod
    def rerank(*, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Return ``hits`` unchanged (identity pass-through)."""
        return list(hits)

    @staticmethod
    async def arerank(*, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async identity pass-through."""
        return list(hits)


def rerank_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up."""
    record_rerank_latency(provider, seconds)


class Cohere:
    """Cohere cross-encoder reranker.

    Attributes:
        name: ``"cohere"``.

    """

    name = "cohere"

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        *,
        model: str = "rerank-english-v3.0",
        top_k: int = 20,
        client: "cohere.Client | None" = None,
    ) -> None:
        """Initialise the reranker.

        Args:
            api_key: Cohere API key. Defaults to ``COHERE_API_KEY`` env var.
            model: Cohere rerank model name.
            top_k: Maximum candidates scored.
            client: Optional pre-built :class:`cohere.Client` (skips init).

        Raises:
            RerankerError: When no API key is available.

        """
        resolved = api_key
        if resolved is None:
            env = os.getenv(ENV_COHERE_API_KEY)
            if not env:
                raise RerankerError("Cohere requires COHERE_API_KEY or an explicit api_key")
            resolved = env
        self.api_key = resolved if isinstance(resolved, SecretStr) else SecretStr(resolved)
        self.model = model
        self.top_k = top_k
        self.client = client

    def ensure_client(self) -> "cohere.Client":
        """Return the underlying :class:`cohere.Client`."""
        if self.client is None:
            import cohere

            self.client = cohere.Client(api_key=self.api_key.get_secret_value())
        return self.client

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Reorder ``hits`` by Cohere's relevance score."""
        if not hits:
            return []
        started = time.perf_counter()
        ordered = self.score(question, hits)
        rerank_latency(self.name, time.perf_counter() - started)
        return ordered

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async shim that pushes the sync rerank onto a worker thread."""
        return cast(
            list[Hit],
            await __import__("asyncio").to_thread(self.rerank, question=question, hits=list(hits)),
        )

    def score(self, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Call the Cohere API and reorder ``hits`` by its output."""
        client = self.ensure_client()
        documents = [hit.chunk.text for hit in hits]
        response = client.rerank(
            model=self.model,
            query=question,
            documents=documents,
            top_n=min(self.top_k, len(documents)),
        )
        ordered: list[Hit] = []
        for result in getattr(response, "results", []):
            idx = getattr(result, "index", None)
            if idx is None or idx < 0 or idx >= len(hits):
                continue
            ordered.append(hits[idx])
        return ordered


class Cascade:
    """Two-stage reranker: ``cheap`` then ``expensive`` (conditionally).

    The expensive reranker is invoked only when the cheap reranker did
    not reorder the input list — i.e. cheap "didn't have an opinion".

    Attributes:
        name: ``"cascade"``.

    """

    name = "cascade"

    def __init__(
        self,
        cheap: "Rerank",
        expensive: "Rerank",
        *,
        spread_threshold: float = 0.05,
    ) -> None:
        """Initialise the cascade.

        Args:
            cheap: First-stage reranker (sync or async).
            expensive: Second-stage reranker invoked only when cheap
                returned the input unchanged.
            spread_threshold: Reserved for future use when cheap
                rerankers expose confidence.

        """
        self.cheap = cheap
        self.expensive = expensive
        self.spread_threshold = float(spread_threshold)

    @staticmethod
    def changed_order(input_hits: Sequence[Hit], ranked: Sequence[Hit]) -> bool:
        """Return ``True`` when ``ranked`` is not the input order."""
        if len(input_hits) != len(ranked):
            return True
        return [h.chunk_id for h in input_hits] != [h.chunk_id for h in ranked]

    @staticmethod
    async def call(reranker: "Rerank", question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Call ``arerank`` when available, else ``rerank`` in a thread."""
        arerank = getattr(reranker, "arerank", None)
        if callable(arerank):
            return list(await arerank(question=question, hits=list(hits)))
        sync = getattr(reranker, "rerank", None)
        if callable(sync):
            return cast(
                list[Hit],
                await __import__("asyncio").to_thread(sync, question=question, hits=list(hits)),
            )
        raise TypeError(f"reranker {reranker!r} has neither arerank nor rerank")

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async cascade.

        Returns ``cheap.rerank(hits)`` when cheap reordered, else
        ``expensive.rerank(cheap.rerank(hits))``.
        """
        if not hits:
            return []
        cheap_ranked = await self.call(self.cheap, question, hits)
        if self.changed_order(hits, cheap_ranked):
            return list(cheap_ranked)
        expensive_ranked = await self.call(self.expensive, question, cheap_ranked)
        id_to_hit = {h.chunk_id: h for h in cheap_ranked}
        ordered = [id_to_hit.get(h.chunk_id, h) for h in expensive_ranked]
        ordered_set = {h.chunk_id for h in ordered}
        for h in cheap_ranked:
            if h.chunk_id not in ordered_set:
                ordered.append(h)
        return ordered

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Sync shim around :meth:`arerank`."""
        return cast(
            list[Hit],
            __import__("asyncio").run(self.arerank(question=question, hits=hits)),
        )


__all__ = [
    "Cascade",
    "Cohere",
    "Identity",
    "rerank_latency",
]
