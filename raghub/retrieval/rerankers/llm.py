"""LLM-as-judge listwise / pairwise reranker.

Uses the project's existing LLM provider (``BaseLLMProvider``) — no
extra dependencies. Two strategies:

* **listwise** (≤ :data:`LISTWISE_MAX` hits): one prompt asks the model
  to return a JSON array of ``{"index": int, "score": float}`` items
  in relevance order.
* **windowed RRF** (> :data:`LISTWISE_MAX` hits): chunk the list into
  ``LISTWISE_MAX``-sized windows, listwise-rank each, then merge
  via RRF (avoids a quadratic comparison).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Sequence
from typing import Any

from raghub.models import RetrievalHit
from raghub.observability import record_rerank_latency

LISTWISE_MAX = 10


def extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Pull a JSON array of objects out of a (possibly fenced) string.

    Args:
        raw: The LLM's raw output.

    Returns:
        A list of dict entries; empty when none can be found or
        parsed.
    """
    if not raw:
        return []
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return []
    try:
        parsed = json.loads(candidate[start:end])
    except ValueError:
        return []
    return [item for item in parsed if isinstance(item, dict)]


def record_latency(provider: str, seconds: float) -> None:
    """Push a histogram observation when Prometheus is wired up.

    Args:
        provider: The reranker provider label (``"llm"`` here).
        seconds: Observed wall-clock latency.
    """
    record_rerank_latency(provider, seconds)


def merge_with_rrf(
    per_window: list[list[RetrievalHit]], rrf_k: int = 60
) -> list[RetrievalHit]:
    """Reciprocal-Rank-Fusion merge across ranked windows.

    Args:
        per_window: Each window's listwise ranking.
        rrf_k: RRF damping constant. ``60`` matches the literature.

    Returns:
        The unique hits in RRF-merged order.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranked in per_window:
        for rank, hit in enumerate(ranked, start=1):
            cid = hit.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in order:
                order[cid] = len(order)
    # Stable sort by score desc, then first-seen order.
    return sorted(
        {hit.chunk_id: hit for window in per_window for hit in window}.values(),
        key=lambda h: (-scores.get(h.chunk_id, 0.0), order.get(h.chunk_id, 0)),
    )


class LLMReranker:
    """LLM-as-judge reranker.

    Attributes:
        name: ``"llm"``.
    """

    name = "llm"

    def __init__(
        self,
        *,
        llm: Any,
        top_k: int = 20,
    ) -> None:
        """Initialise the reranker.

        Args:
            llm: Any object with an ``async_generate`` method
                matching :class:`raghub.llm.BaseLLMProvider`.
            top_k: Maximum candidates the reranker is asked to score.
        """
        self.llm = llm
        self.top_k = top_k

    async def arerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Async rerank.

        Args:
            question: User query.
            hits: Candidate hits from the retriever.

        Returns:
            The same hits reordered by the LLM's relevance ranking.
        """
        if not hits:
            return []
        started = time.perf_counter()
        ordered = await self.do_rerank(question, list(hits))
        record_latency(self.name, time.perf_counter() - started)
        return ordered

    def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Sync rerank via ``asyncio.run``.

        ``QueryPipeline`` always calls ``arerank`` directly — this
        shim is only used by callers that treat the reranker as a
        regular sync :class:`Reranker`.
        """
        return asyncio.run(self.arerank(question=question, hits=hits))

    async def do_rerank(
        self, question: str, hits: list[RetrievalHit]
    ) -> list[RetrievalHit]:
        """Pick listwise vs windowed RRF based on candidate count.

        Args:
            question: The user query.
            hits: The candidates to rerank.

        Returns:
            The hits reordered to match the LLM's ranking.
        """
        if len(hits) <= LISTWISE_MAX:
            return (await self.rank_window(question, hits))[: self.top_k]
        windows = [
            hits[i : i + LISTWISE_MAX]
            for i in range(0, len(hits), LISTWISE_MAX)
        ]
        per_window = [await self.rank_window(question, w) for w in windows]
        merged = merge_with_rrf(per_window)
        return merged[: self.top_k]

    async def rank_window(
        self, question: str, hits: list[RetrievalHit]
    ) -> list[RetrievalHit]:
        """Listwise-rank a single window of candidates.

        Args:
            question: The user query.
            hits: The window to rank.

        Returns:
            The hits in the LLM's ranked order, with any chunk the
            model forgot appended in input order.
        """
        lines = []
        for idx, hit in enumerate(hits):
            snippet = (hit.chunk.text or "").replace("\n", " ")[:400]
            lines.append(f"[{idx}] {snippet}")
        prompt = (
            f"Rank the following passages by relevance to the question.\n"
            f"Question: {question}\n\n"
            "Passages:\n"
            + "\n".join(lines)
            + "\n\nReturn a JSON array of objects [{\"index\": <int>, \"score\": <0..1>}] "
            "sorted by descending score. No prose, no markdown."
        )
        raw = await self.llm.async_generate(
            system_prompt="You rank passages for retrieval relevance.",
            conversation=[],
            context=[],
            question=prompt,
        )
        parsed = extract_json_array(raw or "")
        ordered: list[RetrievalHit] = []
        seen: set[int] = set()
        for item in parsed:
            idx = item.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(hits) or idx in seen:
                continue
            ordered.append(hits[idx])
            seen.add(idx)
        # Any hit the model forgot → append in input order.
        for idx, hit in enumerate(hits):
            if idx not in seen:
                ordered.append(hit)
        return ordered


__all__ = [
    "LISTWISE_MAX",
    "LLMReranker",
    "extract_json_array",
    "merge_with_rrf",
    "record_latency",
]