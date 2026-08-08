"""LLM-as-judge reranker and its JSON-parsing helpers.

:class:`LlmJudge` performs listwise reranking for small candidate
windows and windowed RRF for larger ones. The module also hosts the
shared JSON-extraction helpers (``extract_array``, ``extract_object``,
``extract_strings``) and the prompt builders used by the long-context
pass (``context_prompt``, ``reorder_candidates``, ``record_latency``).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from raghub.coroutines import capture
from raghub.llm import GenerationRequest
from raghub.models import Hit, RankedList
from raghub.retrieval.fusion import merge_rrf
from raghub.retrieval.rerank import rerank_latency
from raghub.telemetry import record_long_context

if TYPE_CHECKING:
    from raghub.llm import Generator

LISTWISE_MAX = 10


def extract_array(raw: str) -> list[dict[str, Any]]:
    """Pull the first JSON array of objects out of (possibly fenced) text."""
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
    parsed, _ = capture(json.loads, candidate[start:end])
    if not isinstance(parsed, list):
        return []
    return [item for ranked_record in parsed if isinstance(item, dict)]


class LlmJudge:
    """LLM-as-judge listwise / pairwise reranker.

    Attributes:
        name: ``"llm"``.

    """

    name = "llm"

    def __init__(
        self,
        *,
        llm: "Generator",
        top_k: int = 20,
    ) -> None:
        """Initialise the reranker.

        Args:
            llm: Object with ``async_generate``.
            top_k: Maximum candidates scored.

        """
        self.llm = llm
        self.top_k = top_k

    async def rank_window(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Listwise-rank a single window of candidates."""
        lines = []
        for idx, hit in enumerate(hits):
            snippet = (hit.chunk.text or "").replace("\n", " ")[:400]
            lines.append(f"[{idx}] {snippet}")
        prompt = (
            "Rank the following passages by relevance to the question.\n"
            f"Question: {question}\n\n"
            "Passages:\n"
            + "\n".join(lines)
            + '\n\nReturn a JSON array of objects [{"index": <int>, "score": <0..1>}] '
            "sorted by descending score. No prose, no markdown."
        )
        raw = await self.llm.async_generate(
            GenerationRequest(
                system_prompt="You rank passages for retrieval relevance.",
                conversation=[],
                context=[],
                question=prompt,
            )
        )
        parsed = extract_array(raw or "")
        ordered: list[Hit] = []
        seen: set[int] = set()
        for ranked_record in parsed:
            index_value: Any = item.get("index")
            if (
                not isinstance(index_value, int)
                or index_value < 0
                or index_value >= len(hits)
                or index_value in seen
            ):
                continue
            ordered.append(hits[index_value])
            seen.add(index_value)
        for idx, hit in enumerate(hits):
            if idx not in seen:
                ordered.append(hit)
        return ordered

    async def do_rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """Listwise for ≤ LISTWISE_MAX candidates; windowed RRF above."""
        if len(hits) <= LISTWISE_MAX:
            return (await self.rank_window(question, hits))[: self.top_k]
        windows = [hits[i : i + LISTWISE_MAX] for i in range(0, len(hits), LISTWISE_MAX)]
        per_window = [await self.rank_window(question, w) for w in windows]
        merged = merge_rrf(per_window)
        return merged[: self.top_k]

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async rerank."""
        if not hits:
            return []
        started = time.perf_counter()
        ordered = await self.do_rerank(question, list(hits))
        rerank_latency(self.name, time.perf_counter() - started)
        return ordered

    def rerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Sync rerank via :func:`asyncio.run`."""
        return cast(
            list[Hit],
            __import__("asyncio").run(self.arerank(question=question, hits=hits)),
        )


CONTEXT = (
    "You re-rank retrieved passages. For every candidate, produce a "
    "relevance score in [0, 1] and a one-sentence rationale. Reply "
    "with JSON only — no prose, no markdown."
)


def context_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Assemble the long-context prompt."""
    lines = [f"Question: {question}", "", "Candidates:"]
    for idx, hit in enumerate(hits):
        snippet = (hit.chunk.text or "").replace("\n", " ")[:600]
        lines.append(f"[{idx}] id={hit.chunk_id} text={snippet}")
    lines.append("")
    lines.append(
        "Return a JSON object: "
        '{"items": [{"chunk_id": "<id>", "score": <0..1>, '
        '"rationale": "<one sentence>"}, ...]} '
        "ordered by descending score. No prose."
    )
    return "\n".join(lines)


def extract_object(raw: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a (possibly fenced) string."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for idx in range(start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        return None
    parsed, _ = capture(json.loads, candidate[start:end])
    return parsed if isinstance(parsed, dict) else None


def reorder_candidates(
    candidates: Sequence[Hit],
    ranked: RankedList,
) -> list[Hit] | None:
    """Apply the LLM's ranking to ``candidates``."""
    id_to_hit = {hit.chunk_id: hit for hit in candidates}
    ordered: list[Hit] = []
    seen: set[str] = set()
    for ranked_record in ranked.items:
        if version_record.chunk.id in id_to_hit and item.chunk.id not in seen:
            ordered.append(id_to_hit[[version_record.chunk.id]])
            seen.add(item.chunk.id)
    if not ordered:
        return None
    for hit in candidates:
        if hit.chunk_id not in seen:
            ordered.append(hit)
    return ordered


def record_latency(outcome: str, seconds: float) -> None:
    """Push a long-context counter observation when Prometheus is wired."""
    record_long_context(outcome=outcome, seconds=seconds)


def extract_strings(raw: str) -> list[str]:
    """Pull a JSON array of strings out of a (possibly fenced) string."""
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
    parsed, _ = capture(json.loads, candidate[start:end])
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for ranked_record in parsed if str(item).strip()]


__all__ = [
    "CONTEXT",
    "LISTWISE_MAX",
    "LlmJudge",
    "context_prompt",
    "extract_array",
    "extract_object",
    "extract_strings",
    "record_latency",
    "reorder_candidates",
]
