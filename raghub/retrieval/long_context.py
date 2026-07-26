"""Long-context second-pass rerank (Phase 5.1).

After the first-pass retrieval + cross-encoder rerank, the top-K
candidates are pushed into a long-context LLM (Claude 3.5/3.7,
Gemini 1.5/2.0, Command-R+, GPT-4.1, …) which re-orders them with
a one-sentence rationale per chunk.

The pass is opt-in via :class:`LongContextConfig` and silently
no-ops when:

* ``long_context_pass.enabled`` is ``False``;
* the configured LLM model is not in the allowlist;
* the LLM call fails (caller sees the original reranked order).

This is the conservative behaviour the regression test for
"missing-model" is meant to lock in: a typo in
``LLM_MODEL`` cannot crash the chat surface; the LLM default
takes over and the second pass becomes a no-op.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any

from raghub.config import LongContextConfig
from raghub.models import RankedList, RetrievalHit
from raghub.observability import record_long_context

SYSTEM_PROMPT = (
    "You re-rank retrieved passages. For every candidate, produce a "
    "relevance score in [0, 1] and a one-sentence rationale. Reply "
    "with JSON only — no prose, no markdown."
)


def build_prompt(question: str, hits: Sequence[RetrievalHit]) -> str:
    """Assemble the long-context prompt.

    Each candidate is numbered so the LLM can refer to it by id
    in the returned JSON. The first hit's chunk text is included
    in full; the rationale can quote from it.

    Args:
        question: The user's question.
        hits: First-pass reranked hits (top-K).

    Returns:
        A multi-line prompt body.
    """
    lines = [
        f"Question: {question}",
        "",
        "Candidates:",
    ]
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


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a (possibly fenced) string.

    Args:
        raw: The LLM's raw output.

    Returns:
        The first balanced JSON object, or ``None`` when none can
        be found or parsed.
    """
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
    try:
        return json.loads(candidate[start:end])
    except ValueError:
        return None


def reorder_candidates(
    candidates: Sequence[RetrievalHit],
    ranked: RankedList,
) -> list[RetrievalHit] | None:
    """Apply the LLM's ranking to ``candidates``.

    Args:
        candidates: The first-pass hits (top-K).
        ranked: The LLM's :class:`RankedList` response.

    Returns:
        The candidates reordered to match ``ranked.items``, or
        ``None`` when the LLM omitted every chunk id (so the caller
        can keep the first-pass order instead of guessing).
    """
    id_to_hit = {hit.chunk_id: hit for hit in candidates}
    ordered: list[RetrievalHit] = []
    seen: set[str] = set()
    for item in ranked.items:
        if item.chunk_id in id_to_hit and item.chunk_id not in seen:
            ordered.append(id_to_hit[item.chunk_id])
            seen.add(item.chunk_id)
    if not ordered:
        return None
    # Append any candidates the model forgot, preserving their
    # original (first-pass) order.
    for hit in candidates:
        if hit.chunk_id not in seen:
            ordered.append(hit)
    return ordered


def record_latency(outcome: str, seconds: float) -> None:
    """Push a long-context counter observation when Prometheus is wired.

    Args:
        outcome: One of ``"ran"``, ``"skipped"``, ``"bad_json"``,
            ``"error"``.
        seconds: Observed wall-clock duration.
    """
    record_long_context(outcome=outcome, seconds=seconds)


class LongContextRerankPass:
    """Long-context second-pass rerank (Phase 5.1).

    Attributes:
        name: ``"long_context"``.
    """

    name = "long_context"

    def __init__(self, llm: Any, settings: LongContextConfig) -> None:
        """Initialise the pass.

        Args:
            llm: Any object with ``async_generate`` matching the
                :class:`raghub.llm.BaseLLMProvider` interface.
                The LLM's ``model_name`` is checked against
                ``settings.allowlist_models`` before the pass runs.
            settings: The :class:`LongContextConfig` block.
        """
        self.llm = llm
        self.settings = settings

    def is_eligible(self) -> bool:
        """Return ``True`` when this pass should run for the current LLM.

        The check is intentionally strict: a misconfigured LLM model
        silently disables the pass rather than failing the request.
        """
        if not self.settings.enabled:
            return False
        model_name = getattr(self.llm, "model_name", "") or ""
        if not model_name:
            return False
        return model_name in (self.settings.allowlist_models or [])

    async def rerank(
        self,
        *,
        question: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Re-order ``hits`` with a long-context LLM call.

        Args:
            question: The user's question.
            hits: First-pass reranked hits. The pass trims the list
                to :attr:`settings.candidate_k` before sending.

        Returns:
            The same hits in the long-context order.

        Raises:
            ValueError: When the LLM response cannot be parsed or
                validated.
            RuntimeError: When the underlying LLM call fails.
        """
        if not self.is_eligible() or not hits:
            return list(hits)
        candidates = list(hits[: max(1, self.settings.candidate_k)])
        started = time.perf_counter()
        raw = await self.llm.async_generate(
            system_prompt=SYSTEM_PROMPT,
            conversation=[],
            context=[],
            question=build_prompt(question, candidates),
        )
        parsed = extract_json_object(raw or "")
        if parsed is None:
            record_latency(
                outcome="bad_json", seconds=time.perf_counter() - started
            )
            raise ValueError("long-context rerank produced unparseable JSON")
        ranked = RankedList.model_validate(parsed)
        reordered = reorder_candidates(candidates, ranked)
        if reordered is None:
            record_latency(
                outcome="bad_json", seconds=time.perf_counter() - started
            )
            raise ValueError("long-context rerank omitted every candidate id")
        record_latency(outcome="ran", seconds=time.perf_counter() - started)
        return reordered


__all__ = [
    "LongContextRerankPass",
    "build_prompt",
    "extract_json_object",
    "record_latency",
    "reorder_candidates",
]