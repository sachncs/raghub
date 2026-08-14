"""Long-context LLM rerank second pass.

:class:`Context` is the second-pass reranker that uses a long-context
LLM to re-rank a slice of candidates, falling back to the input order
when the pass is ineligible, the LLM errors, or the response is
unparseable.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from raghub.config import LongContextConfig
from raghub.llm import GenerationRequest
from raghub.models import Hit
from raghub.retrieval.judge import (
    CONTEXT,
    context_prompt,
    extract_object,
    record_latency,
    reorder_candidates,
)
from raghub.retrieval.types import Rerank

if TYPE_CHECKING:
    from raghub.llm import Generator


@Rerank.register("long_context")
class Context(Rerank):
    """Long-context second-pass reranker.

    Was named ``LongContextRerankPass``. The "long_context" prefix
    was redundant — context is what this thing consumes.
    """

    name = "long_context"

    def __init__(self, llm: Generator, settings: LongContextConfig) -> None:
        """Initialise the pass.

        Args:
            llm: LLM provider with an ``async_generate`` method.
            settings: The :class:`LongContextConfig` block.

        """
        self.llm = llm
        self.settings = settings

    def is_eligible(self) -> bool:
        """Return ``True`` when the pass should run for the current LLM."""
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
        hits: Sequence[Hit],
    ) -> list[Hit]:
        """Re-order ``hits`` with a long-context LLM call.

        Returns the original order when the pass is not eligible,
        the LLM errors, or the response cannot be parsed.
        """
        if not self.is_eligible() or not hits:
            return list(hits)
        candidates = list(hits[: max(1, self.settings.candidate_k)])
        started = time.perf_counter()
        try:
            raw = await self.llm.async_generate(
                GenerationRequest(
                    system_prompt=CONTEXT,
                    conversation=[],
                    context=[],
                    question=context_prompt(question, candidates),
                )
            )
            parsed = extract_object(raw or "")
            if parsed is None:
                record_latency("bad_json", time.perf_counter() - started)
                return list(hits)
            from raghub.models import RankedList

            ranked = RankedList.model_validate(parsed)
            reordered = reorder_candidates(candidates, ranked)
            if reordered is None:
                record_latency("bad_json", time.perf_counter() - started)
                return list(hits)
            record_latency("ran", time.perf_counter() - started)
            return reordered
        except Exception:  # pragma: no cover - defensive envelope
            record_latency("error", time.perf_counter() - started)
            return list(hits)

    async def arerank(self, *, question: str, hits: Sequence[Hit]) -> list[Hit]:
        """Async alias preserved for symmetry with other rerankers."""
        return await self.rerank(question=question, hits=hits)


__all__ = ["Context"]
