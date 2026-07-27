"""Multi-query: ask the LLM to rephrase the question ``N`` ways.

Each rephrasing is embedded and searched in parallel; results are
fused via RRF downstream. Improves recall when the user's wording
differs from the corpus's vocabulary.
"""

from __future__ import annotations

import json
import re
from typing import Any
from collections.abc import Sequence

from raghub.exceptions import TransformError
from raghub.models import ConversationTurn
from raghub.retrieval.transforms.base import QueryVariant
from raghub.utils import capture

SYSTEM_PROMPT = (
    "You generate alternative phrasings of a question for retrieval. "
    "Reply with a JSON array of strings only — no prose, no preamble."
)


def build_prompt(question: str, n: int) -> str:
    return (
        f"Rewrite the following question as {n} distinct search queries. "
        f"Vary vocabulary and structure; keep the intent identical. "
        f"Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


def extract_json_array(raw: str) -> list[str]:
    """Pull the first JSON array out of ``raw`` (markdown-tolerant)."""
    if not raw:
        return []
    # Strip ```json fences if the model added them.
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else raw
    # Find the first '[' and matching ']' greedily.
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
    return [str(item).strip() for item in parsed if str(item).strip()]


class MultiQueryTransformer:
    """Multi-query rewriter.

    Attributes:
        name: Always ``"multi_query"``.
    """

    name = "multi_query"

    def __init__(self, llm: Any, *, n: int = 4) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.
            n: Number of rephrasings to request. Defaults to 4.
        """
        if n < 1:
            raise ValueError("multi-query ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> list[QueryVariant]:
        """Generate ``n`` alternative phrasings.

        Args:
            question: User question.
            history: Unused; kept for interface symmetry.

        Returns:
            Up to ``n`` :class:`QueryVariant` objects with
            ``kind="multi_query"``. On parse failure the empty list is
            returned and the caller falls back to the original.
        """
        raw = await self.llm.async_generate(
            system_prompt=SYSTEM_PROMPT,
            conversation=list(history),
            context=[],
            question=build_prompt(question, self.n),
        )
        phrasings = extract_json_array(raw or "")
        variants: list[QueryVariant] = []
        for phrase in phrasings[: self.n]:
            variants.append(QueryVariant(text=phrase, kind="multi_query"))
        return variants


