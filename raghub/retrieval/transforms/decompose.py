"""Decomposition: ask the LLM to split a multi-hop question into sub-questions.

Each sub-question is searched independently and the hits are fused.
Particularly effective for questions that combine multiple facts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from raghub.exceptions import TransformError
from raghub.models import ConversationTurn
from raghub.retrieval.transforms.base import QueryVariant

SYSTEM_PROMPT = (
    "You decompose compound questions into independent sub-questions "
    "for retrieval. Reply with a JSON array of strings only — no prose."
)


def build_prompt(question: str) -> str:
    return (
        "Split the following compound question into the minimum set of "
        "independent sub-questions whose answers together imply the "
        "original answer. Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


def extract_json_array(raw: str) -> list[str]:
    """Mirror of :func:`raghub.retrieval.transforms.multi_query.extract_json_array`.

    Inlined here to keep the two transforms independent — there is no
    shared ``_llm_json`` helper yet because only two callers need it
    and a third would be the right time to factor one out.
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
    return [str(item).strip() for item in parsed if str(item).strip()]


class DecomposeTransformer:
    """Decomposition transformer.

    Attributes:
        name: Always ``"decompose"``.
    """

    name = "decompose"

    def __init__(self, llm) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.
        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[ConversationTurn] = (),
    ) -> list[QueryVariant]:
        """Produce sub-question variants.

        Args:
            question: User question.
            history: Unused; kept for interface symmetry.

        Returns:
            One :class:`QueryVariant` per sub-question, all with
            ``kind="sub"``. Empty list on parse failure.
        """
        raw = await self.llm.async_generate(
            system_prompt=SYSTEM_PROMPT,
            conversation=list(history),
            context=[],
            question=build_prompt(question),
        )
        sub_questions = extract_json_array(raw or "")
        return [QueryVariant(text=q, kind="sub") for q in sub_questions]


__all__ = ["DecomposeTransformer"]