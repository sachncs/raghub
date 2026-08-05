"""LLM-as-judge scoring for faithfulness and answer relevance.

The :class:`Judge` wraps a :class:`raghub.llm.Generator` and uses
prompt templates to score a ``(question, answer, contexts)`` triple
on a 0-1 scale. The judge LLM is expected to reply with a single
number; :func:`parse` extracts and clamps it to ``[0.0, 1.0]``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from raghub.errors import EvaluationError
from raghub.llm import GenerationRequest

SCORE_RE = re.compile(r"(-?)([0-1](?:\.\d+)?|0\.\d+)(?![0-9])")


def parse(text: str) -> float | None:
    """Extract the first 0..1 float from an LLM-as-judge response.

    Args:
        text: The judge's raw response.

    Returns:
        The parsed score in ``[0.0, 1.0]``, or ``None`` when no
        parsable number is found. A leading negative sign is
        accepted so the clamp can map ``-0.5`` to ``0.0``.

    """
    match = SCORE_RE.search(text or "")
    if not match:
        return None
    sign, digits = match.group(1), match.group(2)
    try:
        value = float(f"{sign}{digits}")
    except ValueError:
        return None
    return max(0.0, min(1.0, value))


class Judge:
    """LLM-as-judge scorer for faithfulness and answer relevance.

    Wraps a :class:`raghub.llm.Generator` and uses prompt templates to
    score a ``(question, answer, contexts)`` triple on a 0-1 scale. The
    judge LLM is expected to reply with a single number; the response
    is parsed by :func:`parse` and clamped to ``[0.0, 1.0]``.

    Args:
        llm: The generator used as the judge. Note: the same LLM
            used for answer generation is acceptable, but a stronger
            model (e.g. GPT-4o for a GPT-3.5-turbo pipeline) reduces
            self-bias.
        max_retries: Number of retries on parse failure before
            raising :class:`EvaluationError`. Defaults to 1 retry
            (2 attempts total).

    """

    FAITHFULNESS_PROMPT = (
        "You are evaluating whether an answer is factually supported "
        "by the retrieved context.\n\n"
        "Context:\n{contexts}\n\n"
        "Answer:\n{answer}\n\n"
        "Score the answer's faithfulness from 0.0 to 1.0. A score of "
        "1.0 means every claim in the answer is directly supported by "
        "the context. A score of 0.0 means the answer contains claims "
        "not present in the context.\n\n"
        "Reply with only a single number between 0.0 and 1.0:"
    )

    RELEVANCE_PROMPT = (
        "You are evaluating whether an answer directly addresses the "
        "user's question.\n\n"
        "Question: {question}\n\n"
        "Answer: {answer}\n\n"
        "Score the answer's relevance from 0.0 to 1.0. A score of 1.0 "
        "means the answer fully addresses the question. A score of 0.0 "
        "means the answer is completely unrelated.\n\n"
        "Reply with only a single number between 0.0 and 1.0:"
    )

    def __init__(self, llm: Any, *, max_retries: int = 1) -> None:
        """Store the judge LLM and retry budget."""
        self.llm = llm
        self.max_retries = max_retries

    async def score_once(self, prompt_template: str, **kwargs: str) -> float | None:
        """Run a single prompt, parse the response, or return None on failure."""
        prompt = prompt_template.format(**kwargs)
        try:
            response = await self.llm.async_generate(
                GenerationRequest(
                    system_prompt=prompt,
                    conversation=(),
                    context=(),
                    question=prompt,
                )
            )
        except Exception:
            return None
        return parse(response)

    async def score(self, prompt_template: str, **kwargs: str) -> float:
        """Run a prompt with retries; raise when every attempt fails to parse.

        Args:
            prompt_template: Prompt with ``{placeholders}``.
            **kwargs: Values for the placeholders.

        Returns:
            The parsed score in ``[0.0, 1.0]``.

        Raises:
            EvaluationError: When no attempt yields a parseable score,
                so an LLM outage is never silently read as ``0.0``.

        """
        for _ in range(self.max_retries + 1):
            value = await self.score_once(prompt_template, **kwargs)
            if value is not None:
                return value
        raise EvaluationError(
            f"Judge returned no parseable score after {self.max_retries + 1} attempts"
        )

    async def faithfulness(self, answer: str, contexts: Sequence[str]) -> float:
        r"""Score the answer's faithfulness on a 0-1 scale.

        Args:
            answer: The generated answer.
            contexts: The retrieved context strings; joined with
                ``\n\n---\n\n`` before being inserted into the prompt.

        Returns:
            A score in ``[0.0, 1.0]``.

        Raises:
            EvaluationError: When every retry fails to parse.

        """
        joined = "\n\n---\n\n".join(contexts)
        return await self.score(self.FAITHFULNESS_PROMPT, answer=answer, contexts=joined)

    async def answer_relevance(self, answer: str, question: str) -> float:
        """Score the answer's relevance to the question on a 0-1 scale.

        Args:
            answer: The generated answer.
            question: The user's question.

        Returns:
            A score in ``[0.0, 1.0]``.

        Raises:
            EvaluationError: When every retry fails to parse.

        """
        return await self.score(self.RELEVANCE_PROMPT, answer=answer, question=question)


__all__ = ["Judge", "parse"]
