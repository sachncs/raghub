"""Query transformation primitives.

Each transformer turns a single question into one or more rephrased
:class:`Variant`s. :class:`Compose` chains several transforms in order
and always prepends the original question so retrieval stays biased
toward the user's literal wording.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from raghub.llm import GenerationRequest
from raghub.models import Turn
from raghub.retrieval.judge import extract_strings
from raghub.retrieval.types import ORIGINAL_WEIGHT, Transformer, Variant

if TYPE_CHECKING:
    from raghub.llm import Generator

HYDE = (
    "You generate hypothetical passages for retrieval. Reply with the "
    "passage only — no preamble, no heading, no commentary."
)


def hyde_prompt(question: str) -> str:
    """Build the HyDE prompt that elicits a hypothetical passage for retrieval."""
    return (
        "Write a short paragraph (3-5 sentences) that would answer the "
        "following question. The paragraph does not need to be factual — "
        "it just needs to use the same vocabulary and phrasing a real "
        "source document would use.\n\n"
        f"Question: {question}\n\nPassage:"
    )


class Hyde:
    """HyDE (Hypothetical Document Embeddings) transformer.

    Attributes:
        name: Always ``"hyde"``.

    """

    name = "hyde"

    def __init__(self, llm: Generator, *, n: int = 1) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with an ``async_generate`` method.
            n: Number of hypothetical passages to generate.

        """
        if n < 1:
            raise ValueError("HyDE ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Generate ``n`` hypothetical passages."""
        prompt = hyde_prompt(question)
        variants: list[Variant] = []
        for _ in range(self.n):
            text = await self.llm.async_generate(
                GenerationRequest(
                    system_prompt=HYDE,
                    conversation=list(history),
                    context=[],
                    question=prompt,
                )
            )
            text = (text or "").strip()
            if text:
                variants.append(Variant(text=text, kind="hyde"))
        return variants


MULTI_QUERY = (
    "You generate alternative phrasings of a question for retrieval. "
    "Reply with a JSON array of strings only — no prose, no preamble."
)


def query_prompt(question: str, n: int) -> str:
    """Build the multi-query prompt that returns N rephrasings as JSON."""
    return (
        f"Rewrite the following question as {n} distinct search queries. "
        "Vary vocabulary and structure; keep the intent identical. "
        "Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


class MultiQuery:
    """Multi-query rewriter.

    Attributes:
        name: Always ``"multi_query"``.

    """

    name = "multi_query"

    def __init__(self, llm: Generator, *, n: int = 4) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.
            n: Number of rephrasings to request.

        """
        if n < 1:
            raise ValueError("multi-query ``n`` must be >= 1")
        self.llm = llm
        self.n = n

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Generate ``n`` alternative phrasings."""
        raw = await self.llm.async_generate(
            GenerationRequest(
                system_prompt=MULTI_QUERY,
                conversation=list(history),
                context=[],
                question=query_prompt(question, self.n),
            )
        )
        phrasings = extract_strings(raw or "")
        return [Variant(text=phrase, kind="multi_query") for phrase in phrasings[: self.n]]


DECOMPOSE = (
    "You decompose compound questions into independent sub-questions "
    "for retrieval. Reply with a JSON array of strings only — no prose."
)


def decompose_prompt(question: str) -> str:
    """Build the decomposition prompt that splits a compound question."""
    return (
        "Split the following compound question into the minimum set of "
        "independent sub-questions whose answers together imply the "
        "original answer. Output a JSON array of strings.\n\n"
        f"Question: {question}\n\nJSON:"
    )


class Decompose:
    """Decomposition transformer.

    Attributes:
        name: Always ``"decompose"``.

    """

    name = "decompose"

    def __init__(self, llm: Generator) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.

        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Produce sub-question variants."""
        raw = await self.llm.async_generate(
            GenerationRequest(
                system_prompt=DECOMPOSE,
                conversation=list(history),
                context=[],
                question=decompose_prompt(question),
            )
        )
        sub_questions = extract_strings(raw or "")
        return [Variant(text=q, kind="sub") for q in sub_questions]


STEP_BACK = (
    "You reframe a specific question as a more abstract, principle-"
    "level question. Reply with one sentence only — no preamble."
)


def step_prompt(question: str) -> str:
    """Build the step-back prompt that asks for the principle-level question."""
    return (
        "Given the specific question below, write the more general, "
        "principle-level question that would provide useful background. "
        "Reply with one sentence only.\n\n"
        f"Specific: {question}\n\nAbstract:"
    )


class StepBack:
    """Step-back prompting transformer.

    Attributes:
        name: Always ``"step_back"``.

    """

    name = "step_back"

    def __init__(self, llm: Generator) -> None:
        """Initialise the transformer.

        Args:
            llm: Object with ``async_generate``.

        """
        self.llm = llm

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Produce the abstract reformulation."""
        abstract = await self.llm.async_generate(
            GenerationRequest(
                system_prompt=STEP_BACK,
                conversation=list(history),
                context=[],
                question=step_prompt(question),
            )
        )
        text = (abstract or "").strip()
        if not text:
            return []
        return [Variant(text=text, kind="step_back", weight=1.2)]


class Compose:
    """Run several transforms in order; prepend the original question.

    The original question is always present in the output (weight
    ``1.5``) so retrieval is biased toward the user's literal phrasing
    even when every transform fails or returns nothing.

    Attributes:
        name: ``"compose"``.

    """

    name = "compose"

    def __init__(self, transformers: Sequence[Transformer]) -> None:
        """Initialise the composer.

        Args:
            transformers: Ordered list of transforms to apply. Each is
                awaited sequentially.

        """
        self.transformers: list[Transformer] = list(transformers)

    async def transform(
        self,
        *,
        question: str,
        history: Sequence[Turn] = (),
    ) -> list[Variant]:
        """Combine the original question with every transformer's output."""
        variants: list[Variant] = [Variant(text=question, kind="original", weight=ORIGINAL_WEIGHT)]
        for t in self.transformers:
            produced = await t.transform(question=question, history=list(history))
            variants.extend(produced)
        return variants


__all__ = [
    "DECOMPOSE",
    "HYDE",
    "MULTI_QUERY",
    "STEP_BACK",
    "Compose",
    "Decompose",
    "Hyde",
    "MultiQuery",
    "StepBack",
    "decompose_prompt",
    "hyde_prompt",
    "query_prompt",
    "step_prompt",
]
