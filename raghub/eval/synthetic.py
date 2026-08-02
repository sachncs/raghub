"""Synthetic dataset generator for RAGHub evaluation.

:class:`SyntheticDataset` generates (question, answer, contexts)
triples from a corpus of documents. The result is suitable for
the same evaluators as :class:`raghub.eval.Finance` —
:func:`Metrics.evaluate` and the benchmarks all consume the
canonical schema.

The generator needs an LLM to produce the questions and answers.
Any :class:`raghub.llm.Generator` with an ``async_generate``
method works. The offline ``HeuristicProvider`` is enough to
exercise the pipeline but cannot produce meaningful synthetic
data — it returns the most token-overlap sentence from the
context, which is the context itself.

For real synthesis, set ``RAG_LLM_API_KEY`` (or any provider-specific
env var) and the generator will use a real LLM.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, cast

from raghub.llm import GenerationRequest, Generator


def chunk_text(chunk: Any) -> str:
    """Extract the text from a chunk-like object.

    Args:
        chunk: An object with a ``.text`` attribute, or a string.

    Returns:
        The chunk's text content.

    Raises:
        ValueError: When ``chunk`` has neither a ``.text`` attribute
            nor is a string.

    """
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    if text is None:
        raise ValueError(f"corpus item has no .text attribute: {chunk!r}")
    return cast(str, text)


def chunk_id(chunk: Any) -> str:
    """Extract a stable id from a chunk-like object.

    Args:
        chunk: An object with a ``.chunk_id`` or ``.id`` attribute,
            or a string.

    Returns:
        The chunk's id (from ``.chunk_id`` first, then ``.id``,
        then a hash-based fallback).

    """
    if isinstance(chunk, str):
        return f"chunk-{hash(chunk)}"
    return str(
        getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or f"chunk-{id(chunk)}"
    )


class SyntheticDataset:
    """Generate (question, contexts, answer) triples from a corpus.

    Args:
        corpus: Sequence of objects with a ``.text`` attribute (e.g.
            :class:`raghub.models.Chunk`).
        llm: The generator used to produce questions and answers.
        n_questions: Target number of synthetic examples. When the
            corpus is smaller than ``n_questions``, chunks are
            sampled with replacement.
        seed: Optional seed for reproducibility.
        question_types: Subset of ``{"factual", "multi_hop",
            "comparison", "summary"}``. Currently informational —
            the prompt template is the same for all. Hook for
            future per-type prompt variation.

    """

    QUESTION_PROMPT = (
        "Given the following passage, generate a single question that "
        "can be answered from it. The question should be answerable "
        "using ONLY the passage. Return the question as a single line "
        "with no preamble.\n\n"
        "Passage:\n{passage}\n\n"
        "Question:"
    )

    ANSWER_PROMPT = (
        "Given the passage and the question, write a concise factual "
        "answer in 1-2 sentences. Use ONLY information from the passage. "
        "Return the answer as a single line with no preamble.\n\n"
        "Passage: {passage}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(
        self,
        *,
        corpus: Sequence[Any],
        llm: Generator,
        n_questions: int = 50,
        seed: int | None = None,
        question_types: Sequence[str] = ("factual",),
    ) -> None:
        """Store the corpus and the generator."""
        if not corpus:
            raise ValueError("corpus must be non-empty")
        if n_questions <= 0:
            raise ValueError("n_questions must be positive")
        self.corpus = list(corpus)
        self.llm = llm
        self.n_questions = n_questions
        self.seed = seed
        self.question_types = tuple(question_types)

    async def generate(self) -> list[dict[str, Any]]:
        """Generate the synthetic dataset.

        Returns:
            A list of example dicts, each with ``question``,
            ``answer``, ``contexts`` (list of strings), and
            ``relevant_ids`` (list of chunk ids). The shape matches
            the Finance/Frames canonical schema.

        """
        rng = random.Random(self.seed)
        examples: list[dict[str, Any]] = []

        for _ in range(self.n_questions):
            chunk = rng.choice(self.corpus)
            text = chunk_text(chunk)
            cid = chunk_id(chunk)

            question = await self.generate_question(text)
            answer = await self.generate_answer(text, question)

            examples.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": [text],
                    "relevant_ids": [cid],
                }
            )

        return examples

    async def generate_question(self, text: str) -> str:
        """Run the question-generation prompt and return the cleaned response."""
        prompt = self.QUESTION_PROMPT.format(passage=text)
        response = await self.llm.async_generate(
            GenerationRequest(
                system_prompt=prompt,
                conversation=(),
                context=(),
                question=prompt,
            )
        )
        return clean_response(response)

    async def generate_answer(self, text: str, question: str) -> str:
        """Run the answer-generation prompt and return the cleaned response."""
        prompt = self.ANSWER_PROMPT.format(passage=text, question=question)
        response = await self.llm.async_generate(
            GenerationRequest(
                system_prompt=prompt,
                conversation=(),
                context=(),
                question=prompt,
            )
        )
        return clean_response(response)


def clean_response(text: str) -> str:
    """Strip leading/trailing whitespace and common prompt prefixes.

    Args:
        text: The LLM's raw response.

    Returns:
        The cleaned text with leading ``"Answer:"`` / ``"Question:"``
        prefixes stripped (LLMs often echo the prompt).

    """
    text = (text or "").strip()
    for prefix in ("Answer:", "Question:", "answer:", "question:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


__all__ = ["SyntheticDataset", "chunk_id", "chunk_text", "clean_response"]
