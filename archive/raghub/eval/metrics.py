"""Retrieval and answer-quality metrics.

This module is the home of every benchmark-agnostic scoring
primitive used by RAGHub evaluators.

Class summary::

    Metrics       - retrieval-quality primitives (recall@k,
                   precision@k, MRR, context recall/precision,
                   faithfulness, answer correctness) plus the
                   one-stop ``Metrics.evaluate`` that runs them
                   all for a single example.
    Scoring       - tiny string-overlap helpers (``jaccard``,
                   ``first_number``) shared across adapters.

.. seealso:: :mod:`raghub.eval.scoring`
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import cast

from raghub.runtime import capture
from raghub.types import JSONValue

from raghub.eval.scoring import Scoring  # re-export for backward compat

TOKEN_RE = re.compile(r"\w+")

MIN_SENTENCES_FOR_COHERENCE = 2


class Metrics:
    """Retrieval and answer-quality metrics used by every evaluator.

    All methods are static; the class exists purely to namespace the
    primitives and to expose the one-stop :meth:`evaluate` bundle.
    """

    STOPWORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "this",
            "to",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }
    )
    CLAIM_SPLITTER = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """Lower-case word tokens for set-based overlap metrics."""
        return set(t.lower() for t in TOKEN_RE.findall(text or ""))

    @staticmethod
    def recall(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
        """Recall@K — fraction of relevant items in the top-k.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            k: Cutoff.

        Returns:
            A value in ``[0, 1]``. ``1.0`` when there are no relevant
            items; ``0.0`` when none of the relevant items appear in
            the top-k.

        """
        relevant = set(relevant_ids)
        if not relevant:
            return 1.0
        top = set(retrieved_ids[:k])
        return len(relevant & top) / len(relevant)

    @staticmethod
    def precision(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
        """Precision@K — fraction of the top-k that is relevant.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            k: Cutoff.

        Returns:
            A value in ``[0, 1]``. ``0.0`` when the top-k is empty.

        """
        relevant = set(relevant_ids)
        if k <= 0:
            return 0.0
        top = retrieved_ids[:k]
        if not top:
            return 0.0
        return sum(1 for r in top if r in relevant) / k

    @staticmethod
    def f1_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
        """F1@K — harmonic mean of precision@k and recall@k.

        Returns ``0.0`` when either precision or recall is zero
        (the harmonic mean collapses to zero); the underlying
        :meth:`precision` / :meth:`recall` methods handle
        degenerate edge cases (empty top-k, empty relevant set).

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            k: Cutoff.

        Returns:
            A value in ``[0.0, 1.0]``.

        """
        precision = Metrics.precision(retrieved_ids, relevant_ids, k)
        recall = Metrics.recall(retrieved_ids, relevant_ids, k)
        if math.isclose(precision, 0.0) and math.isclose(recall, 0.0):
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def mrr(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
        """Mean Reciprocal Rank (MRR) — 1 / rank of first relevant hit.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.

        Returns:
            A value in ``[0, 1]``. ``0.0`` when no relevant item is
            found; ``1.0`` when the first hit is relevant.

        """
        relevant = set(relevant_ids)
        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def hit_rate(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
        """Hit rate@K — ``1.0`` if any retrieved item is in the relevant set.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            k: Cutoff.

        Returns:
            A value in ``[0, 1]``. ``1.0`` when the relevant set is
            empty (vacuously retrieved) or at least one relevant item
            is in the top-k; otherwise ``0.0``.

        """
        relevant = set(relevant_ids)
        if not relevant:
            return 1.0
        top = set(retrieved_ids[:k])
        return 1.0 if any(rid in relevant for rid in top) else 0.0

    @staticmethod
    def map(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
        """Mean Average Precision (MAP) for a single query.

        Computes the AP for a single ranked list:

            AP = (1 / |relevant|) * sum_{k=1..N} Precision@k * rel(k)

        where ``rel(k)`` is 1 if the k-th retrieved item is in
        ``relevant_ids`` and 0 otherwise.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.

        Returns:
            A value in ``[0, 1]``. ``0.0`` when no relevant ids are
            present (or none of the top-k items are relevant).

        """
        relevant = set(relevant_ids)
        if not relevant:
            return 0.0
        score = 0.0
        hits = 0
        for k, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant:
                hits += 1
                score += hits / k
        return score / len(relevant)

    @staticmethod
    def answer_relevance(predicted: str, question: str) -> float:
        """Deterministic answer-relevance heuristic (Anyscale).

        Computes the Jaccard overlap between the lower-cased token
        set of ``predicted`` and the question's content words. Skips
        common stopwords so an answer like "I don't know" doesn't
        score high just because it shares words with the question.

        Args:
            predicted: The generated answer.
            question: The user's question.

        Returns:
            A value in ``[0, 1]``. ``1.0`` if every predicted token
            appears in the question (or one of them is empty);
            ``0.0`` if the sets are disjoint.

        """
        pred = {t for t in Metrics.tokenize(predicted) if t not in Metrics.STOPWORDS}
        q = {t for t in Metrics.tokenize(question) if t not in Metrics.STOPWORDS}
        if not pred or not q:
            return 0.0
        return len(pred & q) / len(pred | q)

    CLAIM_SPLITTER = re.compile(r"(?<=[.!?])\s+")

    @staticmethod
    def faithfulness_claims(answer: str, contexts: Sequence[str]) -> float:
        """Deterministic faithfulness via claim-substring check.

        Splits ``answer`` into sentence-level claims and counts the
        fraction whose longest non-stopword n-gram also appears
        somewhere in ``contexts``. A claim without any content word
        (e.g. "Yes.") is ignored. Empty ``answer`` is reported as
        ``1.0`` (no unsupported claim); empty ``contexts`` is ``0.0``
        (no evidence to support anything).

        Args:
            answer: The generated answer.
            contexts: Sequence of retrieved context strings.

        Returns:
            A value in ``[0, 1]``. ``1.0`` when every claim is
            supported; ``0.0`` when no claim is supported.

        """
        text = (answer or "").strip()
        if not text:
            return 1.0
        if not contexts:
            return 0.0
        ctx_tokens = Metrics.context_token_union(contexts)
        claims = Metrics.split_claims(text)
        supported, considered = Metrics.count_supported(claims, ctx_tokens)
        if considered == 0:
            return 1.0
        return supported / considered

    @staticmethod
    def context_token_union(contexts: Sequence[str]) -> set[str]:
        """Return the union of tokens across ``contexts``."""
        tokens: set[str] = set()
        for context in contexts:
            tokens |= Metrics.tokenize(context or "")
        return tokens

    @staticmethod
    def split_claims(text: str) -> list[str]:
        """Split ``text`` into sentence-level claim strings."""
        return [claim.strip() for claim in Metrics.CLAIM_SPLITTER.split(text) if claim.strip()]

    @staticmethod
    def count_supported(claims: Sequence[str], ctx_tokens: set[str]) -> tuple[int, int]:
        """Count (supported, considered) claims against ``ctx_tokens``."""
        supported = 0
        considered = 0
        for claim in claims:
            tokens = {t for t in Metrics.tokenize(claim) if t not in Metrics.STOPWORDS}
            if not tokens:
                continue
            considered += 1
            # The claim is "supported" if any of its non-stopword tokens
            # appears in the union of retrieved contexts. We avoid the
            # brittle "all tokens must be present" check.
            if tokens & ctx_tokens:
                supported += 1
        return supported, considered

    @staticmethod
    def context_recall(answer: str, contexts: Sequence[str]) -> float:
        """Fraction of answer tokens present in the retrieved context.

        Args:
            answer: The generated answer.
            contexts: Sequence of retrieved context strings.

        Returns:
            A value in ``[0, 1]``. ``0.0`` when the answer is empty;
            ``1.0`` when the context is empty.

        """
        answer_tokens = Metrics.tokenize(answer)
        if not answer_tokens:
            return 0.0
        context_tokens: set[str] = set()
        for c in contexts:
            context_tokens |= Metrics.tokenize(c)
        if not context_tokens:
            return 1.0
        return len(answer_tokens & context_tokens) / len(answer_tokens)

    @staticmethod
    def context_precision(question: str, contexts: Sequence[str]) -> float:
        """Fraction of retrieved context relevant to the question.

        Args:
            question: The user's question.
            contexts: Sequence of retrieved context strings.

        Returns:
            A value in ``[0, 1]``.

        """
        question_tokens = Metrics.tokenize(question)
        if not contexts:
            return 0.0
        if not question_tokens:
            return 0.0
        total = 0
        matched = 0
        for c in contexts:
            toks = Metrics.tokenize(c)
            if not toks:
                continue
            total += len(toks)
            matched += len(toks & question_tokens)
        if total == 0:
            return 0.0
        return matched / total

    @staticmethod
    def completeness(answer: str, contexts: Sequence[str]) -> float:
        """Fraction of context tokens that appear in the answer.

        Inverse of :meth:`context_recall` in spirit: rewards the
        answer for using the retrieved evidence. Empty answer
        returns ``0.0`` (no evidence used); empty context returns
        ``1.0`` (vacuously complete).

        Args:
            answer: The generated answer.
            contexts: Sequence of retrieved context strings.

        Returns:
            A value in ``[0.0, 1.0]``.

        """
        answer_tokens = Metrics.tokenize(answer)
        context_tokens: set[str] = set()
        for c in contexts:
            context_tokens |= Metrics.tokenize(c)
        if not context_tokens:
            return 1.0
        if not answer_tokens:
            return 0.0
        return len(answer_tokens & context_tokens) / len(context_tokens)

    @staticmethod
    def coherence(text: str) -> float:
        """Sentence-level coherence proxy.

        Splits ``text`` on sentence-ending punctuation and measures
        the fraction of consecutive sentence pairs that share at
        least one content token (topical continuity). Empty text
        returns ``0.0``; a single sentence returns ``0.5`` (neutral
        score — no internal incoherence but no evidence of
        continuity either).

        Note: this is a deterministic proxy. Higher-fidelity
        coherence requires an LLM-as-a-judge (see :class:`Judge`).

        Args:
            text: The generated answer.

        Returns:
            A value in ``[0.0, 1.0]``.

        """
        text = (text or "").strip()
        if not text:
            return 0.0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < MIN_SENTENCES_FOR_COHERENCE:
            return 0.5
        from itertools import pairwise

        shared = 0
        for prev, curr in pairwise(sentences):
            prev_tokens = Metrics.tokenize(prev)
            curr_tokens = Metrics.tokenize(curr)
            if prev_tokens and curr_tokens and (prev_tokens & curr_tokens):
                shared += 1
        return shared / (len(sentences) - 1)

    @staticmethod
    def faithfulness(answer: str, contexts: Sequence[str]) -> float:
        """Fraction of answer tokens grounded in the retrieved context."""
        return Metrics.context_recall(answer, contexts)

    @staticmethod
    def answer_correctness(answer: str, ground_truth: str) -> float:
        """Jaccard overlap between answer and ground-truth tokens.

        Args:
            answer: The generated answer.
            ground_truth: The reference answer.

        Returns:
            A value in ``[0, 1]``.

        """
        a = Metrics.tokenize(answer)
        g = Metrics.tokenize(ground_truth)
        if not a and not g:
            return 1.0
        if not a or not g:
            return 0.0
        return len(a & g) / len(a | g)

    @staticmethod
    def evaluate(
        *,
        retrieved_ids: Sequence[str],
        relevant_ids: Sequence[str],
        answer: str,
        contexts: Sequence[str],
        **options: JSONValue,
    ) -> dict[str, float]:
        """Compute every retrieval and answer metric for a single example.

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            answer: The generated answer.
            contexts: The retrieved context strings.
            ground_truth: The reference answer (optional).
            question: The question (optional, for context precision).
            k: Cutoff for @K metrics.
            **options: Reserved for future overrides.

        Returns:
            A dict of metric name → value.

        """
        ground_truth: str = cast(str, options.get("ground_truth", ""))
        question: str = cast(str, options.get("question", ""))
        k: int = cast(int, options.get("k", 5))
        metrics: dict[str, float] = {
            f"recall_at_{k}": Metrics.recall(retrieved_ids, relevant_ids, k),
            f"precision_at_{k}": Metrics.precision(retrieved_ids, relevant_ids, k),
            f"f1_at_{k}": Metrics.f1_at_k(retrieved_ids, relevant_ids, k),
            f"hit_rate_at_{k}": Metrics.hit_rate(retrieved_ids, relevant_ids, k),
            "mrr": Metrics.mrr(retrieved_ids, relevant_ids),
            "map": Metrics.map(retrieved_ids, relevant_ids),
            "context_recall": Metrics.context_recall(answer, contexts),
            "context_precision": Metrics.context_precision(question, contexts),
            "completeness": Metrics.completeness(answer, contexts),
            "coherence": Metrics.coherence(answer),
            "faithfulness": Metrics.faithfulness(answer, contexts),
            "faithfulness_claims": Metrics.faithfulness_claims(answer, contexts),
            "answer_relevance": Metrics.answer_relevance(answer, question),
        }
        if ground_truth:
            metrics["answer_correctness"] = Metrics.answer_correctness(answer, ground_truth)
        return metrics

    @staticmethod
    def within_tolerance(predicted: str, gold: str, tolerance: float = 0.05) -> float:
        """Return 1.0 when the first number in ``predicted`` is within tolerance of gold.

        Args:
            predicted: Model output.
            gold: Ground truth.
            tolerance: Relative tolerance
                (``abs(pred - gold) / max(abs(gold), 1)``).

        Returns:
            ``1.0`` when a number is found in both and it is within
            tolerance; ``0.0`` otherwise (including unparseable input).

        """
        p_raw = Scoring.first_number(predicted)
        g_raw = Scoring.first_number(gold)
        p_parsed, _ = capture(float, p_raw) if p_raw else (None, None)
        g_parsed, _ = capture(float, g_raw) if g_raw else (None, None)
        if not isinstance(p_parsed, int | float) or not isinstance(g_parsed, int | float):
            return 0.0
        p, g = p_parsed, g_parsed
        if g == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(p - g) / max(abs(g), 1.0) <= tolerance else 0.0


__all__ = ["Metrics", "Scoring"]
