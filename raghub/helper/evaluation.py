"""Evaluation framework for RAGHub.

The whole benchmark-agnostic scoring layer collapses into this
single module. Class summary::

    Metrics       - retrieval-quality primitives (recall@k, precision@k,
                   MRR, context recall/precision, faithfulness, answer
                   correctness) plus the one-stop ``Metrics.evaluate``
                   that runs them all for a single example.
    Scoring       - tiny string-overlap helpers (``jaccard``,
                   ``first_number``) shared across adapters.
    FinanceBench  - the default benchmark adapter.

The :func:`run` harness owns the error envelope around any adapter.

Adding a new benchmark means writing a new ``Foo`` class with an
``async evaluate(examples, *, response_factory)`` method that yields
:class:`raghub.models.EvaluationResult` items. Everything else is
reusable.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from raghub.exceptions import EvaluationError
from raghub.models import Evaluator, EvaluationResult
from raghub.utils import capture

TOKEN_RE = re.compile(r"\w+")


class Metrics:
    """Retrieval and answer-quality metrics used by every evaluator.

    All methods are static; the class exists purely to namespace the
    primitives and to expose the one-stop :meth:`evaluate` bundle.
    """

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """Lower-case word tokens for set-based overlap metrics."""
        return set(t.lower() for t in TOKEN_RE.findall(text or ""))

    @staticmethod
    def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
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
    def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
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
    def mean_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
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
        ground_truth: str = "",
        question: str = "",
        k: int = 5,
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

        Returns:
            A dict of metric name → value.
        """
        metrics: dict[str, float] = {
            f"recall_at_{k}": Metrics.recall_at_k(retrieved_ids, relevant_ids, k),
            f"precision_at_{k}": Metrics.precision_at_k(retrieved_ids, relevant_ids, k),
            "mrr": Metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            "context_recall": Metrics.context_recall(answer, contexts),
            "context_precision": Metrics.context_precision(question, contexts),
            "faithfulness": Metrics.faithfulness(answer, contexts),
        }
        if ground_truth:
            metrics["answer_correctness"] = Metrics.answer_correctness(answer, ground_truth)
        return metrics


class Scoring:
    """Tiny string-overlap helpers shared by adapters."""

    @staticmethod
    def jaccard(predicted: str, expected: str) -> float:
        """Token-overlap (Jaccard) score.

        Args:
            predicted: Model output.
            expected: Ground truth.

        Returns:
            A score in ``[0, 1]``.
        """
        pred_tokens = set(predicted.lower().split())
        exp_tokens = set(expected.lower().split())
        if not exp_tokens:
            return 1.0 if not pred_tokens else 0.0
        union = pred_tokens | exp_tokens
        if not union:
            return 0.0
        return len(pred_tokens & exp_tokens) / len(union)

    @staticmethod
    def first_number(text: str) -> str:
        """Return the first whitespace-delimited token that parses as a number.

        Args:
            text: Arbitrary text.

        Returns:
            The first numeric token as a string. Empty if none.
        """
        for token in text.replace(",", "").split():
            parsed, _ = capture(float, token)
            if isinstance(parsed, float):
                return token
        return ""


class FinanceBench(Evaluator):
    """The default RAGHub benchmark adapter.

    Loads the dataset from a local JSONL/JSON file when supplied, or
    falls back to the HuggingFace Hub (when ``datasets`` is installed)
    with caching under ``~/.cache/raghub/financebench/``.

    Attributes:
        benchmark: The benchmark identifier persisted on every result.
    """

    benchmark: str = "financebench"

    DEFAULT_NAME = "PatronusAI/financebench"
    DEFAULT_SPLIT = "train"
    CACHE_DIR = Path(
        os.getenv(
            "RAGHUB_FINANCEBENCH_CACHE",
            str(Path.home() / ".cache" / "raghub" / "financebench"),
        )
    )

    def __init__(
        self,
        *,
        dataset_path: Path | None = None,
        dataset_name: str = DEFAULT_NAME,
        split: str = DEFAULT_SPLIT,
        tolerance: float = 0.05,
    ) -> None:
        """Initialise the evaluator.

        Args:
            dataset_path: Optional local file (JSONL/JSON). When set,
                takes precedence over the HuggingFace dataset.
            dataset_name: HuggingFace dataset id.
            split: Dataset split.
            tolerance: Relative tolerance for numeric answers
                (``abs(pred - gold) / max(|gold|, 1)``).
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.tolerance = tolerance
        self.examples: list[dict[str, Any]] | None = None

    def ensure_examples(self) -> list[dict[str, Any]]:
        """Load the FinanceBench dataset from local cache or HuggingFace.

        The dataset is cached in ``~/.cache/raghub/financebench/``
        after the first download.

        Returns:
            A list of example dicts, each with ``question``,
            ``answer``, and optional ``relevant_ids`` fields.
        """
        if self.examples is not None:
            return self.examples
        if self.dataset_path is not None:
            self.examples = FinanceBench.load_jsonl(Path(self.dataset_path))
            if self.examples:
                return self.examples
        FinanceBench.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = FinanceBench.CACHE_DIR / "financebench.jsonl"
        if not cached.exists():
            self.examples = FinanceBench.load_huggingface(self.dataset_name, self.split)
            cached.write_text(
                "\n".join(json.dumps(ex) for ex in self.examples),
                encoding="utf-8",
            )
        else:
            self.examples = FinanceBench.load_jsonl(cached)
        return self.examples

    async def evaluate(
        self,
        examples: Sequence[dict[str, Any]] | None = None,
        *,
        response_factory: Any,
    ) -> list[EvaluationResult]:
        """Score every example.

        Args:
            examples: Optional explicit examples; defaults to the
                benchmark's own dataset.
            response_factory: Async callable taking an example dict
                and returning the model's answer. The callable may
                return either a plain string or a
                ``(answer, contexts, retrieved_ids, relevant_ids)``
                tuple; the latter enables the full retrieval-quality
                metrics. When the simple form is returned, only
                token-overlap and numeric scores are computed.

        Returns:
            A list of :class:`EvaluationResult`.
        """
        rows = list(examples) if examples is not None else self.ensure_examples()
        results: list[EvaluationResult] = []
        for idx, example in enumerate(rows):
            question = example.get("question") or example.get("query") or ""
            gold = example.get("answer") or example.get("evidence_text") or ""
            out = await response_factory(example)
            contexts: list[str] = []
            retrieved_ids: list[str] = []
            relevant_ids: list[str] = list(example.get("relevant_ids", [])) or [
                str(example.get("id", idx))
            ]
            predicted: object
            if isinstance(out, tuple) and len(out) == 4:
                predicted, contexts, retrieved_ids, relevant_ids = out
            else:
                predicted = out
            overlap = Scoring.jaccard(str(predicted), str(gold))
            numeric = self.numeric_within_tolerance(str(predicted), str(gold))
            metrics = {"token_overlap": overlap, "numeric_within_tolerance": numeric}
            # Add retrieval-quality metrics when the response
            # factory returned the tuple form.
            if contexts is not None and retrieved_ids is not None:
                retrieval_metrics = Metrics.evaluate(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    answer=str(predicted),
                    contexts=contexts,
                    ground_truth=str(gold),
                    question=question,
                )
                metrics.update(retrieval_metrics)
            results.append(
                EvaluationResult(
                    benchmark=self.benchmark,
                    example_id=str(example.get("id", idx)),
                    metrics=metrics,
                    passed=numeric >= 0.99 or overlap >= 0.6,
                    details={
                        "question": question,
                        "gold": str(gold),
                        "predicted": str(predicted),
                    },
                )
            )
        return results

    def numeric_within_tolerance(self, predicted: str, gold: str) -> float:
        """Return 1.0 if the predicted number is within tolerance of gold.

        Args:
            predicted: Predicted string.
            gold: Gold string.

        Returns:
            ``1.0`` when within tolerance, ``0.0`` otherwise.
        """
        p_raw = Scoring.first_number(predicted)
        g_raw = Scoring.first_number(gold)
        p_parsed, _ = capture(float, p_raw) if p_raw else (None, None)
        g_parsed, _ = capture(float, g_raw) if g_raw else (None, None)
        if not isinstance(p_parsed, (int, float)) or not isinstance(g_parsed, (int, float)):
            return 0.0
        p, g = p_parsed, g_parsed
        if g == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(p - g) / max(abs(g), 1.0) <= self.tolerance else 0.0

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load FinanceBench examples from a local JSONL/JSON file.

        Args:
            path: Path to a JSON or JSONL file.

        Returns:
            A list of example dicts.
        """
        if not path.exists():
            return []
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [raw] if isinstance(raw, dict) else raw

    @staticmethod
    def load_huggingface(dataset_name: str, split: str) -> list[dict[str, Any]]:
        """Load FinanceBench from the HuggingFace Hub.

        Args:
            dataset_name: Hub dataset id.
            split: Dataset split.

        Returns:
            A list of example dicts.

        Raises:
            EvaluationError: When the dataset cannot be loaded.
        """
        try:
            datasets_module = import_module("datasets")
        except Exception as exc:  # pragma: no cover - optional dep
            raise EvaluationError(
                "datasets is not installed; install it via "
                "`pip install datasets` or place a JSONL/JSON file at "
                f"{FinanceBench.CACHE_DIR / 'financebench.jsonl'}."
            ) from exc
        ds, error = capture(datasets_module.load_dataset, dataset_name, split=split)
        if error is not None or ds is None:
            raise EvaluationError(
                f"Failed to load FinanceBench from {dataset_name!r}: {error}"
            ) from error
        return [dict(record) for record in ds]


async def run(
    evaluator: Evaluator,
    examples: Sequence[dict[str, Any]],
    response_factory: Any,
) -> list[EvaluationResult]:
    """Run ``evaluator`` on ``examples`` with a shared error envelope.

    Args:
        evaluator: The benchmark-specific evaluator.
        examples: Per-example records.
        response_factory: Async callable returning the model's answer.

    Returns:
        A list of :class:`EvaluationResult` objects.

    Raises:
        EvaluationError: When the evaluator raises unexpectedly.
    """
    try:
        return await evaluator.evaluate(examples, response_factory=response_factory)
    except EvaluationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive envelope
        raise EvaluationError(f"Evaluator {evaluator.benchmark!r} failed: {exc}") from exc


__all__ = [
    "Metrics",
    "Scoring",
    "FinanceBench",
    "run",
]
