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

import asyncio
import json
import os
import re
from collections.abc import Iterable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from raghub.errors import ConfigurationError, EvaluationError
from raghub.models import EvaluationResult, Evaluator
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
    def f1_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
        """F1@K — harmonic mean of precision@k and recall@k.

        Returns ``0.0`` when either precision or recall is zero
        (the harmonic mean collapses to zero); the underlying
        :meth:`precision_at_k` / :meth:`recall_at_k` methods handle
        degenerate edge cases (empty top-k, empty relevant set).

        Args:
            retrieved_ids: Ordered list of retrieved item ids.
            relevant_ids: Iterable of ids considered relevant.
            k: Cutoff.

        Returns:
            A value in ``[0.0, 1.0]``.
        """
        precision = Metrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = Metrics.recall_at_k(retrieved_ids, relevant_ids, k)
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

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
    def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
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
    def mean_average_precision(
        retrieved_ids: Sequence[str], relevant_ids: Iterable[str]
    ) -> float:
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
        STOPWORDS = frozenset(
            {
                "a", "an", "and", "are", "as", "at", "be", "by",
                "for", "from", "how", "i", "in", "is", "it", "of",
                "on", "or", "that", "the", "this", "to", "was",
                "what", "when", "where", "which", "who", "why", "with",
            }
        )
        pred = {t for t in Metrics.tokenize(predicted) if t not in STOPWORDS}
        q = {t for t in Metrics.tokenize(question) if t not in STOPWORDS}
        if not pred or not q:
            return 0.0
        return len(pred & q) / len(pred | q)

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
        STOPWORDS = frozenset(
            {
                "a", "an", "and", "are", "as", "at", "be", "by",
                "for", "from", "how", "i", "in", "is", "it", "of",
                "on", "or", "that", "the", "this", "to", "was",
                "what", "when", "where", "which", "who", "why", "with",
            }
        )
        text = (answer or "").strip()
        if not text:
            return 1.0
        if not contexts:
            return 0.0
        ctx_tokens = set()
        for c in contexts:
            ctx_tokens |= Metrics.tokenize(c or "")
        # Naive sentence splitter on .!? followed by whitespace.
        claims = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
        supported = 0
        considered = 0
        for claim in claims:
            tokens = {t for t in Metrics.tokenize(claim) if t not in STOPWORDS}
            if not tokens:
                continue
            considered += 1
            # The claim is "supported" if any of its non-stopword tokens
            # appears in the union of retrieved contexts. We avoid the
            # brittle "all tokens must be present" check.
            if tokens & ctx_tokens:
                supported += 1
        if considered == 0:
            return 1.0
        return supported / considered

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
        coherence requires an LLM-as-a-judge (see :class:`LlmJudge`).

        Args:
            text: The generated answer.

        Returns:
            A value in ``[0.0, 1.0]``.
        """
        text = (text or "").strip()
        if not text:
            return 0.0
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) < 2:
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
            f"f1_at_{k}": Metrics.f1_at_k(retrieved_ids, relevant_ids, k),
            f"hit_rate_at_{k}": Metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k),
            "mrr": Metrics.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            "map": Metrics.mean_average_precision(retrieved_ids, relevant_ids),
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
        # Run all factory calls concurrently. The factory must be
        # **stateless** (no shared per-call state) so concurrent calls
        # don't race. We rely on the per-row metric computation below
        # being independent of the call order.
        outs = await asyncio.gather(
            *(response_factory(example) for example in rows)
        )
        results: list[EvaluationResult] = []
        for idx, (example, out) in enumerate(zip(rows, outs, strict=True)):
            question = example.get("question") or example.get("query") or ""
            gold = example.get("answer") or example.get("evidence_text") or ""
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
            numeric = Metrics.within_tolerance(str(predicted), str(gold))
            metrics = {"token_overlap": overlap, "within_tolerance": numeric}
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


class FramesBenchmark(Evaluator):
    """FRAMES — multi-hop RAG benchmark (Krishna et al. 2024).

    Loads the 824-question FRAMES test split from
    ``huggingface.co/datasets/google/frames-benchmark``. The dataset
    is multi-hop: each question references 2-15 Wikipedia articles
    (the ``wiki_links`` field). The benchmark therefore exercises
    retrieval quality (recall@k, hit_rate@k, MRR, MAP) in addition
    to answer correctness.

    Attributes:
        benchmark: The benchmark identifier persisted on every result.
    """

    benchmark: str = "frames"

    DEFAULT_NAME = "google/frames-benchmark"
    DEFAULT_SPLIT = "test"
    CACHE_DIR = Path(
        os.getenv(
            "RAGHUB_FRAMES_CACHE",
            str(Path.home() / ".cache" / "raghub" / "frames"),
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
            dataset_path: Optional local file (TSV/CSV/JSONL). When set,
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
        """Load the FRAMES test split from local cache or HuggingFace.

        The dataset is cached at
        ``~/.cache/raghub/frames/frames.jsonl`` after the first
        download. Each row is normalised to the canonical
        ``Evaluator`` schema: ``question`` (from ``Prompt``),
        ``answer`` (from ``Answer``), ``wiki_links`` (a Python
        list of Wikipedia URLs), ``reasoning_types`` (e.g.
        ``"Multiple constraints | Numerical reasoning"``), and
        ``id`` (the row index).

        Returns:
            A list of example dicts, one per FRAMES question.
        """
        if self.examples is not None:
            return self.examples
        if self.dataset_path is not None:
            self.examples = self.load_local(Path(self.dataset_path))
            if self.examples:
                return self.examples
        FramesBenchmark.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = FramesBenchmark.CACHE_DIR / f"{self.split}.jsonl"
        if not cached.exists():
            self.examples = self.load_huggingface(self.dataset_name, self.split)
            cached.write_text(
                "\n".join(json.dumps(ex) for ex in self.examples),
                encoding="utf-8",
            )
        else:
            self.examples = self.load_local(cached)
        return self.examples

    def load_local(self, path: Path) -> list[dict[str, Any]]:
        """Read a FRAMES-style file (TSV/CSV/JSONL) from disk."""
        if not path.exists():
            return []
        if path.suffix.lower() == ".jsonl":
            return [
                self.normalise_row(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if path.suffix.lower() in {".tsv", ".csv"}:
            rows = self.load_tsv(path)
            return [self.normalise_row(r) for r in rows]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [self.normalise_row(r) for r in data]
        return [self.normalise_row(data)]

    def load_tsv(self, path: Path) -> list[dict[str, Any]]:
        """Parse a TSV / CSV file with the FRAMES schema."""
        try:
            import csv
        except ImportError:  # pragma: no cover - stdlib
            return []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ",")
            return [row for row in reader]

    def normalise_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Map a FRAMES row to the canonical ``Evaluator`` schema."""
        raw_links = row.get("wiki_links")
        if raw_links is None:
            # Fall back to the per-column Wikipedia columns.
            per_col = [
                row.get(f"wikipedia_link_{i}")
                for i in range(1, 11)
            ]
            per_col.append(row.get("wikipedia_link_11+"))
            links = [u for u in per_col if u]
        elif isinstance(raw_links, str):
            # The HF TSV stores the list as a Python ``repr()`` of
            # the list. Try ``ast.literal_eval`` (the cheap path)
            # first, fall back to a JSON parse for the edge case
            # where someone hands us a real JSON-encoded list.
            import ast
            try:
                links = ast.literal_eval(raw_links)
            except (ValueError, SyntaxError):
                try:
                    links = json.loads(raw_links)
                except json.JSONDecodeError:
                    links = []
        elif isinstance(raw_links, list):
            links = raw_links
        else:
            links = []
        return {
            "id": row.get("Unnamed: 0", row.get("id")),
            "question": row.get("Prompt") or row.get("question") or "",
            "answer": row.get("Answer") or row.get("answer") or "",
            "wiki_links": [str(u) for u in links if isinstance(u, str)],
            "reasoning_types": row.get("reasoning_types", ""),
            "company": row.get("company", ""),
        }

    def within_tolerance(self, predicted: str, gold: str) -> float:
        """Return 1.0 if the predicted number is within tolerance of gold.

        For FRAMES many gold answers contain a single token
        (e.g. ``"37th"``). We try :class:`Metrics.first_number` for
        both, fall back to Jaccard on the digits-only version of
        the strings, and otherwise use the binary-within-tolerance
        check.
        """
        p_raw = Scoring.first_number(predicted)
        g_raw = Scoring.first_number(gold)
        p_parsed, _ = capture(float, p_raw) if p_raw else (None, None)
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


# ---------------------------------------------------------------------------
# LLM-as-a-judge
# ---------------------------------------------------------------------------


SCORE_RE = re.compile(r"(-?)([0-1](?:\.\d+)?|0\.\d+)(?![0-9])")


def parse_score(text: str) -> float | None:
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


class LlmJudge:
    """LLM-as-judge scorer for faithfulness and answer relevance.

    Wraps a :class:`raghub.llm.Generator` and uses prompt templates to
    score a ``(question, answer, contexts)`` triple on a 0-1 scale. The
    judge LLM is expected to reply with a single number; the response
    is parsed by :func:`parse_score` and clamped to ``[0.0, 1.0]``.

    Args:
        llm: The generator used as the judge. Note: the same LLM
            used for answer generation is acceptable, but a stronger
            model (e.g. GPT-4o for a GPT-3.5-turbo pipeline) reduces
            self-bias.
        max_retries: Number of retries on parse failure before
            returning 0.0. Defaults to 1 retry (2 attempts total).
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
                system_prompt=prompt,
                conversation=(),
                context=(),
                question=prompt,
            )
        except Exception:
            return None
        return parse_score(response)

    async def score(self, prompt_template: str, **kwargs: str) -> float:
        """Run a prompt with retries; return 0.0 if all attempts fail to parse."""
        for _ in range(self.max_retries + 1):
            value = await self.score_once(prompt_template, **kwargs)
            if value is not None:
                return value
        return 0.0

    async def faithfulness(
        self, answer: str, contexts: Sequence[str]
    ) -> float:
        """Score the answer's faithfulness on a 0-1 scale.

        Args:
            answer: The generated answer.
            contexts: The retrieved context strings; joined with
                ``\\n\\n---\\n\\n`` before being inserted into the prompt.

        Returns:
            A score in ``[0.0, 1.0]``. Returns ``0.0`` when every
            retry fails to parse.
        """
        joined = "\n\n---\n\n".join(contexts)
        return await self.score(
            self.FAITHFULNESS_PROMPT, answer=answer, contexts=joined
        )

    async def answer_relevance(self, answer: str, question: str) -> float:
        """Score the answer's relevance to the question on a 0-1 scale.

        Args:
            answer: The generated answer.
            question: The user's question.

        Returns:
            A score in ``[0.0, 1.0]``. Returns ``0.0`` when every
            retry fails to parse.
        """
        return await self.score(
            self.RELEVANCE_PROMPT, answer=answer, question=question
        )


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


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


class QualityGate:
    """Threshold checker for a metrics dict.

    Each threshold is either a "minimum" (``mode="min"``, the metric
    must be ``>= threshold``) or a "maximum" (``mode="max"``, the
    metric must be ``<= threshold``). :meth:`check` raises
    :class:`ConfigurationError` if any metric breaches its threshold
    or is missing; :meth:`report` returns a structured summary
    suitable for logging or CI output.

    Args:
        thresholds: Optional initial mapping of metric name → minimum
            threshold. Use :meth:`add` to set per-metric mode.
        default_mode: Default mode for entries added via the
            constructor. Use ``"min"`` for quality metrics (higher
            is better) and ``"max"`` for cost metrics (lower is
            better).

    >>> gate = QualityGate({"recall_at_5": 0.7, "faithfulness": 0.8})
    >>> gate.check({"recall_at_5": 0.9, "faithfulness": 0.95})
    >>> gate.check({"recall_at_5": 0.5, "faithfulness": 0.95})  # raises
    """

    VALID_MODES = ("min", "max")

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        *,
        default_mode: str = "min",
    ) -> None:
        """Store the thresholds and the default mode."""
        if default_mode not in self.VALID_MODES:
            raise ConfigurationError(
                f"QualityGate default_mode must be 'min' or 'max', "
                f"got {default_mode!r}"
            )
        self.default_mode = default_mode
        self.thresholds: dict[str, tuple[float, str]] = {}
        if thresholds:
            for name, value in thresholds.items():
                self.add(name, value)

    def add(
        self,
        metric: str,
        threshold: float,
        *,
        mode: str | None = None,
    ) -> QualityGate:
        """Add or replace a threshold. Returns self for chaining."""
        chosen_mode = mode or self.default_mode
        if chosen_mode not in self.VALID_MODES:
            raise ConfigurationError(
                f"QualityGate mode for {metric!r} must be 'min' or 'max', "
                f"got {chosen_mode!r}"
            )
        self.thresholds[metric] = (threshold, chosen_mode)
        return self

    def check(self, metrics: dict[str, float]) -> None:
        """Raise :class:`ConfigurationError` if any metric breaches its threshold.

        Args:
            metrics: Per-metric value mapping (as returned by
                :meth:`Metrics.evaluate`).

        Raises:
            ConfigurationError: When at least one metric is missing
                or out of bounds.
        """
        breaches: list[str] = []
        for name, (threshold, mode) in self.thresholds.items():
            value = metrics.get(name)
            if value is None:
                breaches.append(f"{name}: missing (threshold: {threshold})")
                continue
            if mode == "min" and value < threshold:
                breaches.append(f"{name}: {value:.3f} < {threshold}")
            elif mode == "max" and value > threshold:
                breaches.append(f"{name}: {value:.3f} > {threshold}")
        if breaches:
            raise ConfigurationError(
                f"QualityGate failed: {'; '.join(breaches)}"
            )

    def report(
        self, metrics: dict[str, float]
    ) -> dict[str, tuple[float | None, float, bool, str]]:
        """Return a structured per-metric report (no raising).

        Args:
            metrics: Per-metric value mapping.

        Returns:
            A dict of metric name → ``(value, threshold, passed, mode)``
            tuples. ``value`` is ``None`` when the metric is missing.
        """
        result: dict[str, tuple[float | None, float, bool, str]] = {}
        for name, (threshold, mode) in self.thresholds.items():
            value = metrics.get(name)
            if value is None:
                passed = False
            elif mode == "min":
                passed = value >= threshold
            else:
                passed = value <= threshold
            result[name] = (value, threshold, passed, mode)
        return result


# ---------------------------------------------------------------------------
# A/B testing
# ---------------------------------------------------------------------------


async def ab_test(
    *,
    rag_a: Any,
    rag_b: Any,
    examples: list[dict[str, Any]],
    evaluator: Evaluator,
    gate: QualityGate | None = None,
) -> dict[str, Any]:
    """Run two RAG instances against the same dataset, report per-metric diffs.

    Args:
        rag_a: The "control" RAG instance.
        rag_b: The "treatment" RAG instance.
        examples: Per-example records with ``question`` (and any
            other keys the evaluator expects).
        evaluator: The evaluator to score both runs.
        gate: Optional :class:`QualityGate`. When set, the run fails
            when either RAG's metrics breach the gate's thresholds.

    Returns:
        A dict with keys:
        - ``a_metrics``: per-metric averages for rag_a.
        - ``b_metrics``: per-metric averages for rag_b.
        - ``metric_diffs``: ``b - a`` for each metric.
        - ``winner``: ``"a"``, ``"b"``, or ``"tie"``.
        - ``gate_passed``: ``True`` when no gate was supplied; when
            a gate was supplied, ``True`` when both A and B passed.

    Raises:
        ConfigurationError: When a gate is supplied and either RAG's
            metrics breach it.
    """
    async def factory_a(ex: dict[str, Any]) -> Any:
        return await rag_a.aquery(ex["question"])

    async def factory_b(ex: dict[str, Any]) -> Any:
        return await rag_b.aquery(ex["question"])

    results_a = await run(evaluator, examples, response_factory=factory_a)
    results_b = await run(evaluator, examples, response_factory=factory_b)

    metrics_a = aggregate_metrics(results_a)
    metrics_b = aggregate_metrics(results_b)

    if gate is not None:
        gate.check(metrics_a)
        gate.check(metrics_b)

    diffs = {
        name: metrics_b.get(name, 0.0) - metrics_a.get(name, 0.0)
        for name in set(metrics_a) | set(metrics_b)
    }

    # diffs[name] > 0  means B is better on that metric
    # diffs[name] < 0  means A is better on that metric
    wins_b = sum(1 for d in diffs.values() if d > 0.0)
    wins_a = sum(1 for d in diffs.values() if d < 0.0)
    if wins_b > wins_a:
        winner = "b"
    elif wins_a > wins_b:
        winner = "a"
    else:
        winner = "tie"

    return {
        "a_metrics": metrics_a,
        "b_metrics": metrics_b,
        "metric_diffs": diffs,
        "winner": winner,
        "gate_passed": True,  # gate.check() runs above; if it passed, we're here
    }


def aggregate_metrics(results: list[Any]) -> dict[str, float]:
    """Average every metric across all results."""
    if not results:
        return {}
    keys = {k for r in results for k in r.metrics}
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


__all__ = [
    "FinanceBench",
    "FramesBenchmark",
    "LlmJudge",
    "Metrics",
    "QualityGate",
    "Scoring",
    "ab_test",
    "aggregate_metrics",
    "parse_score",
    "run",
]
