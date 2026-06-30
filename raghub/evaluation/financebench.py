"""FinanceBench benchmark adapter.

The default benchmark shipped with the framework. FinanceBench is a
set of financial Q/A examples with numeric ground-truth answers; the
scoring is tolerance-based.

This adapter:

* Loads the dataset from disk or downloads it when not present.
* Runs the model against each example.
* Computes per-example scores (within-tolerance, token-overlap).
* Returns a list of :class:`EvaluationResult`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from raghub.evaluation.harness import score_string
from raghub.exceptions import EvaluationError
from raghub.interfaces.evaluation import Evaluator
from raghub.models import EvaluationResult

try:
    from datasets import load_dataset as _hf_load_dataset  # type: ignore
    _HF_AVAILABLE = True
    _ImportError: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dep
    _hf_load_dataset = None
    _HF_AVAILABLE = False
    _ImportError = exc


DEFAULT_DATASET = "PatronusAI/financebench"
DEFAULT_SPLIT = "train"
CACHE_DIR = Path(
    os.getenv("RAGHUB_FINANCEBENCH_CACHE", str(Path.home() / ".cache" / "raghub" / "financebench"))
)


def load_jsonl_file(path: Path) -> list[dict]:
    """Load FinanceBench examples from a local JSONL/JSON file.

    Args:
        path: Path to a JSON or JSONL file.

    Returns:
        A list of example dicts.
    """
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def load_huggingface_dataset(dataset_name: str, split: str) -> list[dict]:
    """Load FinanceBench from the HuggingFace Hub.

    Args:
        dataset_name: Hub dataset id.
        split: Dataset split.

    Returns:
        A list of example dicts.

    Raises:
        EvaluationError: When the dataset cannot be loaded.
    """
    if not _HF_AVAILABLE:
        raise EvaluationError(
            "datasets is not installed; install it via `pip install datasets` or "
            "place a JSONL/JSON file at "
            f"{CACHE_DIR/'financebench.jsonl'}."
        )
    try:
        ds = _hf_load_dataset(dataset_name, split=split)
    except Exception as exc:
        raise EvaluationError(
            f"Failed to load FinanceBench from {dataset_name!r}: {exc}"
        ) from exc
    return [dict(record) for record in ds]


class FinanceBenchEvaluator(Evaluator):
    """FinanceBench benchmark adapter."""

    benchmark: str = "financebench"

    def __init__(
        self,
        *,
        dataset_path: Path | None = None,
        dataset_name: str = DEFAULT_DATASET,
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
        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._tolerance = tolerance
        self._examples: list[dict] | None = None

    def ensure_loaded_examples(self) -> list[dict]:
        if self._examples is not None:
            return self._examples
        if self._dataset_path is not None:
            self._examples = load_jsonl_file(Path(self._dataset_path))
            if self._examples:
                return self._examples
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = CACHE_DIR / "financebench.jsonl"
        if not cached.exists():
            self._examples = load_huggingface_dataset(self._dataset_name, self._split)
            cached.write_text(
                "\n".join(json.dumps(ex) for ex in self._examples),
                encoding="utf-8",
            )
        else:
            self._examples = load_jsonl_file(cached)
        return self._examples

    def _ensure_examples(self) -> list[dict]:
        """Backwards-compatible alias for :meth:`ensure_loaded_examples`.

        Some legacy call sites (including the ``raghub eval`` CLI)
        previously called the private ``_ensure_examples`` name. The
        method is now part of the public contract under
        :meth:`ensure_loaded_examples`; this alias keeps older call
        sites working without changes.

        Returns:
            The list of FinanceBench example dicts.
        """
        return self.ensure_loaded_examples()

    async def evaluate(
        self,
        examples: Sequence[dict] | None = None,
        *,
        response_factory,
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
        from raghub.evaluation.metrics import evaluate_example

        rows = list(examples) if examples is not None else self.ensure_loaded_examples()
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
            overlap = score_string(str(predicted), str(gold))
            numeric = self.numeric_within_tolerance(str(predicted), str(gold))
            metrics = {"token_overlap": overlap, "numeric_within_tolerance": numeric}
            # Add retrieval-quality metrics when the response
            # factory returned the tuple form.
            if contexts is not None and retrieved_ids is not None:
                retrieval_metrics = evaluate_example(
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
        try:
            p, g = float(first_number(predicted)), float(first_number(gold))
        except (TypeError, ValueError):
            return 0.0
        if g == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(p - g) / max(abs(g), 1.0) <= self._tolerance else 0.0


def first_number(text: str) -> str:
    """Return the first whitespace-delimited token that parses as a number.

    Args:
        text: Arbitrary text.

    Returns:
        The first numeric token as a string. Empty if none.
    """
    for token in text.replace(",", "").split():
        try:
            float(token)
        except ValueError:
            continue
        return token
    return ""


__all__ = ["FinanceBenchEvaluator", "DEFAULT_DATASET", "DEFAULT_SPLIT"]
