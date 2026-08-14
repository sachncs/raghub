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

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from raghub.errors import EvaluationError
from raghub.eval.benchmarks.base import Evaluator
from raghub.models import Result
from raghub.runtime import capture


@Evaluator.register("frames")
class Frames(Evaluator):
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
        Frames.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = Frames.CACHE_DIR / f"{self.split}.jsonl"
        if not cached.exists():
            self.examples = self.load_huggingface(self.dataset_name, self.split)
            cached.write_text(
                "\n".join(json.dumps(ex) for ex in self.examples),
                encoding="utf-8",
            )
        else:
            self.examples = self.load_local(cached)
        return self.examples

    async def evaluate(
        self,
        examples: Sequence[dict[str, Any]] | None = None,
        *,
        response_factory: Any,
    ) -> list[Result]:
        """Score every example.

        Args:
            examples: Optional explicit examples; defaults to the
                benchmark's own dataset.
            response_factory: Async callable taking an example dict
                and returning the model's answer. The callable may
                return either a plain string or a
                ``(answer, contexts, retrieved_ids, relevant_ids)``
                tuple; the latter enables the full retrieval-quality
                metrics (recall@k, hit_rate@k, MRR, MAP).

        Returns:
            A list of :class:`Result`.

        """
        from raghub.eval.benchmarks import evaluate as benchmark_evaluate

        rows = list(examples) if examples is not None else self.ensure_examples()
        return await benchmark_evaluate(
            rows,
            response_factory,
            benchmark=self.benchmark,
            tolerance=self.tolerance,
        )

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

    @staticmethod
    def load_tsv(path: Path) -> list[dict[str, Any]]:
        """Parse a TSV / CSV file with the FRAMES schema."""
        try:
            import csv
        except ImportError:  # pragma: no cover - stdlib
            return []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ",")
            return [row for row in reader]

    @staticmethod
    def normalise_row(row: dict[str, Any]) -> dict[str, Any]:
        """Map a FRAMES row to the canonical ``Evaluator`` schema."""
        raw_links = row.get("wiki_links")
        if raw_links is None:
            # Fall back to the per-column Wikipedia columns.
            per_col = [row.get(f"wikipedia_link_{i}") for i in range(1, 11)]
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
        (e.g. ``"37th"``). :func:`Scoring.first_number` extracts the
        number from both sides before the tolerance comparison.
        """
        from raghub.eval.metrics import Metrics

        return Metrics.within_tolerance(predicted, gold, self.tolerance)

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load Finance examples from a local JSONL/JSON file.

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
        """Load Finance from the HuggingFace Hub.

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
            from raghub.eval.benchmarks.finance import Finance

            raise EvaluationError(
                "datasets is not installed; install it via "
                "`pip install datasets` or place a JSONL/JSON file at "
                f"{Finance.CACHE_DIR / 'financebench.jsonl'}."
            ) from exc
        ds, error = capture(datasets_module.load_dataset, dataset_name, split=split)
        if error is not None or ds is None:
            raise EvaluationError(
                f"Failed to load Finance from {dataset_name!r}: {error}"
            ) from error
        return [dict(record) for record in ds]


__all__ = ["Frames"]
