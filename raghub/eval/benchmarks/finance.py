"""Finance benchmark adapter (PatronusAI/financebench).

Loads the dataset from a local JSONL/JSON file when supplied, or
falls back to the HuggingFace Hub (when ``datasets`` is installed)
with caching under ``~/.cache/raghub/financebench/``.

Attributes:
    benchmark: The benchmark identifier persisted on every result.

"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from raghub.models import Result


class Finance:
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
        """Load the Finance dataset from local cache or HuggingFace.

        The dataset is cached in ``~/.cache/raghub/financebench/``
        after the first download.

        Returns:
            A list of example dicts, each with ``question``,
            ``answer``, and optional ``relevant_ids`` fields.

        """
        from raghub.eval.benchmarks.frames import Frames

        if self.examples is not None:
            return self.examples
        if self.dataset_path is not None:
            self.examples = Frames.load_jsonl(Path(self.dataset_path))
            if self.examples:
                return self.examples
        Finance.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = Finance.CACHE_DIR / "financebench.jsonl"
        if not cached.exists():
            self.examples = Frames.load_huggingface(self.dataset_name, self.split)
            cached.write_text(
                "\n".join(json.dumps(ex) for ex in self.examples),
                encoding="utf-8",
            )
        else:
            self.examples = Frames.load_jsonl(cached)
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
                metrics. When the simple form is returned, only
                token-overlap and numeric scores are computed.

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


__all__ = ["Finance"]
