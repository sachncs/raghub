"""RAGAS integration as a RAGHub Evaluator.

The :class:`RagasAdapter` wraps the RAGAS evaluation framework and
exposes it through the standard :class:`raghub.eval.Evaluator` interface
so users can run RAGAS metrics (faithfulness, answer_relevancy,
context_precision, context_recall) inside the same benchmark
harness as :class:`raghub.eval.Finance` and
:class:`raghub.eval.Frames`.

Requires the ``[ragas]`` extra::

    pip install 'raghub[ragas]'

The adapter raises :class:`raghub.errors.MissingDepError` on import
when ragas is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from raghub.errors import ConfigurationError, MissingDepError
from raghub.models import Evaluator, Result


def import_ragas() -> Any:
    """Import ragas lazily; raise ``MissingDepError`` when not installed.

    Returns:
        The ragas module.

    Raises:
        MissingDepError: When ragas is not installed.

    """
    try:
        import ragas
    except ImportError as exc:
        raise MissingDepError(
            "ragas",
            "pip install 'raghub[ragas]'",
        ) from exc
    return ragas


def load_metric(metric_name: str) -> Any:
    """Load a ragas metric by name, with a friendly error on unknown names.

    Args:
        metric_name: One of ``"faithfulness"``, ``"answer_relevancy"``,
            ``"context_precision"``, ``"context_recall"``.

    Returns:
        The metric instance.

    Raises:
        ConfigurationError: When the metric name is unknown.

    """
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    if metric_name not in registry:
        raise ConfigurationError(
            f"Unknown ragas metric {metric_name!r}. Supported: {sorted(registry.keys())}"
        )
    return registry[metric_name]


class RagasAdapter(Evaluator):
    """Adapter that wraps a ragas evaluation as a RAGHub Evaluator.

    Translates a RAGHub ``(question, contexts, answer, ground_truth)``
    example into a row that ragas can evaluate, runs ragas, and
    converts the result back into :class:`Result` objects
    with the same metric names that :class:`raghub.eval.Metrics`
    produces.

    Args:
        metrics: Iterable of metric names to evaluate. Default
            ``("faithfulness", "answer_relevancy", "context_precision",
            "context_recall")`` — the four canonical ragas metrics.
        llm: Optional ragas-compatible LLM (e.g. an OpenAI-wrapped
            model). When ``None``, ragas uses its default evaluator.
        embeddings: Optional ragas-compatible embeddings. When
            ``None``, ragas uses its default embeddings.

    Example:
        >>> adapter = RagasAdapter(metrics=["faithfulness", "answer_relevancy"])
        >>> results = await adapter.evaluate(examples, response_factory=factory)
        >>> for r in results:
        ...     print(r.metrics)

    """

    benchmark: str = "ragas"

    DEFAULT_METRICS: tuple[str, ...] = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )

    def __init__(
        self,
        *,
        metrics: tuple[str, ...] | list[str] = DEFAULT_METRICS,
        llm: Any | None = None,
        embeddings: Any | None = None,
    ) -> None:
        """Store the metric names and optional ragas dependencies.

        The actual ragas import is deferred until :meth:`evaluate` is
        called so the constructor never raises
        :class:`MissingDepError` for callers that only want to inspect
        the adapter's configuration (including tests that supply a
        pre-built ragas module via ``__init__`` field assignment).
        """
        self.metric_names = tuple(metrics)
        self.llm = llm
        self.embeddings = embeddings
        self.metric_instances: list[Any] = []  # populated lazily in evaluate()

    def build_dataset(self, examples: list[dict[str, Any]]) -> Any:
        """Translate RAGHub examples into a ragas Dataset.

        Args:
            examples: RAGHub examples with ``question``, ``answer``,
                ``contexts`` (list of strings), and ``ground_truth``
                (optional).

        Returns:
            A ``datasets.Dataset`` with the ragas schema.

        """
        from datasets import Dataset

        rows: list[dict[str, Any]] = []
        for example in examples:
            rows.append(
                {
                    "question": example.get("question", ""),
                    "answer": example.get("answer", ""),
                    "contexts": list(example.get("contexts", [])),
                    "ground_truth": example.get("ground_truth", ""),
                }
            )
        return Dataset.from_list(rows)

    async def evaluate(
        self,
        examples: Sequence[dict[str, Any]] | None = None,
        *,
        response_factory: Any,
    ) -> list[Result]:
        """Score every example using the configured ragas metrics.

        Args:
            examples: Optional explicit examples. Defaults to an
                empty list (ragas requires its own dataset).
            response_factory: Async callable taking an example and
                returning a ``(answer, contexts, retrieved_ids,
                relevant_ids)`` tuple. The factory's ``answer``
                replaces whatever is in the example dict.

        Returns:
            A list of :class:`Result` objects, one per
            example, with the ragas metrics placed in ``metrics``.

        Raises:
            ConfigurationError: When ragas evaluation fails.

        """
        ragas = import_ragas()
        # Lazy-load metric instances now (or by a pre-set attribute on
        # the instance — see ``metric_instances``).
        if not getattr(self, "metric_instances", None):
            self.metric_instances = [load_metric(name) for name in self.metric_names]
        rows = list(examples) if examples is not None else []

        # Drive the response_factory so the consumer can swap the
        # answer in. RAGAS then evaluates the *factory's* answer,
        # not the static dict.
        outs = []
        for example in rows:
            out = await response_factory(example)
            if isinstance(out, tuple) and len(out) >= 1:
                outs.append(out[0])
            else:
                outs.append(out)
        for example, answer in zip(rows, outs, strict=True):
            if "answer" not in example:
                example["answer"] = answer
            example.setdefault("contexts", [])
            example.setdefault("ground_truth", "")

        dataset = self.build_dataset(rows)

        kwargs: dict[str, Any] = {"metrics": self.metric_instances}
        if self.llm is not None:
            kwargs["llm"] = self.llm
        if self.embeddings is not None:
            kwargs["embeddings"] = self.embeddings

        try:
            result = ragas.evaluate(dataset, **kwargs)
        except Exception as exc:
            raise ConfigurationError(f"ragas evaluation failed: {exc}") from exc

        scores = self.extract_scores(result, len(rows))

        outcomes: list[Result] = []
        for idx, example in enumerate(rows):
            metrics_for_row = {f"ragas_{name}": scores[name][idx] for name in self.metric_names}
            passed = all(metrics_for_row[f"ragas_{name}"] >= 0.5 for name in self.metric_names)
            outcomes.append(
                Result(
                    benchmark=self.benchmark,
                    example_id=str(example.get("id", idx)),
                    metrics=metrics_for_row,
                    passed=passed,
                    details={"predicted": example.get("answer", "")},
                )
            )
        return outcomes

    @staticmethod
    def extract_scores(result: Any, n: int) -> dict[str, list[float]]:
        """Pull per-row scores out of a ragas Result.

        RAGAS returns a wrapper whose ``scores`` attribute is a dict
        mapping metric name → numpy array of length ``n``. The
        array values are in ``[0.0, 1.0]`` for the canonical
        metrics. When a metric is missing from the wrapper, the
        per-row score is ``0.0``.
        """
        scores: dict[str, list[float]] = {}
        raw = getattr(result, "scores", None) or {}
        for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            values = raw.get(name)
            if values is None:
                scores[name] = [0.0] * n
                continue
            try:
                scores[name] = [float(v) for v in values]
            except (TypeError, ValueError):
                scores[name] = [0.0] * n
        return scores


__all__ = ["RagasAdapter", "import_ragas", "load_metric"]
