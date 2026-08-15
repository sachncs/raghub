"""Threshold-based quality gate and A/B comparison utilities.

:class:`Gate` validates a metrics dict against configurable per-metric
thresholds; :func:`compare` runs two RAG instances against the same
dataset and reports per-metric diffs, optionally applying a gate.
"""

from __future__ import annotations

from typing import Any

from raghub.errors import ConfigurationError
from raghub.eval.benchmarks import run


class Gate:
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

    >>> gate = Gate({"recall_at_5": 0.7, "faithfulness": 0.8})
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
                f"Gate default_mode must be 'min' or 'max', got {default_mode!r}"
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
    ) -> Gate:
        """Add or replace a threshold. Returns self for chaining."""
        chosen_mode = mode or self.default_mode
        if chosen_mode not in self.VALID_MODES:
            raise ConfigurationError(
                f"Gate mode for {metric!r} must be 'min' or 'max', got {chosen_mode!r}"
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
            raise ConfigurationError(f"Gate failed: {'; '.join(breaches)}")

    def report(self, metrics: dict[str, float]) -> dict[str, tuple[float | None, float, bool, str]]:
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


async def compare(
    *,
    rag_a: Any,
    rag_b: Any,
    examples: list[dict[str, Any]],
    evaluator: Any,
    gate: Gate | None = None,
) -> dict[str, Any]:
    """Run two RAG instances against the same dataset, report per-metric diffs.

    Args:
        rag_a: The "control" RAG instance.
        rag_b: The "treatment" RAG instance.
        examples: Per-example records with ``question`` (and any
            other keys the evaluator expects).
        evaluator: The evaluator to score both runs.
        gate: Optional :class:`Gate`. When set, the run fails
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
    factory_a = make_answer_factory(rag_a)
    factory_b = make_answer_factory(rag_b)

    results_a = await run(evaluator, examples, response_factory=factory_a)
    results_b = await run(evaluator, examples, response_factory=factory_b)

    metrics_a = compute_average(results_a)
    metrics_b = compute_average(results_b)

    if gate is not None:
        gate.check(metrics_a)
        gate.check(metrics_b)

    diffs = compute_metric_diffs(metrics_a, metrics_b)
    winner = determine_winner(diffs)

    return {
        "a_metrics": metrics_a,
        "b_metrics": metrics_b,
        "metric_diffs": diffs,
        "winner": winner,
        "gate_passed": True,
    }


def make_answer_factory(rag: Any) -> Any:
    """Build an async factory that calls ``rag.aquery`` and returns the answer.

    Defined as a module-level helper so the inner closures created
    by :func:`compare` are simple async functions rather than nested
    definitions that hide the orchestration logic.

    Args:
        rag: The RAG instance to query.

    Returns:
        An async callable ``(example) -> answer_text`` accepted by
        :func:`raghub.eval.benchmarks.run`.

    """

    async def factory(ex: dict[str, Any]) -> Any:
        """Build an :func:`aquery` factory for example ``ex`` returning ``answer``."""
        response = await rag.aquery(ex["question"])
        return response.answer

    return factory


def compute_metric_diffs(
    metrics_a: dict[str, float], metrics_b: dict[str, float]
) -> dict[str, float]:
    """Return ``b - a`` for every key in the union of ``metrics_a`` and ``metrics_b``."""
    return {
        name: metrics_b.get(name, 0.0) - metrics_a.get(name, 0.0)
        for name in set(metrics_a) | set(metrics_b)
    }


def determine_winner(diffs: dict[str, float]) -> str:
    """Return "a", "b", or "tie" based on which side has more wins."""
    wins_b = sum(1 for d in diffs.values() if d > 0.0)
    wins_a = sum(1 for d in diffs.values() if d < 0.0)
    if wins_b > wins_a:
        return "b"
    if wins_a > wins_b:
        return "a"
    return "tie"


def compute_average(results: list[Any]) -> dict[str, float]:
    """Average every metric across all results."""
    if not results:
        return {}
    keys = {k for r in results for k in r.metrics}
    return {k: sum(r.metrics.get(k, 0.0) for r in results) / len(results) for k in keys}


__all__ = ["Gate", "compare", "compute_average"]
