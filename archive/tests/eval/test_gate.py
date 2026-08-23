"""Tests for ``raghub.eval.gate`` (Gate threshold checker, compute_average)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from raghub.errors import ConfigurationError
from raghub.eval.gate import Gate, compute_average


def test_gate_init_with_invalid_default_mode_raises() -> None:
    """``Gate(default_mode='invalid')`` raises ConfigurationError."""

    with pytest.raises(ConfigurationError, match="default_mode"):
        Gate(default_mode="oops")


def test_gate_add_sets_threshold_for_metric() -> None:
    """``Gate.add`` records the threshold for the metric."""

    gate = Gate()
    result = gate.add("recall_at_5", 0.7)
    assert result is gate  # chainable
    assert gate.thresholds["recall_at_5"] == (0.7, "min")


def test_gate_add_uses_explicit_mode_when_provided() -> None:
    """``Gate.add(mode='max')`` records the supplied mode."""

    gate = Gate()
    gate.add("latency_p95", 100.0, mode="max")
    assert gate.thresholds["latency_p95"] == (100.0, "max")


def test_gate_add_rejects_invalid_mode() -> None:
    """``Gate.add(mode='invalid')`` raises ConfigurationError."""

    gate = Gate()
    with pytest.raises(ConfigurationError, match="must be 'min' or 'max'"):
        gate.add("m", 0.5, mode="oops")


def test_gate_check_passes_when_all_metrics_meet_min_thresholds() -> None:
    """``Gate.check`` returns silently when every metric meets its 'min' threshold."""

    gate = Gate({"recall_at_5": 0.7, "faithfulness": 0.8})
    gate.check({"recall_at_5": 0.9, "faithfulness": 0.95})


def test_gate_check_raises_when_min_metric_below_threshold() -> None:
    """``Gate.check`` raises when a 'min' metric is below threshold."""

    gate = Gate({"recall_at_5": 0.7})
    with pytest.raises(ConfigurationError, match="recall_at_5: 0.500 < 0.7"):
        gate.check({"recall_at_5": 0.5})


def test_gate_check_raises_when_metric_missing() -> None:
    """``Gate.check`` raises when a required metric is missing from the input."""

    gate = Gate({"recall_at_5": 0.7})
    with pytest.raises(ConfigurationError, match="missing"):
        gate.check({"other_metric": 0.9})


def test_gate_check_passes_when_max_metric_meets_threshold() -> None:
    """``Gate.check`` returns silently when a 'max' metric is at/below threshold."""

    gate = Gate({"latency_p95": 100.0, "cost": 5.0}, default_mode="max")
    gate.check({"latency_p95": 50.0, "cost": 5.0})


def test_gate_check_raises_when_max_metric_above_threshold() -> None:
    """``Gate.check`` raises when a 'max' metric is above threshold."""

    gate = Gate({"latency_p95": 100.0}, default_mode="max")
    with pytest.raises(ConfigurationError, match="latency_p95: 200.000 > 100.0"):
        gate.check({"latency_p95": 200.0})


def test_gate_check_aggregates_multiple_breaches_in_message() -> None:
    """``Gate.check`` reports all breaches in a single exception message."""

    gate = Gate({"a": 0.5, "b": 0.5})
    with pytest.raises(ConfigurationError, match="a: .* < 0.5.*b: .* < 0.5"):
        gate.check({"a": 0.1, "b": 0.1})


def test_gate_report_returns_per_metric_tuple() -> None:
    """``Gate.report`` returns ``(value, threshold, passed, mode)`` for each metric."""

    gate = Gate({"a": 0.5})
    report = gate.report({"a": 0.7})
    assert report == {"a": (0.7, 0.5, True, "min")}


def test_gate_report_marks_missing_metrics_as_failed() -> None:
    """``Gate.report`` returns ``passed=False`` when the metric is missing."""

    gate = Gate({"a": 0.5})
    report = gate.report({})
    assert report == {"a": (None, 0.5, False, "min")}


def test_gate_report_marks_below_threshold_as_failed() -> None:
    """``Gate.report`` returns ``passed=False`` when a 'min' metric is below threshold."""

    gate = Gate({"a": 0.5})
    report = gate.report({"a": 0.3})
    assert report == {"a": (0.3, 0.5, False, "min")}


def test_gate_report_marks_above_threshold_as_failed_for_max_mode() -> None:
    """``Gate.report`` returns ``passed=False`` when a 'max' metric is above threshold."""

    gate = Gate({"cost": 5.0}, default_mode="max")
    report = gate.report({"cost": 10.0})
    assert report == {"cost": (10.0, 5.0, False, "max")}


def test_compute_average_returns_empty_for_empty_input() -> None:
    """``compute_average([])`` returns an empty dict."""

    assert compute_average([]) == {}


def test_compute_average_averages_metrics_across_results() -> None:
    """``compute_average`` averages each metric across the result list."""

    results = [
        SimpleNamespace(metrics={"a": 1.0, "b": 0.0}),
        SimpleNamespace(metrics={"a": 3.0, "b": 2.0}),
    ]
    assert compute_average(results) == {"a": 2.0, "b": 1.0}


def test_compute_average_unions_metric_keys_across_results() -> None:
    """``compute_average`` collects metrics from every result into one dict.

    Missing metrics in a result count as 0, so the average is
    diluted toward zero when not every result reports every metric.
    """

    results = [
        SimpleNamespace(metrics={"a": 1.0}),
        SimpleNamespace(metrics={"b": 2.0}),
    ]
    assert compute_average(results) == {"a": 0.5, "b": 1.0}


def test_compute_average_treats_missing_metric_in_a_result_as_zero() -> None:
    """``compute_average`` treats a missing metric in one result as 0."""

    results = [
        SimpleNamespace(metrics={"a": 1.0, "b": 2.0}),
        SimpleNamespace(metrics={"a": 3.0}),  # 'b' missing
    ]
    # a = (1+3)/2 = 2, b = (2+0)/2 = 1
    assert compute_average(results) == {"a": 2.0, "b": 1.0}
