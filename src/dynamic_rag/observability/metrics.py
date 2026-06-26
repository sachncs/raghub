"""Metrics hooks."""

from __future__ import annotations

from typing import Any


class NullMetrics:
    """No-op metrics recorder."""

    def record_latency(self, name: str, value_ms: float, **labels: Any) -> None:
        return None

    def increment(self, name: str, value: int = 1, **labels: Any) -> None:
        return None

