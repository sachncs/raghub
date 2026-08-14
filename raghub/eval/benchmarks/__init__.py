"""Domain package: ``raghub.eval.benchmarks``.

Re-exports the public surface from :mod:`raghub.eval.benchmarks.core`,
:mod:`raghub.eval.benchmarks.finance`, and
:mod:`raghub.eval.benchmarks.frames`.
"""

from __future__ import annotations

from raghub.eval.benchmarks.core import (
    evaluate,
    run,
)
from raghub.eval.benchmarks.finance import Finance
from raghub.eval.benchmarks.frames import Frames

__all__ = [
    "Finance",
    "Frames",
    "evaluate",
    "run",
]
