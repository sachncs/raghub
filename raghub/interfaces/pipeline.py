"""Pipeline contract.

A pipeline composes a fixed sequence of stages into a deterministic,
observable, testable run. Concrete implementations include the
default end-to-end ingest/query pipelines in
:class:`raghub.pipelines.rag.DefaultRagPipeline`.
"""

from __future__ import annotations

from typing import Any, Protocol

from raghub.models import PipelineContext, PipelineResult


class Pipeline(Protocol):
    """A deterministic, multi-stage computation."""

    name: str

    async def run(self, context: PipelineContext, **inputs: Any) -> PipelineResult:
        """Execute the pipeline.

        Args:
            context: Per-invocation state.
            **inputs: Stage-specific inputs keyed by stage name.

        Returns:
            The populated :class:`PipelineResult`.
        """
