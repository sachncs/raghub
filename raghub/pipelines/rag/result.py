"""Helpers for constructing :class:`PipelineResult` records.

The query and ingest pipelines both build :class:`PipelineResult`
records with the same shape. Centralising the construction here
keeps the pipelines thin and gives tests a single seam to mock.
"""

from __future__ import annotations

from typing import Any

from raghub.models import PipelineContext, PipelineResult


class PipelineResultBuilder:
    """Fluent builder for :class:`PipelineResult` records.

    Args:
        context: Per-invocation state carrying the pipeline id.
        pipeline_name: The pipeline's ``name`` attribute; used as
            the ``pipeline_name`` field of the result.
    """

    def __init__(self, context: PipelineContext, pipeline_name: str) -> None:
        """Store the context and pipeline name for subsequent builds."""
        self.context = context
        self.pipeline_name = pipeline_name

    def success(self, outputs: dict[str, Any]) -> PipelineResult:
        """Build a successful :class:`PipelineResult` with ``outputs``.

        Args:
            outputs: The output mapping (e.g. ``{"answer": ...,
                "citations": [...]}``).

        Returns:
            A :class:`PipelineResult` with ``success=True`` and the
            supplied outputs.
        """
        return PipelineResult(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            success=True,
            outputs=outputs,
        )

    def failure(self, error: str, outputs: dict[str, Any] | None = None) -> PipelineResult:
        """Build a failed :class:`PipelineResult` with ``error``.

        Args:
            error: Human-readable error message.
            outputs: Optional partial outputs captured before the
                failure (default: ``None``).

        Returns:
            A :class:`PipelineResult` with ``success=False``.
        """
        return PipelineResult(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            success=False,
            error=error,
            outputs=outputs,
        )


__all__ = ["PipelineResultBuilder"]