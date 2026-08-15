"""Pipeline builder — fluent factory for :class:`Pipeline` records.

Exposes :class:`Flow`; no alias is retained.
"""

from __future__ import annotations

from raghub.models import ErrorInfo, Pipeline, PipelineCtx, PipelineOutputs


class Flow:
    """Fluent builder for :class:`Pipeline` records."""

    def __init__(self, context: PipelineCtx, pipeline_name: str) -> None:
        """Store the context and pipeline name for subsequent builds."""
        self.context = context
        self.pipeline_name = pipeline_name

    def success(self, outputs: PipelineOutputs) -> Pipeline:
        """Build a successful :class:`Pipeline` with ``outputs``."""
        return Pipeline(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            outputs=outputs,
        )

    def failure(self, error: str, outputs: PipelineOutputs | None = None) -> Pipeline:
        """Build a failed :class:`Pipeline` with ``error``."""
        return Pipeline(
            pipeline_id=self.context.pipeline_id,
            pipeline_name=self.pipeline_name,
            error=ErrorInfo(kind="ingestion", message=error),
            outputs=outputs or PipelineOutputs(),
        )


__all__ = ["Flow", "Flow"]
