"""Tests for ``raghub.pipeline.pipeline_assembly`` (Flow / PipelineBuilder)."""

from __future__ import annotations

from raghub.models import ErrorInfo, Pipeline, PipelineCtx, PipelineOutputs
from raghub.pipeline.pipeline_assembly import Flow, PipelineBuilder


def makemake_ctx() -> PipelineCtx:
    """Build a minimal PipelineCtx for tests."""

    return PipelineCtx(
        pipeline_id="run-123",
    )


def _outputs_with(extra: dict[str, object]) -> PipelineOutputs:
    """Build a :class:`PipelineOutputs` carrying the supplied ``extra`` map."""
    return PipelineOutputs(extra={k: v for k, v in extra.items()})


def test_flow_stores_context_and_pipeline_name() -> None:
    """``Flow.__init__`` stores the supplied context and pipeline name."""

    ctx = makemake_ctx()
    flow = Flow(ctx, "ingest")
    assert flow.context is ctx
    assert flow.pipeline_name == "ingest"


def test_flow_success_builds_pipeline_with_outputs() -> None:
    """``Flow.success(outputs)`` returns a Pipeline carrying outputs and pipeline_id."""

    flow = Flow(makemake_ctx(), "ingest")
    pipeline = flow.success(_outputs_with({"chunks": 3, "documents": 1}))
    assert isinstance(pipeline, Pipeline)
    assert pipeline.pipeline_id == "run-123"
    assert pipeline.pipeline_name == "ingest"
    assert pipeline.outputs.extra == {"chunks": 3, "documents": 1}
    assert pipeline.error is None


def test_flow_failure_populates_error_info() -> None:
    """``Flow.failure(error)`` sets the error field with kind='ingestion'."""

    flow = Flow(makemake_ctx(), "ingest")
    pipeline = flow.failure("failed to parse")
    assert pipeline.error is not None
    assert isinstance(pipeline.error, ErrorInfo)
    assert pipeline.error.kind == "ingestion"
    assert pipeline.error.message == "failed to parse"
    assert pipeline.outputs.extra == {}


def test_flow_failure_with_outputs_preserves_them() -> None:
    """``Flow.failure(error, outputs=...)`` carries partial outputs alongside the error."""

    flow = Flow(makemake_ctx(), "ingest")
    pipeline = flow.failure("partial", outputs=_outputs_with({"attempted": 2}))
    assert pipeline.error is not None
    assert pipeline.error.message == "partial"
    assert pipeline.outputs.extra == {"attempted": 2}


def test_flow_failure_without_outputs_defaults_to_empty_dict() -> None:
    """``Flow.failure(error)`` without ``outputs`` defaults to an empty dict."""

    flow = Flow(makemake_ctx(), "ingest")
    pipeline = flow.failure("boom")
    assert pipeline.outputs.extra == {}


def test_pipeline_builder_is_alias_for_flow() -> None:
    """``PipelineBuilder`` is the backward-compatible alias for :class:`Flow`."""

    assert PipelineBuilder is Flow

    builder = PipelineBuilder(makemake_ctx(), "query")
    pipeline = builder.success(_outputs_with({"answer": "42"}))
    assert pipeline.pipeline_name == "query"
    assert pipeline.outputs.extra == {"answer": "42"}


def test_pipeline_builder_can_be_constructed_via_alias_and_used() -> None:
    """The :class:`PipelineBuilder` alias accepts the same arguments as :class:`Flow`."""

    builder = PipelineBuilder(makemake_ctx(), "test")
    pipeline = builder.failure("err", outputs=_outputs_with({"x": 1}))
    assert pipeline.error is not None
    assert pipeline.error.message == "err"
    assert pipeline.outputs.extra == {"x": 1}


def test_flow_success_pipeline_id_is_unique_per_context() -> None:
    """Two Flow instances sharing one context produce pipelines with the same id."""

    ctx = makemake_ctx()
    flow_a = Flow(ctx, "a")
    flow_b = Flow(ctx, "b")
    empty = PipelineOutputs()
    assert flow_a.success(empty).pipeline_id == flow_b.success(empty).pipeline_id == "run-123"
