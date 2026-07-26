"""Build typed :class:`Response` objects from pipeline results.

A small module that owns the construction of the public
:class:`raghub.models.Response` (the canonical response type).
Keeping this logic in its own module lets the ``RAG`` class
stay small and makes the response format easy to test.
"""

from __future__ import annotations

from raghub.models import (
    PipelineResult,
    Response,
    SearchResult,
)


def build_response(result: PipelineResult) -> Response:
    """Build a typed :class:`Response` from a query pipeline result.

    Args:
        result: The :class:`PipelineResult` returned by
            :class:`QueryPipeline`.

    Returns:
        A typed :class:`Response` carrying ``answer`` (string or
        JSON-serialised structured model), ``citations``,
        ``source_chunks``, ``structured`` (the raw typed model
        when a ``response_model`` was supplied), and ``metadata``.
    """
    outputs = result.outputs
    answer = outputs.get("answer", "")
    structured = outputs.get("structured")
    structured_payload = None

    if structured is not None:
        answer = structured.model_dump_json()
        structured_payload = structured.model_dump()

    metadata = {
        "pipeline_id": result.pipeline_id,
        "structured": structured is not None,
    }
    resolved_config = outputs.get("resolved_config")
    if resolved_config:
        metadata["resolved_config"] = resolved_config

    return Response(
        answer=answer,
        citations=list(outputs.get("citations", [])),
        source_chunks=[
            SearchResult(chunk_id=h.chunk_id, score=h.score, chunk=h.chunk)
            for h in outputs.get("hits", [])
        ],
        metadata=metadata,
        structured=structured_payload,
        transforms_applied=list(outputs.get("transforms_applied", []) or []),
        planner_trace=list(outputs.get("planner_trace") or []) or None,
        tools_invoked=list(outputs.get("tools_invoked") or []),
    )


