"""Build typed :class:`Response` objects from pipeline results.

A small module that owns the construction of the public
:class:`raghub.models.canonical.Response` (and its ``CanonicalResponse``
alias). Keeping this logic in its own module lets the ``RAG`` class
stay small and makes the response format easy to test.
"""

from __future__ import annotations

from typing import Any

from raghub.models import (
    CanonicalResponse as Response,
)
from raghub.models import (
    PipelineResult,
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
    citations: list = list(outputs.get("citations", []))
    hits = list(outputs.get("hits", []))
    structured = outputs.get("structured")
    structured_payload: dict[str, Any] | None = None

    if structured is not None:
        try:
            answer = structured.model_dump_json()
            structured_payload = structured.model_dump()
        except Exception:
            answer = str(structured)

    metadata: dict[str, Any] = {
        "pipeline_id": result.pipeline_id,
        "structured": structured is not None,
    }
    # Phase 8.7: surface the resolved advanced-RAG config that the
    # facade attached to the pipeline outputs via the context.
    resolved_config = outputs.get("resolved_config")
    if resolved_config:
        metadata["resolved_config"] = resolved_config

    return Response(
        answer=answer,
        citations=citations,
        source_chunks=[
            SearchResult(chunk_id=h.chunk_id, score=h.score, chunk=h.chunk) for h in hits
        ],
        metadata=metadata,
        structured=structured_payload,
        transforms_applied=list(outputs.get("transforms_applied", []) or []),
        planner_trace=list(outputs.get("planner_trace") or []) or None,
        tools_invoked=list(outputs.get("tools_invoked") or []),
    )
