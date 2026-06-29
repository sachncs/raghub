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
    SearchResult,
)
from raghub.models import PipelineResult


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

    return Response(
        answer=answer,
        citations=citations,
        source_chunks=[
            SearchResult(chunk_id=h.chunk_id, score=h.score, chunk=h.chunk)
            for h in hits
        ],
        metadata={"pipeline_id": result.pipeline_id, "structured": structured is not None},
        structured=structured_payload,
    )
