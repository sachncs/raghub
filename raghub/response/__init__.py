"""Response-shape and redaction helpers.

The :class:`Redaction` class scrubs sensitive keys from a serialised
user payload. :class:`ResponseBuilder` maps a pipeline result into a
typed response so the API surface stays consistent.
"""

from __future__ import annotations

from typing import Any

from raghub.models import Hit, Pipeline, Response


class Redaction:
    """Strip hash-like and other sensitive keys from a serialised user payload."""

    SENSITIVE: frozenset[str] = frozenset({"password_hash", "password", "token", "secret"})

    @classmethod
    def user(cls: type[Redaction], payload: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of ``payload`` with sensitive fields replaced.

        Args:
            payload: A user dict produced by ``UserRecord.model_dump``.

        Returns:
            A shallow copy with sensitive keys replaced by ``"***"``.

        """
        redacted = dict(payload)
        for key in list(redacted.keys()):
            if key.lower() in cls.SENSITIVE or "hash" in key.lower():
                redacted[key] = "***"
        return redacted


class ResponseBuilder:
    """Map a :class:`Pipeline` to a typed :class:`Response`."""

    @staticmethod
    def from_pipeline(result: Pipeline) -> Response:
        """Build a typed response from a pipeline result."""
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
                Hit(
                    score=h.score,
                    chunk=h.chunk,
                )
                for h in outputs.get("hits", [])
            ],
            metadata=metadata,
            structured=structured_payload,
            transforms_applied=list(outputs.get("transforms_applied", []) or []),
            planner_trace=list(outputs.get("planner_trace") or []) or None,
            tools_invoked=list(outputs.get("tools_invoked") or []),
        )


__all__ = [
    "Redaction",
    "ResponseBuilder",
]
