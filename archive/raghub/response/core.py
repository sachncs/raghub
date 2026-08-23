"""Response-shape and redaction helpers.

The :class:`Redaction` class scrubs sensitive keys from a serialised
user payload. :class:`ResponseBuilder` maps a pipeline result into a
typed response so the API surface stays consistent.
"""

from __future__ import annotations

from typing import Any, cast

from raghub.models import Hit, Pipeline, Response


class Redaction:
    """Strip hash-like and other sensitive keys from a serialised user payload."""

    SENSITIVE: frozenset[str] = frozenset({"password_hash", "password", "token", "secret"})

    @classmethod
    def user(cls: type[Redaction], payload: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of ``payload`` with sensitive fields replaced.

        Args:
            payload: A user dict produced by ``UserRecord.dump``.

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
        answer: str = str(result.get("answer", "") or "")
        structured = result.get("structured")
        structured_payload = None

        if structured is not None:
            answer = cast(
                str,
                structured.dump(mode="json")
                if isinstance(structured, Pipeline)
                else str(structured),
            )
            structured_payload = structured.dump() if hasattr(structured, "dump") else structured

        metadata = {
            "pipeline_id": result.pipeline_id,
            "structured": structured is not None,
        }
        resolved_config = result.get("resolved_config")
        if resolved_config:
            metadata["resolved_config"] = resolved_config

        return Response(
            answer=answer,
            citations=list(result.get("citations", [])),
            source_chunks=[
                Hit(
                    score=h.score,
                    chunk=h.chunk,
                )
                for h in result.get("hits", [])
            ],
            metadata=metadata,
            structured=structured_payload,
            transforms_applied=list(result.get("transforms_applied", []) or []),
            planner_trace=list(result.get("planner_trace") or []) or None,
            tools_invoked=list(result.get("tools_invoked") or []),
        )


__all__ = [
    "Redaction",
    "ResponseBuilder",
]
