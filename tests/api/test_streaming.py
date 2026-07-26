"""Phase 10 — SSE formatter + streaming endpoint tests."""

from __future__ import annotations

import json

from raghub.api.streaming import sse_comment, sse_format


def test_sse_format_emits_event_and_data_lines() -> None:
    payload = sse_format("answer_chunk", {"text": "hi"})
    text = payload.decode("utf-8")
    assert text.startswith("event: answer_chunk\n")
    assert "data: " in text
    assert text.endswith("\n\n")
    # The data payload must be JSON-decodable.
    body = text.rstrip("\n").splitlines()
    data_line = next(line for line in body if line.startswith("data:"))
    assert json.loads(data_line.split("data: ", 1)[1]) == {"text": "hi"}


def test_sse_format_string_data_passthrough() -> None:
    """String ``data`` is passed through verbatim (no JSON encoding)."""
    payload = sse_format("thought", "free-form text")
    assert b"data: free-form text" in payload


def test_sse_comment_starts_with_colon() -> None:
    assert sse_comment("ping").startswith(b": ping\n\n")


def test_sse_format_handles_non_serialisable_payload() -> None:
    class Foo:
        def __str__(self) -> str:
            return "foo"

    payload = sse_format("x", {"obj": Foo()})
    assert b'"foo"' in payload