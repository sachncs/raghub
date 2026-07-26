"""Tests for ``raghub.api.streaming`` SSE helpers."""
from __future__ import annotations

import json

from raghub.api.streaming import sse_comment, sse_format


def test_sse_format_returns_bytes() -> None:
    """The encoder returns ``bytes`` ready for streaming."""
    assert isinstance(sse_format("evt", {"x": 1}), bytes)


def test_sse_format_event_line() -> None:
    """The first line declares the event label."""
    payload = sse_format("answer_chunk", {"text": "hi"})
    text = payload.decode("utf-8")
    assert text.startswith("event: answer_chunk\n")


def test_sse_format_data_line_included() -> None:
    """The output contains a ``data:`` line."""
    payload = sse_format("evt", {"k": "v"})
    assert b"data: " in payload


def test_sse_format_terminates_with_double_newline() -> None:
    """SSE frames are delimited by a blank line."""
    payload = sse_format("evt", {})
    assert payload.endswith(b"\n\n")


def test_sse_format_json_serialises_dict_payload() -> None:
    """A dict payload is rendered as JSON text."""
    payload = sse_format("answer_chunk", {"text": "hi"})
    text = payload.decode("utf-8")
    body = text.rstrip("\n").splitlines()
    data_line = next(line for line in body if line.startswith("data:"))
    assert json.loads(data_line.split("data: ", 1)[1]) == {"text": "hi"}


def test_sse_format_passes_through_string_payload() -> None:
    """String payloads are not double-encoded."""
    payload = sse_format("thought", "free-form text")
    assert b"data: free-form text" in payload


def test_sse_format_serialises_list_payload() -> None:
    """Lists are serialised as JSON arrays."""
    payload = sse_format("evt", [1, 2, 3])
    assert b"data: [1, 2, 3]" in payload


def test_sse_format_serialises_numeric_payload() -> None:
    """Numeric payloads are serialised as their JSON representation."""
    payload = sse_format("evt", 42)
    assert b"data: 42" in payload


def test_sse_format_serialises_boolean_payload() -> None:
    """Boolean payloads are serialised as ``true`` / ``false``."""
    payload = sse_format("evt", True)
    assert b"data: true" in payload


def test_sse_format_handles_non_serialisable_objects() -> None:
    """Objects without a JSON encoder are stringified via ``default=str``."""

    class _Foo:
        def __str__(self) -> str:
            return "foo"

    payload = sse_format("evt", {"obj": _Foo()})
    assert b'"foo"' in payload


def test_sse_format_uses_utf8_encoding() -> None:
    """The output is UTF-8 encoded."""
    payload = sse_format("evt", {"text": "café"})
    assert b"caf" in payload


def test_sse_comment_returns_bytes() -> None:
    """``sse_comment`` returns ``bytes``."""
    assert isinstance(sse_comment("ping"), bytes)


def test_sse_comment_starts_with_colon() -> None:
    """SSE comments begin with ``:`` to mark them as non-data."""
    assert sse_comment("ping").startswith(b": ping\n\n")


def test_sse_comment_carries_arbitrary_text() -> None:
    """The comment body appears after the leading colon."""
    assert b"keep-alive" in sse_comment("keep-alive")


def test_sse_comment_encodes_as_utf8() -> None:
    """Non-ASCII comments are UTF-8 encoded."""
    payload = sse_comment("café")
    assert payload == ": café\n\n".encode("utf-8")


def test_sse_format_event_label_arbitrary_string() -> None:
    """The event label can be any string."""
    payload = sse_format("custom.event_name", {"k": 1})
    assert b"event: custom.event_name\n" in payload


def test_sse_format_handles_empty_dict_payload() -> None:
    """An empty dict payload yields an empty JSON object."""
    payload = sse_format("evt", {})
    assert b"data: {}" in payload