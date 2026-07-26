"""Tests for the parser base class and ``ParsedSection`` value object."""
from __future__ import annotations

import pytest

from raghub.documents.parsers.base import FileParser, ParsedSection


class _DummyParser(FileParser):
    """A minimal concrete parser for exercising the abstract base."""

    def parse(
        self, file_bytes: bytes, file_name: str, mime_type: str
    ) -> list[ParsedSection]:
        text = file_bytes.decode("utf-8", errors="replace")
        return [
            ParsedSection(
                section_index=0,
                source_location=file_name,
                text=text,
                metadata={"filename": file_name},
            )
        ]


def test_parsed_section_is_immutable() -> None:
    """``ParsedSection`` is frozen — attribute assignment is rejected."""
    section = ParsedSection(
        section_index=0, source_location="full file", text="hi", metadata={}
    )
    with pytest.raises((AttributeError, TypeError)):
        section.text = "other"  # type: ignore[misc]


def test_parsed_section_equality() -> None:
    """Equal field-by-field sections compare equal."""
    a = ParsedSection(section_index=1, source_location="loc", text="t", metadata={"k": 1})
    b = ParsedSection(section_index=1, source_location="loc", text="t", metadata={"k": 1})
    assert a == b


def test_parsed_section_metadata_accepts_empty_dict() -> None:
    """An empty ``metadata`` dict is accepted and stored verbatim."""
    section = ParsedSection(
        section_index=0, source_location="x", text="y", metadata={}
    )
    assert section.metadata == {}


def test_parsed_section_attributes_are_stored() -> None:
    """All four attributes round-trip via the constructor."""
    section = ParsedSection(
        section_index=3, source_location="page 4", text="body", metadata={"a": 1}
    )
    assert section.section_index == 3
    assert section.source_location == "page 4"
    assert section.text == "body"
    assert section.metadata == {"a": 1}


def test_file_parser_is_abstract() -> None:
    """``FileParser`` cannot be instantiated directly."""
    with pytest.raises(TypeError):
        FileParser()  # type: ignore[abstract]


def test_concrete_parser_implementation_can_be_instantiated() -> None:
    """A subclass that implements ``parse`` can be constructed."""
    parser = _DummyParser()
    assert isinstance(parser, FileParser)


def test_concrete_parser_receives_bytes_filename_and_mime() -> None:
    """The concrete ``parse`` sees the supplied bytes / filename / MIME."""
    parser = _DummyParser()
    sections = parser.parse(b"hello", "doc.txt", "text/plain")
    assert len(sections) == 1
    assert sections[0].text == "hello"
    assert sections[0].source_location == "doc.txt"
    assert sections[0].metadata == {"filename": "doc.txt"}


def test_concrete_parser_can_return_empty_list() -> None:
    """A parser that finds no extractable text may return ``[]``."""

    class _EmptyParser(FileParser):
        def parse(
            self, file_bytes: bytes, file_name: str, mime_type: str
        ) -> list[ParsedSection]:
            return []

    assert _EmptyParser().parse(b"", "empty.bin", "application/octet-stream") == []