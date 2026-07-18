"""Comprehensive tests for converter modules: markdown, marker, directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from raghub.converters.directory import convert_path, select_converter_for_path
from raghub.converters.markdown import (
    markdown_to_document_blocks,
    normalise_markdown,
)
from raghub.converters.marker import (
    MARKER_AVAILABLE,
    MarkerConverter,
    build_marker_converter,
    looks_like_pdf,
)
from raghub.converters.plaintext import PlainTextConverter
from raghub.exceptions import ConfigurationError, ConversionError
from raghub.models import BlockKind, KnowledgeBundle

# =========================================================================
# markdown.py
# =========================================================================


class TestLooksLikePdf:
    def test_pdf_magic_number(self) -> None:
        assert looks_like_pdf(b"%PDF-1.4 fake content")
        assert looks_like_pdf(b"%PDF-")
        assert looks_like_pdf(b"%PDF-2.0")

    def test_non_pdf_bytes(self) -> None:
        assert not looks_like_pdf(b"hello world")

    def test_empty_bytes(self) -> None:
        assert not looks_like_pdf(b"")

    def test_partial_match(self) -> None:
        assert not looks_like_pdf(b"%PD")
        assert not looks_like_pdf(b"PDF-")


class TestMarkerConverterInit:
    def test_init_when_marker_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("raghub.converters.marker.MARKER_AVAILABLE", False)
        with pytest.raises(ConfigurationError, match="marker-pdf is not installed"):
            MarkerConverter()

    def test_init_when_marker_available(self) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()
        assert converter.converter is None


class TestMarkerConverterConvert:
    def test_empty_bytes(self) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()
        with pytest.raises(ConfigurationError, match="empty bytes"):
            converter.convert(source_uri="file://empty.pdf", file_bytes=b"")

    def test_delegates_non_pdf_to_plaintext(self) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()
        bundle = converter.convert(
            source_uri="file://notes.txt",
            file_bytes=b"hello world. this is plain text.",
            mime_type="text/plain",
        )
        assert isinstance(bundle, KnowledgeBundle)
        assert bundle.mime_type == "text/plain"
        assert any(
            "hello" in (block.content or "")
            for section in bundle.sections
            for block in section.blocks
        )

    def test_delegation_uses_passed_mime(self) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()
        bundle = converter.convert(
            source_uri="file://notes.txt",
            file_bytes=b"custom mime",
            mime_type="text/markdown",
        )
        assert bundle.mime_type == "text/markdown"

    def test_delegation_without_explicit_mime(self) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()
        bundle = converter.convert(
            source_uri="file://notes.txt",
            file_bytes=b"default mime",
        )
        assert bundle.mime_type == "text/plain"

    def test_pdf_bytes_marker_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()

        class _FailingConverter:
            def __call__(self, _path: str) -> None:
                raise RuntimeError("marker exploded")

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _FailingConverter())
        with pytest.raises(ConversionError, match="Marker conversion failed"):
            converter.convert(
                source_uri="file://doc.pdf",
                file_bytes=b"%PDF-1.4 garbage",
            )

    def test_pdf_bytes_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()

        class _MockRendered:
            markdown = "# Hello\n\nworld"

        class _MockConverter:
            def __call__(self, _path: str) -> _MockRendered:
                return _MockRendered()

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _MockConverter())
        bundle = converter.convert(
            source_uri="file://doc.pdf",
            file_bytes=b"%PDF-1.4 content",
        )
        assert isinstance(bundle, KnowledgeBundle)
        assert bundle.mime_type == "application/pdf"
        assert bundle.source_uri == "file://doc.pdf"

    def test_pdf_bytes_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()

        class _MockRendered:
            markdown = "# Title\nbody"

        class _MockConverter:
            def __call__(self, _path: str) -> _MockRendered:
                return _MockRendered()

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _MockConverter())
        bundle = converter.convert(
            source_uri="file://doc.pdf",
            file_bytes=b"%PDF-1.4 content",
            language="en",
            metadata={"source": "test"},
        )
        assert bundle.language == "en"
        assert bundle.metadata == {"source": "test"}

    def test_pdf_bytes_rendered_str_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the rendered object has no .markdown attr we use str()."""
        if not MARKER_AVAILABLE:
            pytest.skip("marker-pdf not installed")
        converter = MarkerConverter()

        class _MockConverter:
            def __call__(self, _path: str) -> str:
                return "fallback text"

        monkeypatch.setattr(converter, "marker_converter_instance", lambda: _MockConverter())
        bundle = converter.convert(
            source_uri="file://doc.pdf",
            file_bytes=b"%PDF-1.4 content",
        )
        assert any(
            "fallback" in (block.content or "")
            for section in bundle.sections
            for block in section.blocks
        )


class TestBuildMarkerConverter:
    def test_raises_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("raghub.converters.marker.MARKER_AVAILABLE", False)
        monkeypatch.setattr("raghub.converters.marker.MarkerPdfConverter", None)
        with pytest.raises(ConfigurationError, match="marker-pdf is not installed"):
            build_marker_converter()


# =========================================================================
# directory.py
# =========================================================================


class TestSelectConverterForPath:
    def test_pdf_returns_marker_or_plaintext(self) -> None:
        converter = select_converter_for_path(Path("doc.pdf"))
        if MARKER_AVAILABLE:
            from raghub.converters.marker import MarkerConverter

            assert isinstance(converter, MarkerConverter)
        else:
            assert isinstance(converter, PlainTextConverter)

    def test_pdf_uppercase(self) -> None:
        converter = select_converter_for_path(Path("doc.PDF"))
        if MARKER_AVAILABLE:
            from raghub.converters.marker import MarkerConverter

            assert isinstance(converter, MarkerConverter)
        else:
            assert isinstance(converter, PlainTextConverter)

    def test_txt_returns_plaintext(self) -> None:
        converter = select_converter_for_path(Path("doc.txt"))
        assert isinstance(converter, PlainTextConverter)

    def test_no_extension_returns_plaintext(self) -> None:
        converter = select_converter_for_path(Path("README"))
        assert isinstance(converter, PlainTextConverter)

    def test_non_pdf_extension_returns_plaintext(self) -> None:
        converter = select_converter_for_path(Path("doc.html"))
        assert isinstance(converter, PlainTextConverter)

    def test_pdf_fallback_on_marker_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from raghub.converters import directory as directory_module

        class _BrokenMarker:
            def __init__(self, *args, **kwargs):
                raise ConfigurationError("marker-pdf is not installed")

        monkeypatch.setattr(directory_module, "MarkerConverter", _BrokenMarker)
        converter = select_converter_for_path(Path("doc.pdf"))
        assert isinstance(converter, PlainTextConverter)


class TestConvertPath:
    def test_text_file(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("hello", encoding="utf-8")
        bundle = convert_path(p)
        assert bundle.source_uri == str(p.resolve())
        assert any("hello" in block.content for block in bundle.sections[0].blocks)

    def test_uses_provided_converter(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("explicit", encoding="utf-8")
        bundle = convert_path(p, converter=PlainTextConverter())
        assert any("explicit" in block.content for block in bundle.sections[0].blocks)

    def test_pdf_routes_to_marker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """For a .pdf path convert_path picks MarkerConverter."""
        from raghub.converters import directory as directory_module
        from raghub.converters import marker as marker_module

        recorded_source: list[str] = []

        class _MockMarkerConverter:
            def convert(
                self,
                *,
                source_uri: str,
                file_bytes: bytes,
                mime_type: str = "",
                language: str = "",
                metadata: dict | None = None,
            ) -> KnowledgeBundle:
                recorded_source.append(source_uri)
                return KnowledgeBundle(
                    source_uri=source_uri,
                    mime_type=mime_type,
                    sections=[],
                )

        monkeypatch.setattr(directory_module, "MarkerConverter", _MockMarkerConverter)
        monkeypatch.setattr(marker_module, "MARKER_AVAILABLE", True)

        p = tmp_path / "doc.pdf"
        p.write_text("%PDF-1.4 content", encoding="utf-8")
        bundle = convert_path(p)
        assert bundle.mime_type == "application/pdf"

    def test_pdf_fallback_to_plaintext(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from raghub.converters import directory as directory_module
        from raghub.exceptions import ConfigurationError

        class _BrokenMarker:
            def __init__(self, *_, **__):
                raise ConfigurationError("marker-pdf is not installed")

            def convert(self, *_, **__) -> KnowledgeBundle:
                raise RuntimeError("should not be called")

        monkeypatch.setattr(directory_module, "MarkerConverter", _BrokenMarker)
        p = tmp_path / "doc.pdf"
        p.write_text("fallback content", encoding="utf-8")
        bundle = convert_path(p)
        assert any("fallback" in block.content for block in bundle.sections[0].blocks)

    def test_file_not_found(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            convert_path(p)

    def test_source_uri_is_resolved(self, tmp_path: Path) -> None:
        p = tmp_path / "data.txt"
        p.write_text("resolve me", encoding="utf-8")
        bundle = convert_path(p)
        assert bundle.source_uri == str(p.resolve())

    def test_converter_selected_by_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "notes.html"
        p.write_text("<html>content</html>", encoding="utf-8")
        bundle = convert_path(p)
        assert bundle.mime_type == "text/plain"
        assert isinstance(bundle, KnowledgeBundle)


# =========================================================================
# markdown.py
# =========================================================================


class TestMarkdownToDocumentBlocks:
    def test_plain_text(self) -> None:
        blocks, trailing = markdown_to_document_blocks("hello world")
        assert trailing == ""
        assert len(blocks) == 1
        assert blocks[0].kind == BlockKind.TEXT
        assert "hello world" in blocks[0].content

    def test_empty_string(self) -> None:
        blocks, trailing = markdown_to_document_blocks("")
        assert blocks == []
        assert trailing == ""

    def test_only_whitespace(self) -> None:
        blocks, trailing = markdown_to_document_blocks("   \n\n  ")
        assert blocks == []
        assert trailing == ""

    def test_code_fence(self) -> None:
        md = "```python\nprint('hello')\n```"
        blocks, trailing = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.CODE for b in blocks)
        code_block = next(b for b in blocks if b.kind == BlockKind.CODE)
        assert code_block.content == "print('hello')"
        assert code_block.metadata.get("language") == "python"

    def test_code_fence_without_language(self) -> None:
        md = "```\ncode\n```"
        blocks, _ = markdown_to_document_blocks(md)
        code_block = next(b for b in blocks if b.kind == BlockKind.CODE)
        assert code_block.metadata.get("language") == ""

    def test_code_fence_tilde(self) -> None:
        md = "~~~\ncode\n~~~"
        blocks, _ = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.CODE for b in blocks)
        code_block = next(b for b in blocks if b.kind == BlockKind.CODE)
        assert code_block.content == "code"

    def test_unclosed_fence(self) -> None:
        """An unclosed fence does not emit a code block; text remains."""
        md = "```python\nprint('hello')"
        blocks, trailing = markdown_to_document_blocks(md)
        assert not any(b.kind == BlockKind.CODE for b in blocks)

    def test_table(self) -> None:
        md = "| a | b |\n| c | d |"
        blocks, _ = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.TABLE for b in blocks)

    def test_table_not_matched_if_not_pipe(self) -> None:
        md = "normal text"
        blocks, _ = markdown_to_document_blocks(md)
        assert not any(b.kind == BlockKind.TABLE for b in blocks)

    def test_text_before_table_emitted_as_text(self) -> None:
        md = "prefix\n| a | b |"
        blocks, _ = markdown_to_document_blocks(md)
        assert blocks[0].kind == BlockKind.TEXT
        assert blocks[1].kind == BlockKind.TABLE

    def test_equation_block(self) -> None:
        md = "$$E = mc^2$$"
        blocks, _ = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.EQUATION for b in blocks)
        eq = next(b for b in blocks if b.kind == BlockKind.EQUATION)
        assert eq.content == "E = mc^2"

    def test_equation_block_multiline(self) -> None:
        md = "$$\nE = mc^2\n$$"
        blocks, _ = markdown_to_document_blocks(md)
        eq_blocks = [b for b in blocks if b.kind == BlockKind.EQUATION]
        # multiline pattern only matches single-line $$...$$; this should
        # remain as text blocks
        assert len(eq_blocks) == 0

    def test_inline_equations_converted(self) -> None:
        """Inline $...$ are converted to \\(...\\) in TEXT blocks."""
        md = "This is $E = mc^2$ an equation."
        blocks, _ = markdown_to_document_blocks(md)
        assert blocks[0].kind == BlockKind.TEXT
        assert r"\(E = mc^2\)" in blocks[0].content

    def test_multiple_inline_equations(self) -> None:
        md = "$a$ and $b$"
        blocks, _ = markdown_to_document_blocks(md)
        assert r"\(a\)" in blocks[0].content
        assert r"\(b\)" in blocks[0].content

    def test_dollar_not_equation(self) -> None:
        """Dollar without closing newline stays as-is."""
        md = "cost is $5"
        blocks, _ = markdown_to_document_blocks(md)
        assert "$5" in blocks[0].content

    def test_image(self) -> None:
        md = "text\n![alt](https://example.com/img.png)\nmore"
        blocks, _ = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.IMAGE for b in blocks)
        img = next(b for b in blocks if b.kind == BlockKind.IMAGE)
        assert img.content == "https://example.com/img.png"
        assert img.metadata.get("caption") == "alt"

    def test_image_without_text(self) -> None:
        md = "![](https://example.com/img.png)"
        blocks, _ = markdown_to_document_blocks(md)
        assert any(b.kind == BlockKind.IMAGE for b in blocks)
        img = next(b for b in blocks if b.kind == BlockKind.IMAGE)
        assert img.metadata.get("caption") == ""
        assert img.content == "https://example.com/img.png"

    def test_mixed_content(self) -> None:
        md = """start

```python
code
```

| col1 | col2 |

$$x^2$$

end
"""
        blocks, _ = markdown_to_document_blocks(md)
        kinds = [b.kind for b in blocks]
        assert BlockKind.TEXT in kinds
        assert BlockKind.CODE in kinds
        assert BlockKind.TABLE in kinds
        assert BlockKind.EQUATION in kinds


class TestNormaliseMarkdown:
    def test_simple_text(self) -> None:
        bundle = normalise_markdown("hello world", source_uri="file://test.md")
        assert isinstance(bundle, KnowledgeBundle)
        assert bundle.source_uri == "file://test.md"
        assert len(bundle.sections) == 1
        assert len(bundle.sections[0].blocks) >= 1

    def test_source_uri_and_mime(self) -> None:
        bundle = normalise_markdown(
            "content",
            source_uri="file://doc.txt",
            mime_type="text/plain",
        )
        assert bundle.source_uri == "file://doc.txt"
        assert bundle.mime_type == "text/plain"

    def test_language(self) -> None:
        bundle = normalise_markdown("content", source_uri="u", language="en")
        assert bundle.language == "en"

    def test_metadata(self) -> None:
        bundle = normalise_markdown(
            "content",
            source_uri="u",
            metadata={"author": "test"},
        )
        assert bundle.metadata == {"author": "test"}

    def test_empty_metadata_defaults(self) -> None:
        bundle = normalise_markdown("content", source_uri="u", metadata=None)
        assert bundle.metadata == {}

    def test_page_numbers(self) -> None:
        bundle = normalise_markdown(
            "content",
            source_uri="u",
            page_numbers=[1, 2, 3],
        )
        assert bundle.sections[0].page_numbers == [1, 2, 3]

    def test_empty_page_numbers_defaults(self) -> None:
        bundle = normalise_markdown("content", source_uri="u", page_numbers=None)
        assert bundle.sections[0].page_numbers == []

    def test_empty_markdown_generates_empty_bundle(self) -> None:
        bundle = normalise_markdown("", source_uri="u")
        assert len(bundle.sections) == 1
        assert bundle.sections[0].blocks == []

    def test_table_in_markdown(self) -> None:
        md = "| H1 | H2 |\n|----|----|\n| A  | B  |"
        bundle = normalise_markdown(md, source_uri="u")
        assert any(b.kind == BlockKind.TABLE for b in bundle.sections[0].blocks)

    def test_code_block_in_markdown(self) -> None:
        md = "```python\nprint('hi')\n```"
        bundle = normalise_markdown(md, source_uri="u")
        assert any(b.kind == BlockKind.CODE for b in bundle.sections[0].blocks)

    def test_image_in_markdown(self) -> None:
        md = "![logo](https://example.com/logo.png)"
        bundle = normalise_markdown(md, source_uri="u")
        assert any(b.kind == BlockKind.IMAGE for b in bundle.sections[0].blocks)

    def test_equation_in_markdown(self) -> None:
        md = "$$F = ma$$"
        bundle = normalise_markdown(md, source_uri="u")
        assert any(b.kind == BlockKind.EQUATION for b in bundle.sections[0].blocks)

    def test_heading_in_text(self) -> None:
        """Headings are preserved inside TEXT blocks."""
        md = "# Chapter 1\n\nSome text."
        bundle = normalise_markdown(md, source_uri="u")
        blocks = bundle.sections[0].blocks
        assert any("Chapter 1" in block.content for block in blocks if block.kind == BlockKind.TEXT)
