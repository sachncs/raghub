"""MIME / extension → parser lookup with a UTF-8 fallback.

This module implements the small registry used by the ingestion
pipeline to find the right :class:`FileParser` for an uploaded file.

Lookup order:

1. **MIME type** (exact match). Wins when the validator already
   resolved a concrete MIME.
2. **File extension** (lower-cased, including the leading dot). Wins
   when the MIME type is missing or not registered.
3. **UTF-8 fallback.** If neither lookup yields a parser, the bytes
   are decoded as UTF-8 (``errors="replace"``) and returned as a
   single :class:`ParsedSection`. This is a forgiving default for
   text-like formats that were not explicitly registered.

The default registry registers one parser per format family
(PDF, HTML, plain text, CSV, image, Office) under both the MIME type
and the canonical file extensions.
"""

from __future__ import annotations

from pathlib import Path

from .base import FileParser, ParsedSection
from .csv_parser import CsvParser
from .html_parser import HtmlParser
from .image_parser import ImageParser
from .office_parser import OfficeParser
from .pdf_parser import PdfParser
from .txt_parser import TxtParser


class ParserRegistry:
    """Two-tier (MIME then extension) parser lookup with a UTF-8 fallback."""

    def __init__(self) -> None:
        """Initialise the registry and install the default parsers."""
        self.parsers: dict[str, FileParser] = {}
        self.register_defaults()

    def register_defaults(self) -> None:
        """Install the standard parser set.

        The same :class:`FileParser` instance is registered under
        every MIME and extension in its family (e.g. one
        :class:`ImageParser` for png/jpeg/gif/webp/tiff). This keeps
        the registry compact and lets the same object back several
        lookups.
        """
        pdf = PdfParser()
        html = HtmlParser()
        image = ImageParser()
        office = OfficeParser()
        csv = CsvParser()
        txt = TxtParser()

        self.register("application/pdf", pdf)
        self.register("text/html", html)
        self.register("text/plain", txt)
        self.register("text/csv", csv)
        self.register("image/png", image)
        self.register("image/jpeg", image)
        self.register("image/jpg", image)
        self.register("image/gif", image)
        self.register("image/webp", image)
        self.register("image/tiff", image)
        self.register(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", office
        )
        self.register("application/msword", office)
        self.register("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", office)
        self.register("application/vnd.ms-excel", office)
        self.register(
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", office
        )
        self.register("application/vnd.ms-powerpoint", office)

        # Same parsers, addressed by file extension. The leading dot
        # is included so lookups are unambiguous (``".pdf"`` vs
        # ``"pdf"``).
        self.register(".pdf", pdf)
        self.register(".html", html)
        self.register(".htm", html)
        self.register(".txt", txt)
        self.register(".csv", csv)
        self.register(".png", image)
        self.register(".jpg", image)
        self.register(".jpeg", image)
        self.register(".gif", image)
        self.register(".webp", image)
        self.register(".tiff", image)
        self.register(".tif", image)
        self.register(".docx", office)
        self.register(".doc", office)
        self.register(".xlsx", office)
        self.register(".xls", office)
        self.register(".pptx", office)
        self.register(".ppt", office)

    def register(self, key: str, parser: FileParser) -> None:
        """Register ``parser`` under ``key``.

        Args:
            key: Either a MIME type (``"text/plain"``) or an extension
                including the leading dot (``".txt"``).
            parser: The :class:`FileParser` instance to use for that
                key.
        """
        self.parsers[key] = parser

    def get_parser(self, mime_type: str, file_name: str) -> FileParser | None:
        """Look up a parser by MIME type then by extension.

        Args:
            mime_type: The MIME type to try first.
            file_name: Used to derive the extension fallback.

        Returns:
            The matching :class:`FileParser`, or ``None`` if neither
            key is registered.
        """
        parser = self.parsers.get(mime_type)
        if parser is not None:
            return parser
        # Extension fallback: only consult when the filename has a dot;
        # ``Path.suffix`` returns ``""`` for dot-less names.
        ext = Path(file_name).suffix.lower() if "." in file_name else ""
        return self.parsers.get(ext)

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Dispatch to the appropriate parser, or fall back to UTF-8.

        Args:
            file_bytes: Raw file contents.
            file_name: Original filename.
            mime_type: MIME type reported by the validator.

        Returns:
            A list of :class:`ParsedSection`. When no parser is
            registered the function decodes the bytes as UTF-8 with
            ``errors="replace"`` and returns a single section tagged
            ``source_location="unknown"``. This silent fallback is
            intentional — it lets the pipeline gracefully accept
            unknown text-like formats — but callers that need strict
            format enforcement should validate up front.
        """
        parser = self.get_parser(mime_type, file_name)
        if parser is None:
            # Forgiving fallback: decode as UTF-8 and treat the whole
            # file as one section. ``errors="replace"`` keeps the
            # decoder from raising on invalid byte sequences, at the
            # cost of substituting the Unicode replacement character.
            return [
                ParsedSection(
                    section_index=0,
                    source_location="unknown",
                    text=file_bytes.decode("utf-8", errors="replace"),
                    metadata={},
                )
            ]
        return parser.parse(file_bytes, file_name, mime_type)
