from __future__ import annotations

"""Multi-format document parsers.

Concrete :class:`FileParser` implementations and the
:class:`ParserRegistry` that aggregates them. Each parser handles a
single format family; the registry provides MIME/extension-based
dispatch with a UTF-8 fallback for unknown formats.
"""

__all__ = [
    "FileParser",
    "ParsedSection",
    "ParserRegistry",
    "CsvParser",
    "HtmlParser",
    "ImageParser",
    "OfficeParser",
    "PdfParser",
    "TxtParser",
]

from .base import FileParser, ParsedSection  # noqa: E402
from .csv_parser import CsvParser  # noqa: E402
from .html_parser import HtmlParser  # noqa: E402
from .image_parser import ImageParser  # noqa: E402
from .office_parser import OfficeParser  # noqa: E402
from .pdf_parser import PdfParser  # noqa: E402
from .registry import ParserRegistry  # noqa: E402
from .txt_parser import TxtParser  # noqa: E402

