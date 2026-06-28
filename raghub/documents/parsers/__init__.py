"""Multi-format document parsers.

Concrete :class:`FileParser` implementations and the
:class:`ParserRegistry` that aggregates them. Each parser handles a
single format family; the registry provides MIME/extension-based
dispatch with a UTF-8 fallback for unknown formats.
"""

from .base import FileParser, ParsedSection
from .csv_parser import CsvParser
from .html_parser import HtmlParser
from .image_parser import ImageParser
from .office_parser import OfficeParser
from .pdf_parser import PdfParser
from .registry import ParserRegistry
from .txt_parser import TxtParser

__all__ = [
    "CsvParser",
    "FileParser",
    "HtmlParser",
    "ImageParser",
    "OfficeParser",
    "ParsedSection",
    "ParserRegistry",
    "PdfParser",
    "TxtParser",
]