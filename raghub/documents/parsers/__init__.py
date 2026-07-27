from __future__ import annotations

from .base import FileParser as FileParser
from .base import ParsedSection as ParsedSection
from .csv_parser import CsvParser as CsvParser
from .html_parser import HtmlParser as HtmlParser
from .image_parser import ImageParser as ImageParser
from .office_parser import OfficeParser as OfficeParser
from .pdf_parser import PdfParser as PdfParser
from .registry import ParserRegistry as ParserRegistry
from .txt_parser import TxtParser as TxtParser

"""Multi-format document parsers.

Concrete :class:`FileParser` implementations and the
:class:`ParserRegistry` that aggregates them. Each parser handles a
single format family; the registry provides MIME/extension-based
dispatch with a UTF-8 fallback for unknown formats.
"""
