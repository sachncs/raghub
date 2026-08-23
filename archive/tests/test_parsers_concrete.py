"""Coverage tests for the concrete file parsers in :mod:`raghub.parsers`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from raghub.errors import MissingDepError
from raghub.parsers import (
    HTML,
    Csv,
    File,
    Image,
    Office,
    Pdf,
    Txt,
)

# ---------------------------------------------------------------------------
# Pdf
# ---------------------------------------------------------------------------


def test_pdf_parse_returns_one_section_per_page() -> None:
    """Each PDF page becomes a :class:`ParsedSection`."""

    class _Page:
        def __init__(self, text: str, mediabox: Any) -> None:
            self._text = text
            self.mediabox = mediabox

        def extract_text(self) -> str:
            return self._text

    class _Mediabox:
        width = 100
        height = 200

    class _Reader:
        def __init__(self, _stream: Any) -> None:
            self.pages = [
                _Page("page one", _Mediabox()),
                _Page("", _Mediabox()),
                _Page("page three", _Mediabox()),
            ]

    with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=_Reader)}):
        sections = Pdf.parse(b"fake-pdf-bytes", "doc.pdf", "application/pdf")

    assert len(sections) == 3
    assert sections[0].section_index == 1
    assert sections[0].source_location == "page 1"
    assert sections[0].text == "page one"
    assert sections[1].text == ""
    assert sections[2].text == "page three"


def test_pdf_parse_includes_page_dimensions() -> None:
    """Each section's metadata carries the page's width and height."""

    class _Mediabox:
        width = 612
        height = 792

    class _Page:
        mediabox = _Mediabox()

        def extract_text(self) -> str:
            return "text"

    class _Reader:
        def __init__(self, _stream: Any) -> None:
            self.pages = [_Page()]

    with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=_Reader)}):
        sections = Pdf.parse(b"bytes", "x.pdf", "application/pdf")
    assert sections[0].metadata == {"width": 612, "height": 792}


def test_pdf_parse_handles_page_without_mediabox() -> None:
    """Pages without a media box get an empty metadata dict."""

    class _Page:
        mediabox = None

        def extract_text(self) -> str:
            return "text"

    class _Reader:
        def __init__(self, _stream: Any) -> None:
            self.pages = [_Page()]

    with patch.dict("sys.modules", {"pypdf": MagicMock(PdfReader=_Reader)}):
        sections = Pdf.parse(b"bytes", "x.pdf", "application/pdf")
    assert sections[0].metadata == {}


def test_pdf_parse_raises_missing_dep_when_pypdf_missing() -> None:
    """ImportError → :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"pypdf": None}):
        with pytest.raises(MissingDepError, match="pypdf"):
            Pdf.parse(b"bytes", "x.pdf", "application/pdf")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_html_parse_returns_single_section() -> None:
    """HTML is parsed into a single :class:`ParsedSection`."""

    class _Soup:
        def find(self, name: str) -> Any:
            return self

        def find_all(self, names: list[str]) -> list[Any]:
            return []

        def get_text(self, separator: str = "", strip: bool = False) -> str:
            return "extracted text"

    fake_bs4 = MagicMock()
    fake_bs4.BeautifulSoup.return_value = _Soup()
    with patch.dict("sys.modules", {"bs4": fake_bs4}):
        sections = HTML.parse(b"<html></html>", "doc.html", "text/html")
    assert len(sections) == 1
    assert sections[0].source_location == "full file"
    assert sections[0].text == "extracted text"
    assert sections[0].metadata == {"headings": []}


def test_html_parse_extracts_headings() -> None:
    """The metadata includes the text of every h1/h2/h3 element."""

    class _H:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, strip: bool = False) -> str:
            return self._text

    class _Soup:
        def find(self, name: str) -> Any:
            return self

        def find_all(self, names: list[str]) -> list[Any]:
            return [_H("Title"), _H("Subtitle")]

        def get_text(self, separator: str = "", strip: bool = False) -> str:
            return "body text"

    fake_bs4 = MagicMock()
    fake_bs4.BeautifulSoup.return_value = _Soup()
    with patch.dict("sys.modules", {"bs4": fake_bs4}):
        sections = HTML.parse(b"<html></html>", "doc.html", "text/html")
    assert sections[0].metadata["headings"] == ["Title", "Subtitle"]


def test_html_parse_raises_missing_dep_when_bs4_missing() -> None:
    """ImportError → :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"bs4": None}):
        with pytest.raises(MissingDepError, match="beautifulsoup4"):
            HTML.parse(b"<html></html>", "x.html", "text/html")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def test_image_parse_returns_single_section_with_metadata() -> None:
    """Image parser emits a single section carrying the OCR text and image metadata.

    ``pytesseract.image_to_string`` is stubbed to return
    ``"OCR-TEXT"`` so the test verifies that the parser forwards
    OCR output into ``section.text`` rather than leaving it empty.
    """

    class _Image:
        def __init__(self) -> None:
            self.format = "PNG"
            self.size = (640, 480)
            self.mode = "RGBA"
            self.info: dict[str, Any] = {"dpi": (72, 72)}

        def getexif(self) -> dict[Any, Any]:
            return {}

    image_instance = _Image()
    fake_image_class = MagicMock()
    fake_image_class.open.return_value = image_instance
    fake_pytesseract = MagicMock()
    fake_pytesseract.image_to_string.return_value = "OCR-TEXT"
    with patch.dict(
        "sys.modules",
        {
            "PIL": MagicMock(Image=fake_image_class),
            "pytesseract": fake_pytesseract,
        },
    ):
        sections = Image.parse(b"\x89PNG\r\n\x1a\n", "img.png", "image/png")
    assert len(sections) == 1
    assert sections[0].text == "OCR-TEXT"
    assert sections[0].metadata["size"] == (640, 480)
    assert sections[0].metadata["format"] == "PNG"
    assert sections[0].metadata["mode"] == "RGBA"


def test_image_parse_handles_missing_getexif() -> None:
    """Images without ``getexif`` get an empty ``exif`` mapping."""

    class _NoExifImage:
        format = "PNG"
        size = (10, 20)
        mode = "RGB"
        info: dict[Any, Any] = {}

    image_instance = _NoExifImage()
    fake_image_class = MagicMock()
    fake_image_class.open.return_value = image_instance
    with patch.dict("sys.modules", {"PIL": MagicMock(Image=fake_image_class)}):
        sections = Image.parse(b"bytes", "img.png", "image/png")
    assert sections[0].metadata["exif"] == {}


def test_image_parse_with_ocr_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """When pytesseract is available, OCR text populates the section."""

    class _Image:
        format = "PNG"
        size = (10, 20)
        mode = "RGB"
        info: dict[str, Any] = {}

        def getexif(self) -> dict[Any, Any]:
            return {}

    image_instance = _Image()
    fake_image_class = MagicMock()
    fake_image_class.open.return_value = image_instance
    fake_pytesseract = MagicMock()
    fake_pytesseract.image_to_string.return_value = "ocr-extracted"
    with patch.dict(
        "sys.modules",
        {"PIL": MagicMock(Image=fake_image_class), "pytesseract": fake_pytesseract},
    ):
        sections = Image.parse(b"bytes", "img.png", "image/png")
    assert sections[0].text == "ocr-extracted"


def test_image_parse_handles_ocr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pytesseract error yields an empty text section."""

    class _Image:
        format = "PNG"
        size = (10, 20)
        mode = "RGB"
        info: dict[str, Any] = {}

        def getexif(self) -> dict[Any, Any]:
            return {}

    image_instance = _Image()
    fake_image_class = MagicMock()
    fake_image_class.open.return_value = image_instance
    fake_pytesseract = MagicMock()
    fake_pytesseract.image_to_string.side_effect = RuntimeError("ocr-fail")
    with patch.dict(
        "sys.modules",
        {"PIL": MagicMock(Image=fake_image_class), "pytesseract": fake_pytesseract},
    ):
        sections = Image.parse(b"bytes", "img.png", "image/png")
    assert sections[0].text == ""


def test_image_parse_raises_missing_dep_when_pil_missing() -> None:
    """ImportError → :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"PIL": None}):
        with pytest.raises(MissingDepError, match="Pillow"):
            Image.parse(b"bytes", "x.png", "image/png")


# ---------------------------------------------------------------------------
# Office (delegates to loader)
# ---------------------------------------------------------------------------


def test_office_docx_returns_single_section() -> None:
    """DOCX files produce one section with paragraphs joined by newlines."""

    class _Para:
        text = "Hello"

    class _Doc:
        paragraphs = [_Para(), _Para()]
        tables = []

    class _DocumentFactory:
        def __call__(self, _stream: Any) -> Any:
            return _Doc()

    fake_docx = MagicMock()
    fake_docx.Document = _DocumentFactory()
    with patch.dict("sys.modules", {"docx": fake_docx}):
        sections = Office.parse(
            b"bytes",
            "doc.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    assert len(sections) == 1
    assert sections[0].text == "Hello\nHello"
    assert sections[0].source_location == "document"
    assert sections[0].metadata == {"tables": 0}


def test_office_xlsx_returns_one_section_per_worksheet() -> None:
    """XLSX files produce one section per worksheet with rows joined by ' | '."""

    class _Row:
        def __init__(self, values: tuple[Any, ...]) -> None:
            self._values = values

        def __iter__(self) -> Any:
            return iter(self._values)

    class _Ws:
        def __init__(self, name: str) -> None:
            self._name = name

        def iter_rows(self, values_only: bool = True) -> Any:
            return iter([_Row(("a", 1)), _Row(("b", 2))])

    class _Wb:
        def __init__(self, _stream: Any, read_only: bool = True, data_only: bool = True) -> None:
            self.sheetnames = ["Sheet1", "Sheet2"]
            self._sheets = {"Sheet1": _Ws("Sheet1"), "Sheet2": _Ws("Sheet2")}

        def __getitem__(self, name: str) -> Any:
            return self._sheets[name]

        def close(self) -> None:
            pass

    fake_openpyxl = MagicMock()
    fake_openpyxl.load_workbook = _Wb
    with patch.dict("sys.modules", {"openpyxl": fake_openpyxl}):
        sections = Office.parse(
            b"bytes",
            "sheet.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    assert len(sections) == 2
    assert sections[0].source_location == "worksheet Sheet1"
    assert sections[0].text == "a | 1\nb | 2"
    assert sections[0].metadata == {"sheet_name": "Sheet1"}


def test_office_pptx_returns_one_section_per_slide() -> None:
    """PPTX files produce one section per slide."""

    class _Shape:
        text = "Slide text"

    class _Slide:
        def __init__(self) -> None:
            self.shapes = [_Shape()]

    class _Prs:
        def __init__(self) -> None:
            self.slides = [_Slide(), _Slide()]

    class _PresentationFactory:
        def __call__(self, _stream: Any) -> Any:
            return _Prs()

    fake_pptx = MagicMock()
    fake_pptx.Presentation = _PresentationFactory()
    with patch.dict("sys.modules", {"pptx": fake_pptx}):
        sections = Office.parse(
            b"bytes",
            "deck.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    assert len(sections) == 2
    assert sections[0].source_location == "slide 1"
    assert sections[0].text == "Slide text"
    assert sections[1].source_location == "slide 2"


def test_office_raises_missing_dep_for_docx() -> None:
    """Missing python-docx raises :class:`MissingDepError`."""
    with patch.dict("sys.modules", {"docx": None}):
        with pytest.raises(MissingDepError, match="python-docx"):
            Office.parse(b"bytes", "doc.docx", "application/msword")


def test_office_unknown_extension_returns_empty() -> None:
    """Unknown Office extensions return an empty list."""
    assert Office.parse(b"bytes", "file.xyz", "application/x-unknown") == []


# ---------------------------------------------------------------------------
# Csv
# ---------------------------------------------------------------------------


def test_csv_parse_returns_single_section() -> None:
    """CSV content is a single :class:`ParsedSection`."""
    sections = Csv.parse(b"a,b,c\n1,2,3\n", "data.csv", "text/csv")
    assert len(sections) == 1
    assert sections[0].section_index == 0
    assert sections[0].text == "a,b,c\n1,2,3\n"
    assert sections[0].source_location == "full file"


def test_csv_parse_replaces_invalid_utf8() -> None:
    """Invalid UTF-8 bytes in CSV are replaced."""
    sections = Csv.parse(b"hello \xff world", "data.csv", "text/csv")
    assert "hello" in sections[0].text
    assert "world" in sections[0].text


# ---------------------------------------------------------------------------
# Txt
# ---------------------------------------------------------------------------


def test_txt_parse_returns_single_section() -> None:
    """Plain text is a single :class:`ParsedSection` decoded as UTF-8."""
    sections = Txt.parse(b"hello world", "doc.txt", "text/plain")
    assert len(sections) == 1
    assert sections[0].section_index == 0
    assert sections[0].text == "hello world"
    assert sections[0].source_location == "full file"


def test_txt_parse_replaces_invalid_utf8() -> None:
    """Invalid UTF-8 bytes are replaced with the replacement character."""
    sections = Txt.parse(b"hello \xff world", "doc.txt", "text/plain")
    assert "hello" in sections[0].text
    assert "world" in sections[0].text


# ---------------------------------------------------------------------------
# File abstract base
# ---------------------------------------------------------------------------


def test_file_abstract_parse_raises() -> None:
    """Subclasses must override :meth:`File.parse`."""
    with pytest.raises(TypeError):
        File.parse(b"bytes", "x", "text/plain")  # type: ignore[abstract]
