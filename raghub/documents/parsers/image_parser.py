"""Image parser.

Decodes image bytes via :mod:`PIL`, optionally runs OCR via
:mod:`pytesseract` if it is installed, and returns the metadata
(format, dimensions, EXIF) along with any extracted text.

OCR is opportunistic: a missing or broken ``pytesseract``
installation silently yields an empty text string, leaving the image
metadata still useful for retrieval.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from .base import FileParser, ParsedSection


class ImageParser(FileParser):
    """Parser for PNG/JPEG/GIF/BMP/TIFF/WebP images."""

    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        """Decode an image and extract its metadata (and optional OCR text).

        Args:
            file_bytes: Raw image bytes.
            file_name: Original filename (unused).
            mime_type: MIME type (unused).

        Returns:
            A single-element list with one :class:`ParsedSection`.
            ``source_location`` is ``"image"``. ``metadata`` carries
            ``format``, ``size``, ``mode``, and ``exif`` (a dict of
            EXIF tag → stringified value, empty when EXIF is absent).
            ``text`` contains the OCR result, or ``""`` if
            :mod:`pytesseract` is unavailable or fails.
        """
        image = Image.open(BytesIO(file_bytes))
        text = ""
        try:
            # ``pytesseract`` is an optional dependency: importing it
            # only here keeps the parser importable when OCR is not
            # needed, and the ``try/except`` below tolerates missing
            # system binaries.
            import pytesseract  # type: ignore[import-untyped]

            text = pytesseract.image_to_string(image)
        except Exception:
            # Silent fallback: the metadata alone is still useful
            # for retrieval, so a missing OCR stack is not a failure.
            text = ""
        exif_data = image.getexif() if hasattr(image, "getexif") else {}
        metadata = {
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "exif": {k: str(v) for k, v in exif_data.items()} if exif_data else {},
        }
        return [
            ParsedSection(
                section_index=0,
                source_location="image",
                text=text,
                metadata=metadata,
            )
        ]
