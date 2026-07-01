from __future__ import annotations

import pytest

from raghub.documents.validation import detect_mime_type, validate_upload
from raghub.exceptions import DocumentError


class TestDetectMimeType:
    def test_known_extension_with_magic_match(self) -> None:
        mime = detect_mime_type("report.pdf", b"%PDF-1.4 content")
        assert mime == "application/pdf"

    def test_known_extension_no_magic(self) -> None:
        mime = detect_mime_type("data.csv", b"a,b,c\n1,2,3\n")
        assert mime == "text/csv"

    def test_unknown_extension_default(self) -> None:
        mime = detect_mime_type("file.xyz", b"some content")
        assert mime == "application/octet-stream"

    def test_magic_byte_mismatch_raises(self) -> None:
        with pytest.raises(DocumentError, match="magic bytes do not match"):
            detect_mime_type("evil.pdf", b"not really a pdf")

    def test_magic_byte_mismatch_jpeg(self) -> None:
        with pytest.raises(DocumentError, match="magic bytes do not match"):
            detect_mime_type("photo.jpg", b"GIF89afake content")


class TestValidateUpload:
    def test_valid_upload(self) -> None:
        mime = validate_upload("doc.pdf", b"%PDF-1.4 content", max_bytes=10_000)
        assert mime == "application/pdf"

    def test_missing_filename_raises(self) -> None:
        with pytest.raises(DocumentError, match="Filename must have an extension"):
            validate_upload("", b"content", max_bytes=100)

    def test_no_extension_raises(self) -> None:
        with pytest.raises(DocumentError, match="Filename must have an extension"):
            validate_upload("README", b"content", max_bytes=100)

    def test_exceeds_max_size_raises(self) -> None:
        with pytest.raises(DocumentError, match="Upload exceeds maximum size"):
            validate_upload("file.pdf", b"x" * 100, max_bytes=50)

    def test_unsupported_mime_raises(self) -> None:
        with pytest.raises(DocumentError, match="Unsupported file type"):
            validate_upload("file.xyz", b"content", max_bytes=10_000)
