from __future__ import annotations

import io

from .base import FileParser, ParsedSection


class OfficeParser(FileParser):
    def parse(self, file_bytes: bytes, file_name: str, mime_type: str) -> list[ParsedSection]:
        ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
        sections: list[ParsedSection] = []

        if mime_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        ) or ext in ("docx", "doc"):
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            text_parts = [para.text for para in doc.paragraphs]
            sections.append(ParsedSection(
                section_index=0,
                source_location="document",
                text="\n".join(text_parts),
                metadata={"tables": len(doc.tables)},
            ))

        elif mime_type in (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ) or ext in ("xlsx", "xls"):
            from openpyxl import load_workbook  # type: ignore[import-untyped]
            wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            for i, ws_name in enumerate(wb.sheetnames, start=1):
                ws = wb[ws_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) if c is not None else "" for c in row)
                    rows.append(row_text)
                sections.append(ParsedSection(
                    section_index=i,
                    source_location=f"worksheet {ws_name}",
                    text="\n".join(rows),
                    metadata={"sheet_name": ws_name},
                ))
            wb.close()

        elif mime_type in (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
        ) or ext in ("pptx", "ppt"):
            from pptx import Presentation
            prs = Presentation(io.BytesIO(file_bytes))
            for i, slide in enumerate(prs.slides, start=1):
                texts = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
                sections.append(ParsedSection(
                    section_index=i,
                    source_location=f"slide {i}",
                    text="\n".join(texts),
                    metadata={"slide_number": i},
                ))

        return sections
