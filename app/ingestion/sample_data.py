"""Generate sample PDF documents for the demo dataset."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.pdfgen import canvas  # type: ignore[import-untyped]


SAMPLE_DOCUMENTS = {
    "A": "Company A reported strong quarterly revenue growth, expanding margins, and healthy guidance.",
    "B": "Company B delivered stable earnings, improved cash flow, and disciplined capital allocation.",
    "C": "Company C posted record cloud demand, higher operating income, and continued product momentum.",
    "D": "Company D saw robust subscription growth, resilient retention, and stronger-than-expected profit.",
    "E": "Company E highlighted supply-chain recovery, margin improvement, and a positive outlook for next quarter.",
}


def create_sample_pdfs(output_dir: Path) -> None:
    """Create sample earnings-call PDFs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for company, text in SAMPLE_DOCUMENTS.items():
        path = output_dir / f"{company}_earnings_q4_2024.pdf"
        _write_pdf(path, f"Company {company} Earnings Call", text)


def _write_pdf(path: Path, title: str, body: str) -> None:
    buffer = canvas.Canvas(str(path), pagesize=letter)
    buffer.setTitle(title)
    buffer.drawString(72, 750, title)
    buffer.drawString(72, 720, body)
    buffer.showPage()
    buffer.save()
