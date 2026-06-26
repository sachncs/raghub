"""Command-line ingestion entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.container import build_container
from app.ingestion.sample_data import create_sample_pdfs


def main() -> None:
    """Ingest documents from a folder."""

    parser = argparse.ArgumentParser(description="Ingest PDF documents into SQLite and Zvec.")
    parser.add_argument("--source", type=Path, default=Path("documents"))
    parser.add_argument("--company", type=str, default=None)
    parser.add_argument("--title-prefix", type=str, default="Earnings Call")
    parser.add_argument("--generate-samples", action="store_true")
    args = parser.parse_args()

    if args.generate_samples:
        create_sample_pdfs(args.source)

    container = build_container()
    for pdf_path in sorted(args.source.glob("*.pdf")):
        company = args.company or pdf_path.stem.split("_", 1)[0]
        title = f"{args.title_prefix}: {pdf_path.stem}"
        container.ingestion_service.ingest_pdf(pdf_path, company=company, title=title)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()

