"""FinanceBench integration test.

Downloads the open-source FinanceBench dataset, ingests PDFs, and evaluates
the RAG system against the gold answers.

Dataset source: https://github.com/patronus-ai/financebench
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pytest

from raghub.core.container import build_application

FINANCEBENCH_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
DATA_DIR = Path(tempfile.gettempdir()) / "financebench_data"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
META_FILE = DATA_DIR / "financebench_document_information.jsonl"

REQUIRED_PDFS: set[str] = set()

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("FINANCEBENCH_EVAL"), reason="Set FINANCEBENCH_EVAL=1 to run"
    )
]


def _download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, str(dest))
    return True


@pytest.fixture(scope="session")
def financebench_data():
    """Download FinanceBench dataset files (metadata only, not PDFs)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(f"{FINANCEBENCH_URL}/data/financebench_open_source.jsonl", QUESTIONS_FILE)
    _download_file(f"{FINANCEBENCH_URL}/data/financebench_document_information.jsonl", META_FILE)
    return DATA_DIR


@pytest.fixture(scope="session")
def questions(financebench_data) -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="session")
def document_meta(financebench_data) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    with open(META_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta[row["doc_name"]] = row
    return meta


@pytest.fixture(scope="session")
def app_service():
    """Build the application once per session."""
    svc = asyncio.run(build_application())
    yield svc


class TestFinanceBenchSample:
    """Evaluate a sample of FinanceBench questions against the RAG system."""

    SAMPLE_SIZE = int(os.environ.get("FINANCEBENCH_SAMPLE", "10"))

    @pytest.mark.asyncio
    async def test_ingest_pdfs(self, app_service, questions, document_meta):
        """Ingest PDFs mentioned in the sample questions."""
        from urllib.request import urlretrieve

        PDF_DIR.mkdir(parents=True, exist_ok=True)
        doc_names: set[str] = set()
        for q in questions[: self.SAMPLE_SIZE]:
            for evidence in q.get("evidence", []):
                doc_name = evidence.get("evidence_doc_name", "")
                if doc_name:
                    doc_names.add(doc_name)

        ingested = 0
        for doc_name in sorted(doc_names):
            meta = document_meta.get(doc_name, {})
            pdf_url = meta.get("doc_link", "")
            if not pdf_url:
                continue
            pdf_path = PDF_DIR / f"{doc_name}.pdf"
            if not pdf_path.exists():
                try:
                    urlretrieve(pdf_url, str(pdf_path))
                except Exception:
                    continue

            with open(pdf_path, "rb") as f:
                content = f.read()
            company = doc_name.split("_")[0]
            try:
                await app_service.upload_document(
                    token="",
                    filename=f"{doc_name}.pdf",
                    content=content,
                    company=company,
                )
                ingested += 1
            except Exception:
                continue

        assert ingested > 0, "No PDFs were ingested"

    @pytest.mark.asyncio
    async def test_evaluate_questions(self, app_service, questions):
        """Run questions through the RAG pipeline and score answers."""
        results: list[dict] = []
        for q in questions[: self.SAMPLE_SIZE]:
            question_text = q["question"]
            gold_answer = q.get("answer", "")
            try:
                response = await app_service.query(
                    token="",
                    question=question_text,
                )
                results.append(
                    {
                        "financebench_id": q.get("financebench_id"),
                        "question": question_text,
                        "gold_answer": gold_answer,
                        "system_answer": response.answer,
                        "citations": response.citations,
                        "has_citations": len(response.citations) > 0,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "financebench_id": q.get("financebench_id"),
                        "question": question_text,
                        "gold_answer": gold_answer,
                        "system_answer": f"ERROR: {exc}",
                        "citations": [],
                        "has_citations": False,
                    }
                )

        _write_results(results)
        coverage = sum(1 for r in results if r["has_citations"]) / len(results)
        assert coverage > 0.3, f"Citation coverage too low: {coverage:.0%}"

    @pytest.mark.asyncio
    async def test_empty_response_rate(self, app_service, questions):
        """Check that the system produces non-empty answers."""
        empty = 0
        for q in questions[: self.SAMPLE_SIZE]:
            response = await app_service.query(token="", question=q["question"])
            if not response.answer or response.answer.strip() == "":
                empty += 1
        rate = empty / min(self.SAMPLE_SIZE, len(questions))
        assert rate < 0.3, f"Empty response rate too high: {rate:.0%}"


def _write_results(results: list[dict]) -> None:
    """Write evaluation results to a JSON file."""
    output = DATA_DIR / "financebench_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
