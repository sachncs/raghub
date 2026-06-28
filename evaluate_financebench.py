"""FinanceBench evaluation script.

Downloads the open-source FinanceBench dataset, ingests PDFs,
and evaluates the RAG system against gold answers.

Usage:  python evaluate_financebench.py
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(tempfile.gettempdir()) / "financebench_eval"
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
META_FILE = DATA_DIR / "document_info.jsonl"
RESULTS_FILE = DATA_DIR / "results.json"
SAMPLE_SIZE = int(os.environ.get("FINANCEBENCH_SAMPLE", "10"))


def download_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if not QUESTIONS_FILE.exists():
        print("Downloading questions...")
        urlretrieve(
            "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl",
            str(QUESTIONS_FILE),
        )

    if not META_FILE.exists():
        print("Downloading document metadata...")
        import urllib.request

        req = urllib.request.Request(
            "https://api.github.com/repos/patronus-ai/financebench/contents/data/financebench_document_information.jsonl",
            headers={"Accept": "application/vnd.github.v3.raw"},
        )
        with urllib.request.urlopen(req) as r:
            META_FILE.write_bytes(r.read())

    print(f"Questions: {QUESTIONS_FILE.stat().st_size:,} bytes")
    print(f"Metadata:  {META_FILE.stat().st_size:,} bytes")


def load_data() -> tuple[list[dict], dict[str, dict]]:
    with open(QUESTIONS_FILE) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    meta: dict[str, dict] = {}
    with open(META_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                meta[row["doc_name"]] = row
    return questions, meta


FINANCEBENCH_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"


async def ingest_pdfs(app, token: str, questions: list[dict], meta: dict[str, dict]) -> int:
    """Download and ingest PDFs referenced by the sample questions."""
    from urllib.request import urlretrieve

    doc_names: set[str] = set()
    for q in questions:
        dn = q.get("doc_name", "")
        if dn:
            doc_names.add(dn)
        for evidence in q.get("evidence", []):
            ed = evidence.get("evidence_doc_name") or dn
            if ed:
                doc_names.add(ed)

    print(f"\nNeed {len(doc_names)} unique documents for {len(questions)} questions")
    ingested = 0
    for i, doc_name in enumerate(sorted(doc_names), 1):
        pdf_url = f"{FINANCEBENCH_RAW}/pdfs/{doc_name}.pdf"
        pdf_path = PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            try:
                print(f"  [{i}/{len(doc_names)}] Downloading {doc_name}...")
                urlretrieve(pdf_url, str(pdf_path))
                print(f"         {pdf_path.stat().st_size / 1024:.0f} KB")
            except Exception as e:
                print(f"  [{i}/{len(doc_names)}] {doc_name}: download failed: {e}")
                continue
        with open(pdf_path, "rb") as f:
            content = f.read()
        company = doc_name.split("_")[0]
        try:
            await app.upload_document(
                token=token,
                filename=f"{doc_name}.pdf",
                content=content,
                company=company,
            )
            ingested += 1
        except Exception as e:
            print(f"  [{i}/{len(doc_names)}] {doc_name}: ingest failed: {e}")
    return ingested


async def evaluate(app, token: str, questions: list[dict]) -> list[dict]:
    """Run questions through the RAG pipeline and collect results."""
    results: list[dict] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q.get("financebench_id", i)
        question_text = q["question"]
        gold_answer = q.get("answer", "")
        company = q.get("company", "Unknown")
        print(f"  [{i}/{total}] Q{qid} ({company}): {question_text[:60]}...")
        try:
            response = await app.query(
                token=token,
                question=question_text,
            )
            results.append({
                "financebench_id": qid,
                "company": company,
                "question": question_text,
                "gold_answer": gold_answer,
                "system_answer": response.answer,
                "citation_count": len(response.citations),
                "has_citations": len(response.citations) > 0,
                "error": None,
            })
        except Exception as e:
            results.append({
                "financebench_id": qid,
                "company": company,
                "question": question_text,
                "gold_answer": gold_answer,
                "system_answer": None,
                "citation_count": 0,
                "has_citations": False,
                "error": str(e),
            })
        if (i % 10) == 0:
            _save_results(results)
    _save_results(results)
    return results


def _save_results(results: list[dict]) -> None:
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def report(results: list[dict]) -> None:
    total = len(results)
    errors = [r for r in results if r["error"]]
    with_citations = [r for r in results if r["has_citations"]]
    non_empty = [r for r in results if r.get("system_answer") and r["system_answer"].strip()]

    print("\n" + "=" * 70)
    print("FINANCEBENCH EVALUATION REPORT")
    print("=" * 70)
    print(f"Total questions:      {total}")
    print(f"Answered (non-empty): {len(non_empty)} ({len(non_empty)/total*100:.0f}%)")
    print(f"With citations:       {len(with_citations)} ({len(with_citations)/total*100:.0f}%)")
    print(f"Errors:               {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  - Q{e['financebench_id']}: {e['error']}")
    print()

    for r in results[:5]:
        print("-" * 70)
        print(f"Q{r['financebench_id']} ({r['company']}): {r['question'][:80]}")
        print(f"  Gold:    {r['gold_answer'][:120]}")
        ans = r.get("system_answer", "") or ""
        print(f"  System:  {ans[:120]}")
        print(f"  Cites:   {r['citation_count']}")

    print("\nResults saved to:", RESULTS_FILE)


async def main() -> None:
    print("=" * 70)
    print("FINANCEBENCH EVALUATION")
    print("=" * 70)

    # 1. Download data
    print("\n[1/4] Downloading FinanceBench data...")
    download_data()

    # 2. Load questions
    print("\n[2/4] Loading questions...")
    questions, meta = load_data()
    print(f"  {len(questions)} questions loaded")
    print(f"  {len(meta)} document metadata entries")

    # 3. Build RAG application
    print("\n[3/4] Building RAG application...")
    from raghub.core.container import build_application
    app = await build_application()

    print("Creating admin user for evaluation...")
    admin_email = "financebench@eval.local"
    existing = await app.container.user_store.get_by_email(admin_email)
    if existing is None:
        await app.container.user_store.create_user(
            email=admin_email, password="eval", companies=[], is_admin=True,
        )

    login_resp = await app.login(admin_email, "eval")
    admin_token = login_resp.session_token
    print(f"  Admin token: {admin_token[:16]}...")

    # 4. Ingest PDFs
    print("\n[4/4] Ingesting PDFs...")
    ingested = await ingest_pdfs(app, admin_token, questions[:SAMPLE_SIZE], meta)
    print(f"  {ingested} PDFs ingested")

    # 5. Evaluate
    print(f"\nEvaluating {SAMPLE_SIZE} questions...")
    results = await evaluate(app, admin_token, questions[:SAMPLE_SIZE])

    # 6. Report
    report(results)


if __name__ == "__main__":
    asyncio.run(main())
