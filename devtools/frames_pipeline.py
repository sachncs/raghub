"""End-to-end FRAMES benchmark eval against an RAGHub instance.

Two phases:
  1. ``fetch`` — pre-download the union of Wikipedia URLs in the FRAMES
     test split (2,517 unique pages) and clean each page's text into a
     corpus directory.
  2. ``run`` — ingest the corpus into a fresh ``RAG``, run every FRAMES
     question through the answer pipeline, and write a results table
     with the Anyscale-style RAG metrics (recall@k, hit-rate@k,
     precision@k, MRR, MAP, token overlap, numeric tolerance, answer
     relevance, faithfulness/faithfulness-claims).

Skips auth (no ``RagApplication`` login flow) and uses the
``RAG_LLM_*`` env vars for the LLM endpoint. The shared
``httpx.AsyncClient`` in ``LiteLLMProvider`` plus ``asyncio.gather``
in ``Evaluator.evaluate`` keep the cost of 824 LLM calls bounded.

Usage::

    RAG_LLM_BASE_URL=... RAG_LLM_API_KEY=... RAG_LLM_MODEL=... \\
    python devtools/frames_pipeline.py fetch --corpus-dir /tmp/frames
    python devtools/frames_pipeline.py run   --corpus-dir /tmp/frames

Add ``--with-judge`` to the ``run`` subcommand to layer an LLM-as-a-
judge faithfulness pass on top of the deterministic one. That
quadruples the LLM call count (one extra judge call per question) so
keep it off by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import statistics
import time
import unicodedata
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from raghub.config import Settings
from raghub.eval import Frames
from raghub.rag import RAG

# ---------------------------------------------------------------------------
# Phase 1: Wikipedia pre-fetch
# ---------------------------------------------------------------------------


_FETCH_CONCURRENCY = 8
HTTP_OK = 200


async def _fetch_one(client: Any, url: str, out_dir: Path) -> tuple[str, bool]:
    """Download and clean a single Wikipedia page.

    Returns ``(url, ok)``. Skips pages that already exist on disk.
    """
    slug = _slug_from_url(url)
    if not slug:
        return url, False
    path = out_dir / f"{slug}.txt"
    if path.exists() and path.stat().st_size > 0:
        return url, True
    try:
        response = await client.get(url, timeout=20.0, follow_redirects=True)
    except Exception:  # pragma: no cover - network errors
        return url, False
    if response.status_code != HTTP_OK:
        return url, False
    text = _html_to_text(response.text)
    if not text:
        return url, False
    path.write_text(text, encoding="utf-8")
    return url, True


def _slug_from_url(url: str) -> str:
    """Encode a Wikipedia URL into a stable filename token."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return f"wiki-{digest}"


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text suitable for chunked ingest.

    The FRAMES test set is anchored on Wikipedia, so paragraph-
    preserving extraction is the right granularity. ``main`` and
    ``body`` are preferred when present; otherwise we fall back to
    the whole document.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    target = soup.find("main") or soup.find("body") or soup
    for tag in target.find_all(["script", "style", "sup", "table.infobox"]):
        tag.decompose()
    text = target.get_text(separator="\n", strip=True)
    return unicodedata.normalize("NFKC", text)


async def fetch_corpus(corpus_dir: Path, *, force: bool = False) -> None:
    """Pre-fetch every FRAMES Wikipedia URL into ``corpus_dir``."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for child in corpus_dir.iterdir():
            if child.is_file():
                child.unlink()
    fb = Frames()
    rows = await asyncio.to_thread(fb.ensure_examples)
    urls: set[str] = set()
    for row in rows:
        for u in row.get("wiki_links", []) or []:
            urls.add(u)
    urls = sorted(urls)
    print(f"fetching {len(urls)} unique Wikipedia pages into {corpus_dir} ...")

    import httpx

    sem = asyncio.Semaphore(_FETCH_CONCURRENCY)
    ok = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": "raghub-frames-pipeline/1.0"},
        follow_redirects=True,
    ) as client:

        async def bound(u: str) -> tuple[str, bool]:
            async with sem:
                return await _fetch_one(client, u, corpus_dir)

        for batch in asyncio.as_completed([bound(u) for u in urls]):
            _, fetched_ok = await batch
            if fetched_ok:
                ok += 1
    print(f"fetched {ok}/{len(urls)} pages")


# ---------------------------------------------------------------------------
# Phase 2: end-to-end FRAMES run
# ---------------------------------------------------------------------------


def _build_settings(corpus_dir: Path) -> Settings:
    """Settings for the FRAMES eval run."""
    base = Settings.load()
    return base.model_copy(
        update={
            "data_dir": str(corpus_dir),
            "allow_passwordless_login": False,
            "log_level": "WARNING",
            "chunk_size_words": 500,
            "chunk_overlap_words": 100,
            "top_k": 5,
        }
    )


async def _run_pipeline(
    corpus_dir: Path,
    *,
    examples: int = 0,
    max_workers: int = 4,
    with_judge: bool = False,
) -> list[dict[str, Any]]:
    """Ingest ``corpus_dir`` and run the FRAMES eval end-to-end."""
    settings = _build_settings(corpus_dir)
    rag = RAG(settings=settings)
    print(f"[frames] initializing RAG (data_dir={corpus_dir})", flush=True)
    rag.initialize()
    print(f"[frames] ingesting corpus with {max_workers} workers", flush=True)
    t0 = time.perf_counter()
    await rag.ingest_dir(
        corpus_dir,
        metadata={"source": "frames-wikipedia"},
        user=None,
        show_progress=True,
        max_workers=max_workers,
    )
    print(f"[frames] ingest done in {time.perf_counter() - t0:.1f}s", flush=True)

    fb = Frames()
    rows = await asyncio.to_thread(fb.ensure_examples)
    if examples:
        rows = rows[:examples]
    print(f"[frames] running {len(rows)} queries through rag.aquery() + LLM ...", flush=True)
    t0 = time.perf_counter()

    async def factory(example: dict[str, Any]) -> tuple[str, list[str], list[str], list[str]]:
        # Returns the answer plus the contexts, retrieved ids, and
        # relevant ids so the Evaluator can compute the full set of
        # Anyscale metrics (recall@k, hit-rate@k, MAP, faithfulness).
        response = await rag.aquery(example["question"], top_k=5)
        source_chunks = list(getattr(response, "source_chunks", []) or [])
        contexts = [sc.quote for sc in source_chunks]
        retrieved_ids = [sc.chunk_id for sc in source_chunks]
        return (response.answer, contexts, retrieved_ids, example.get("wiki_links", []))

    results = await fb.evaluate(rows, response_factory=factory)
    print(f"[frames] queries done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Optional: LLM-as-a-judge faithfulness pass (off by default).
    if with_judge:
        await _judge_faithfulness(rag, rows, results)

    rag.shutdown()
    return [_result_to_dict(r) for r in results]


async def _judge_faithfulness(rag: RAG, rows: list[dict[str, Any]], results: list[Any]) -> None:
    """Optional LLM-as-a-judge faithfulness pass.

    Skipped by default; the devtools CLI enables it with
    ``--with-judge``. Adds one extra LLM call per question to
    cross-check the deterministic ``faithfulness_claims`` metric.
    """
    from raghub.llm import LiteLLMProvider

    llm = LiteLLMProvider(
        model=os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("RAG_LLM_API_KEY"),
        api_base=os.environ.get("RAG_LLM_BASE_URL"),
    )

    async def judge_one(predicted: str, question: str, context: str) -> str:
        prompt = (
            "Does the answer only use claims supported by the context?\n"
            f"Context: {context[:1000]}\n"
            f"Answer: {predicted}\n"
            "Reply with a single token: SUPPORTED, PARTIAL, or UNSUPPORTED."
        )
        return await llm.async_generate(
            system_prompt=prompt,
            question="",
            context=[],
        )

    for row, result in zip(rows, results, strict=False):
        predicted = result.details.get("predicted", "")
        if not predicted:
            continue
        verdict = await judge_one(predicted, row["question"], "")
        result.metrics["llm_judgement"] = (
            0.0 if "UNSUPPORTED" in verdict else (0.5 if "PARTIAL" in verdict else 1.0)
        )


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an Result to a JSON-safe dict."""
    return {
        "example_id": result.example_id,
        "metrics": dict(result.metrics),
        "passed": bool(result.passed),
        "details": {k: v for k, v in result.details.items()},
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Pretty-print aggregate metrics for the full FRAMES run."""
    if not rows:
        print("no results")
        return
    metric_keys: set[str] = set()
    for r in rows:
        metric_keys.update(r["metrics"].keys())
    metric_keys = sorted(metric_keys)
    headers = ["metric", "mean", "median", "p25", "p75"]
    summary: list[list[str]] = [headers]
    summary.append(["-" * len(h) for h in headers])
    for k in metric_keys:
        values = sorted(r["metrics"].get(k, 0.0) for r in rows)
        if not values:
            continue
        summary.append(
            [
                k,
                f"{statistics.mean(values):.3f}",
                f"{statistics.median(values):.3f}",
                f"{values[len(values) // 4]:.3f}",
                f"{values[3 * len(values) // 4]:.3f}",
            ]
        )
    widths = [max(len(row[i]) for row in summary) for i in range(len(headers))]
    print()
    for line in summary:
        print("  ".join(line[i].ljust(widths[i]) for i in range(len(headers))))


async def _run_main(args: argparse.Namespace) -> int:
    rows = await _run_pipeline(
        Path(args.corpus_dir),
        examples=args.examples,
        max_workers=args.max_workers,
        with_judge=args.with_judge,
    )
    print(f"\nFRAMES results ({len(rows)} questions):\n")
    _print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRAMES benchmark pre-fetch + end-to-end RAG eval."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser(
        "fetch",
        help="Pre-download the union of FRAMES Wikipedia URLs into a corpus dir.",
    )
    p_fetch.add_argument(
        "--corpus-dir",
        type=str,
        default="/tmp/frames_corpus",
        help="Output directory for the downloaded Wikipedia pages.",
    )
    p_fetch.add_argument(
        "--force",
        action="store_true",
        help="Re-download every URL even if the local copy exists.",
    )

    p_run = sub.add_parser(
        "run",
        help="Ingest the corpus and run the FRAMES eval end-to-end.",
    )
    p_run.add_argument(
        "--corpus-dir",
        type=str,
        default="/tmp/frames_corpus",
        help="Directory holding the pre-fetched Wikipedia pages.",
    )
    p_run.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of FRAMES examples (0 = all 824).",
    )
    p_run.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="ProcessPoolExecutor workers for the corpus ingest.",
    )
    p_run.add_argument(
        "--with-judge",
        action="store_true",
        help="Run an LLM-as-a-judge faithfulness pass on top of the deterministic one.",
    )
    p_run.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write per-question results as JSON.",
    )

    args = parser.parse_args()
    if args.cmd == "fetch":
        asyncio.run(fetch_corpus(Path(args.corpus_dir), force=args.force))
        return 0
    if args.cmd == "run":
        # Hardcoded dev-only env defaults so the script works without
        # the user having to source .env first. The LLM creds are
        # still loaded from the shell or .env; the rest are sandbox
        # tunables.
        os.environ.setdefault("RAG_PROFILE", "development")
        os.environ.setdefault("RAG_DATA_DIR", "/tmp/frames_rag")
        # zvec_dir was removed; the store now uses SqliteVectorStore

        os.environ.setdefault("RAG_LOG_LEVEL", "WARNING")
        os.environ.setdefault("RAG_ENVIRONMENT", "development")
        # Random per-run secret; Settings only requires JWT_SECRET to be
        # non-empty in the production profile.
        os.environ.setdefault("JWT_SECRET", secrets.token_hex(32))
        os.environ.setdefault("ALLOW_PASSWORDLESS", "0")
        # Load .env if available (preserves user overrides via
        # setdefault semantics).
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        raise SystemExit(asyncio.run(_run_main(args)))
    return 0


if __name__ == "__main__":
    main()
