"""Performance benchmark for the RAGHub framework.

Measures:

* **Startup time** — from process start to first ``RAG.health()``
  call.
* **Ingestion throughput** — chunks/second on a synthetic
  document of N tokens.
* **Indexing throughput** — chunks/second through the embedder
  + vector store.
* **Query latency** — p50 / p95 over K queries.
* **Concurrent throughput** — queries/second with C parallel
  users.
* **Memory usage** — peak RSS during the run.

Run::

    python -m bench.benchmark --documents 10 --queries 50 --concurrency 8

The script writes a JSON report to ``bench/report.json`` (or the
path supplied via ``--output``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raghub import RAG  # noqa: E402


def _random_text(words: int, seed: int) -> str:
    rng = random.Random(seed)
    vocab = [
        "the",
        "company",
        "reported",
        "revenue",
        "growth",
        "margin",
        "operating",
        "cash",
        "flow",
        "year",
        "quarter",
        "guidance",
        "forecast",
        "outlook",
        "share",
        "price",
        "earnings",
        "investor",
        "market",
        "capital",
    ]
    return " ".join(rng.choice(vocab) for _ in range(words))


@dataclass
class BenchmarkResult:
    """Aggregate measurements from a benchmark run."""

    documents: int
    words_per_document: int
    queries: int
    concurrency: int
    startup_seconds: float
    ingestion_seconds: float
    ingestion_chunks: int
    ingestion_throughput_chunks_per_sec: float
    query_latency_ms_p50: float
    query_latency_ms_p95: float
    queries_per_second: float
    memory_peak_mb: float | None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return asdict(self)


def _peak_memory_mb() -> float | None:
    """Return peak RSS in MB, or ``None`` when ``psutil`` is missing."""
    try:
        import resource  # noqa: F401
    except ImportError:
        return None
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux, bytes on macOS.
    if sys.platform == "darwin":
        return round(usage.ru_maxrss / (1024 * 1024), 2)
    return round(usage.ru_maxrss / 1024, 2)


async def _run_queries(
    rag: RAG,
    queries: list[str],
    concurrency: int,
) -> list[float]:
    """Run ``queries`` against ``rag`` with ``concurrency`` parallelism.

    Returns the per-query latency in milliseconds.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(q: str) -> float:
        async with sem:
            start = time.perf_counter()
            await rag.aquery(q)
            return (time.perf_counter() - start) * 1000.0

    return await asyncio.gather(*[_one(q) for q in queries])


async def _run(args: argparse.Namespace) -> BenchmarkResult:
    """Execute the benchmark.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The aggregated benchmark result.
    """
    from raghub.converters.plaintext import PlainTextConverter
    from raghub.ingestion.chunkers.word_window import WordWindowChunker

    started = time.perf_counter()
    rag = RAG()
    # Use the plain text converter + a small chunker so the
    # benchmark works without a real PDF or LLM endpoint.
    rag.converter = PlainTextConverter()
    rag.ingest_pipeline.converter = rag.converter
    rag.chunker = WordWindowChunker(chunk_size=20, chunk_overlap=2)
    rag.ingest_pipeline.chunker = rag.chunker
    startup_seconds = time.perf_counter() - started

    # Ingest
    ingest_started = time.perf_counter()
    ingest_chunks = 0
    for i in range(args.documents):
        text = _random_text(args.words_per_document, seed=i)
        result = await rag.aingest(
            text.encode("utf-8"),
            source_uri=f"file://bench/doc-{i}.txt",
            mime_type="text/plain",
        )
        if result.success:
            ingest_chunks += result.outputs.get("chunk_count", 0) or 0
    ingestion_seconds = time.perf_counter() - ingest_started
    throughput = ingest_chunks / ingestion_seconds if ingestion_seconds else 0.0

    # Query latency
    queries = [
        "What was the revenue growth?",
        "How is the cash flow trending?",
        "What is the outlook for next quarter?",
    ]
    latencies = await _run_queries(
        rag, queries * (args.queries // len(queries) + 1), args.concurrency
    )
    latencies = latencies[: args.queries]
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    qps = args.queries / (sum(latencies) / 1000 / args.concurrency)

    return BenchmarkResult(
        documents=args.documents,
        words_per_document=args.words_per_document,
        queries=args.queries,
        concurrency=args.concurrency,
        startup_seconds=round(startup_seconds, 4),
        ingestion_seconds=round(ingestion_seconds, 4),
        ingestion_chunks=ingest_chunks,
        ingestion_throughput_chunks_per_sec=round(throughput, 2),
        query_latency_ms_p50=round(p50, 2),
        query_latency_ms_p95=round(p95, 2),
        queries_per_second=round(qps, 2),
        memory_peak_mb=_peak_memory_mb(),
    )


def main() -> int:
    """Parse CLI args, run the benchmark, and write the report."""
    parser = argparse.ArgumentParser(prog="raghub-benchmark")
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--words-per-document", type=int, default=500)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=str, default="bench/report.json")
    args = parser.parse_args()

    result = asyncio.run(_run(args))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
