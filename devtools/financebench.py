"""Run the Finance evaluation against an LLM-powered RAGHub instance.

Builds a :class:`LiteLLMProvider` from the ``RAG_LLM_BASE_URL``,
``RAG_LLM_API_KEY`` and ``RAG_LLM_MODEL`` env vars; ingests a real
corpus of Finance source 10-K filings; runs every example through
:class:`RAG`'s query path. Errors surface as exceptions — the script
does not paper over them.

The corpus is matched to the dataset — only the 10-K / 10-Q files
matching a question's company are ingested, keeping the in-process
vector store small enough for a fast end-to-end run.

Skips auth entirely (no :class:`RagApplication` login flow).

Usage::

    RAG_LLM_BASE_URL=... RAG_LLM_API_KEY=... RAG_LLM_MODEL=... \\
    python devtools/financebench.py

``--examples N`` runs the first N examples (default: all 150).
``--pipeline NAME`` runs a single pipeline variant; names:
``baseline``, ``small``, ``large``, ``hyde``.
``--pdfs-dir DIR`` points at the source 10-K PDFs (default:
``/tmp/raghub_fb_pdfs``).
``--data-dir DIR`` overrides the per-run data dir.
``--json PATH`` writes raw per-pipeline results to a JSON file.
``--ingest-timeout SECS`` caps the ingest step (default: 600).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raghub.config import Settings
from raghub.evaluation import Finance
from raghub.llm import LiteLLMProvider
from raghub.rag import RAG

# Dev-only defaults for the sweep's local sandbox. These point the
# pipeline at a per-run data dir under /tmp and silence the noisy
# logs. Override via the shell or a sibling .env. The RAG_LLM_*
# variables stay in .env — keys never go in source. The secret is
# intentionally NOT here: unset, Settings falls back to a per-process
# random JWT_SECRET in the development profile.
_DEV_ENV_DEFAULTS = {
    "RAG_PROFILE": "development",
    "RAG_DATA_DIR": "/tmp/raghub_fb",
    "RAG_LOG_LEVEL": "WARNING",
    "RAG_ENVIRONMENT": "development",
    "ALLOW_PASSWORDLESS": "0",
}
for _key, _value in _DEV_ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _value)


# Pipeline configurations: chunking size, top_k, and whether to use
# the HyDE query transform.
PIPELINES: dict[str, dict[str, Any]] = {
    "baseline": {
        "chunk_size_words": 500,
        "chunk_overlap_words": 100,
        "top_k": 5,
        "use_hyde": False,
    },
    "small": {
        "chunk_size_words": 200,
        "chunk_overlap_words": 50,
        "top_k": 5,
        "use_hyde": False,
    },
    "large": {
        "chunk_size_words": 1000,
        "chunk_overlap_words": 200,
        "top_k": 5,
        "use_hyde": False,
    },
    "hyde": {
        "chunk_size_words": 500,
        "chunk_overlap_words": 100,
        "top_k": 5,
        "use_hyde": True,
    },
}


@dataclass
class PipelineResult:
    """Aggregate metrics for one pipeline variant.

    Attributes:
        name: Variant label.
        n: Number of examples evaluated.
        pass_rate: Fraction whose answer passed
            ``numeric >= 0.99 or overlap >= 0.6``.
        token_overlap: Mean Jaccard overlap.
        within_tolerance: Mean numeric-tolerance score.
        answer_length: Mean predicted answer length (chars).
    """

    name: str
    n: int = 0
    pass_rate: float = 0.0
    token_overlap: float = 0.0
    within_tolerance: float = 0.0
    answer_length: float = 0.0


def _build_settings(pipeline: str, data_dir: str) -> Settings:
    """Return a :class:`Settings` instance tuned for ``pipeline``."""
    cfg = PIPELINES[pipeline]
    base = Settings.load()
    new_query_transforms = base.query_transforms.model_copy(
        update={"enabled": ["hyde"] if cfg["use_hyde"] else []}
    )
    overrides = {
        "chunk_size_words": cfg["chunk_size_words"],
        "chunk_overlap_words": cfg["chunk_overlap_words"],
        "top_k": cfg["top_k"],
        "data_dir": data_dir,
        "allow_passwordless_login": False,
        "log_level": "WARNING",
    }
    base_with_new = base.model_copy(
        update={**overrides, "query_transforms": new_query_transforms}
    )
    return Settings.model_validate(base_with_new.model_dump())


def _build_llm() -> LiteLLMProvider:
    """Build a :class:`LiteLLMProvider` from the ``RAG_LLM_*`` env vars."""
    base_url = os.environ.get("RAG_LLM_BASE_URL")
    api_key = os.environ.get("RAG_LLM_API_KEY")
    model = os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini")
    if not base_url or not api_key:
        raise RuntimeError(
            "RAG_LLM_BASE_URL and RAG_LLM_API_KEY must both be set."
        )
    return LiteLLMProvider(model=model, api_key=api_key, api_base=base_url)


def _normalise_company(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _match_pdfs(
    examples: list[dict[str, Any]], pdfs_dir: Path, one_per_company: bool = True
) -> list[Path]:
    """Return the PDFs to ingest for ``examples``.

    The first matching PDF per company is kept (``one_per_company=True``)
    to keep the vector store small. Pass ``one_per_company=False`` to
    ingest every 10-K / 10-Q that mentions a question's company.
    """
    needed: set[str] = {
        _normalise_company(ex.get("company", "")) for ex in examples
    }
    needed.discard("")
    matched: list[Path] = []
    seen_companies: set[str] = set()
    for pdf in sorted(pdfs_dir.glob("*.pdf")):
        company = _normalise_company(pdf.stem.split("_")[0])
        if company in needed:
            if one_per_company and company in seen_companies:
                continue
            matched.append(pdf)
            seen_companies.add(company)
    return matched


def _stage_pdfs(pdfs: list[Path], staged_dir: Path) -> None:
    """Replace the contents of ``staged_dir`` with the matched PDFs.

    Wipes any leftover PDFs from a previous run so the in-process
    index never sees stale files.
    """
    import shutil

    staged_dir.mkdir(parents=True, exist_ok=True)
    for old in staged_dir.iterdir():
        old.unlink()
    for pdf in pdfs:
        shutil.copy2(pdf, staged_dir / pdf.name)


def _build_pdf_converter() -> Any:
    """Build a pypdf-based PDF converter inline.

    The default :class:`MarkerConverter` fails to load its detection
    model on this environment (``KeyError: 'encoder'``); the
    :class:`PlainTextConverter` decodes bytes as UTF-8 and produces
    garbage for PDF binary. This converter reads the PDF page-by-page
    with pypdf, joins the text, and wraps it in a
    :class:`Bundle` the same way the rest of the pipeline
    expects.
    """
    from io import BytesIO

    from pypdf import PdfReader

    from raghub.models import (
        BlockKind,
        Bundle,
        DocumentBlock,
        DocumentSection,
        deterministic_id,
    )

    class PdfConverter:
        """Pypdf-backed PDF text extractor."""

        def convert(
            self,
            *,
            source_uri: str,
            file_bytes: bytes,
            mime_type: str = "",
            language: str = "",
            metadata: dict[str, Any] | None = None,
        ) -> Bundle:
            reader = PdfReader(BytesIO(file_bytes))
            page_texts: list[str] = []
            for page in reader.pages:
                page_texts.append(page.extract_text() or "")
            text = "\n\n".join(page_texts)
            section = DocumentSection(
                section_id=deterministic_id("section", source_uri, "auto"),
                index=0,
                heading="",
                blocks=[DocumentBlock(kind=BlockKind.TEXT, content=text)],
                page_numbers=list(range(1, len(page_texts) + 1)),
                source_location=f"{source_uri}#0",
            )
            return Bundle(
                bundle_id=deterministic_id("bundle", source_uri),
                source_uri=source_uri,
                mime_type=mime_type or "application/pdf",
                language=language,
                metadata=metadata or {},
                sections=[section],
            )

        async def aconvert(
            self,
            *,
            source_uri: str,
            file_bytes: bytes,
            mime_type: str = "",
            language: str = "",
            metadata: dict[str, Any] | None = None,
        ) -> Bundle:
            return self.convert(
                source_uri=source_uri,
                file_bytes=file_bytes,
                mime_type=mime_type,
                language=language,
                metadata=metadata,
            )

    return PdfConverter()


async def _run_pipeline(
    pipeline: str,
    examples: list[dict[str, Any]],
    data_dir: str,
    llm: LiteLLMProvider,
    pdfs_dir: Path,
    ingest_timeout_secs: float,
) -> PipelineResult:
    """Index the matched PDFs, run every example, return aggregate metrics."""
    t_total = time.perf_counter()
    print(f"[{pipeline}] building settings...", flush=True)
    settings = _build_settings(pipeline, data_dir)
    print(f"[{pipeline}] building PDF converter...", flush=True)
    converter = _build_pdf_converter()
    print(f"[{pipeline}] constructing RAG facade...", flush=True)
    rag = RAG(settings=settings, llm=llm, converter=converter)
    print(f"[{pipeline}] calling RAG.initialize()...", flush=True)
    rag.initialize()
    print(f"[{pipeline}] RAG.initialize() done in {time.perf_counter() - t_total:.1f}s", flush=True)

    print(f"[{pipeline}] matching PDFs to {len(examples)} examples...", flush=True)
    pdfs = _match_pdfs(examples, pdfs_dir)
    if not pdfs:
        raise RuntimeError(
            f"no PDFs matched any of the {len(examples)} example companies; "
            f"check that {pdfs_dir} contains the right files."
        )
    print(f"[{pipeline}] matched {len(pdfs)} PDFs in {time.perf_counter() - t_total:.1f}s", flush=True)

    print(f"[{pipeline}] staging {len(pdfs)} PDFs...", flush=True)
    staged_dir = Path(data_dir) / "staged"
    _stage_pdfs(pdfs, staged_dir)
    print(f"[{pipeline}] staging done in {time.perf_counter() - t_total:.1f}s", flush=True)

    print(f"[{pipeline}] ingesting {len(pdfs)} PDFs (timeout {ingest_timeout_secs}s)...", flush=True)
    t0 = time.perf_counter()
    async with asyncio.timeout(ingest_timeout_secs):
        await rag.ingest_directory_async(
            staged_dir,
            metadata={"source": "financebench-pdfs", "pipeline": pipeline},
            user=None,
            show_progress=True,
        )
    print(
        f"[{pipeline}] ingest done in {time.perf_counter() - t0:.1f}s "
        f"(total elapsed {time.perf_counter() - t_total:.1f}s)"
    )

    evaluator = Finance()
    n_workers = max(1, min(8, (multiprocessing.cpu_count() or 1)))
    print(
        f"[{pipeline}] running {len(examples)} queries through rag.aquery() + LLM "
        f"(up to {n_workers} concurrent)...",
        flush=True,
    )
    t0 = time.perf_counter()

    sem = asyncio.Semaphore(n_workers)

    async def one_query(question: str) -> str:
        async with sem:
            return (await rag.aquery(question)).answer

    async def factory(example: dict[str, Any]) -> str:
        question = example.get("question") or example.get("query") or ""
        return await one_query(question)

    results = await evaluator.evaluate(examples, response_factory=factory)
    print(f"[{pipeline}] queries done in {time.perf_counter() - t0:.1f}s")
    rag.shutdown()

    if not results:
        return PipelineResult(name=pipeline)

    metrics_avg = {
        name: statistics.mean(r.metrics.get(name, 0.0) for r in results)
        for name in {k for r in results for k in r.metrics}
    }
    return PipelineResult(
        name=pipeline,
        n=len(results),
        pass_rate=statistics.mean(1.0 if r.passed else 0.0 for r in results),
        token_overlap=metrics_avg.get("token_overlap", 0.0),
        within_tolerance=metrics_avg.get("within_tolerance", 0.0),
        answer_length=statistics.mean(
            len(r.details.get("predicted") or "") for r in results
        ),
    )


def _print_table(results: list[PipelineResult]) -> None:
    headers = [
        "pipeline", "n", "pass_rate", "token_overlap",
        "numeric", "answer_len",
    ]
    rows = [
        [
            r.name,
            str(r.n),
            f"{r.pass_rate:.3f}",
            f"{r.token_overlap:.3f}",
            f"{r.within_tolerance:.3f}",
            f"{r.answer_length:.0f}",
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "  "

    def _line(cells: list[str]) -> str:
        return sep.join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    print()
    print(_line(headers))
    print(_line(["-" * w for w in widths]))
    for row in rows:
        print(_line(row))


async def _async_main(args: argparse.Namespace) -> int:
    print("[main] building LLM provider from env...", flush=True)
    llm = _build_llm()
    print(f"[main] loading {args.examples} Finance examples...", flush=True)
    examples = Finance().ensure_examples()[: args.examples]
    if not examples:
        print("no Finance examples available; aborting")
        return 1
    print(f"[main] loaded {len(examples)} examples", flush=True)

    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.is_dir():
        print(f"PDF directory not found: {pdfs_dir}")
        return 1
    print(f"[main] PDF source dir: {pdfs_dir}", flush=True)

    pipelines = [args.pipeline] if args.pipeline else list(PIPELINES)
    results: list[PipelineResult] = []
    for name in pipelines:
        print(f"[main] === starting pipeline: {name} ===", flush=True)
        t0 = time.perf_counter()
        result = await _run_pipeline(
            name, examples, args.data_dir, llm, pdfs_dir, args.ingest_timeout,
        )
        print(
            f"[main] pipeline {name} done in {time.perf_counter() - t0:.1f}s"
        )
        results.append(result)
    _print_table(results)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                [
                    {
                        "name": r.name,
                        "n": r.n,
                        "pass_rate": r.pass_rate,
                        "token_overlap": r.token_overlap,
                        "within_tolerance": r.within_tolerance,
                        "answer_length": r.answer_length,
                    }
                    for r in results
                ],
                fh,
                indent=2,
            )
    return 0


def _load_env(path: Path) -> None:
    """Load KEY=VALUE pairs from ``path`` into the environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir / ".env", script_dir.parent / ".env"]:
        _load_env(candidate)

    parser = argparse.ArgumentParser(
        description="Run the Finance sweep across pipeline configurations."
    )
    parser.add_argument(
        "--examples", type=int, default=0,
        help="Number of Finance examples (0 = all 150).",
    )
    parser.add_argument(
        "--pipeline", type=str, default=None, choices=list(PIPELINES),
        help="Run a single pipeline variant instead of the full sweep.",
    )
    parser.add_argument(
        "--pdfs-dir", type=str, default="/tmp/raghub_fb_pdfs",
        help="Directory of source 10-K / 10-Q PDFs.",
    )
    parser.add_argument(
        "--data-dir", type=str, default="/tmp/raghub_financebench",
        help="Per-run data dir (default: /tmp/raghub_financebench).",
    )
    parser.add_argument(
        "--ingest-timeout", type=float, default=600.0,
        help="Max seconds to wait for ingest (default: 600).",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Optional path to write raw results as JSON.",
    )
    args = parser.parse_args()
    if args.examples == 0:
        args.examples = 150
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
