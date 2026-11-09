"""Retrieval eval: how often does search put the right passage in the top k?

Run with `make eval-retrieval`. It ingests evals/retrieval/corpus into an in-process
Qdrant (no Docker needed), runs the labelled queries against four setups, and
writes evals/retrieval/report.md.

Metrics, for each query, over the top k results:
- hit@k: 1 if any relevant passage is in the top k, else 0.
- MRR (mean reciprocal rank): 1/rank of the first relevant passage (0 if none).
  Rewards putting the answer *first*, not just somewhere in the list.
- nDCG@k: like MRR, but credits every relevant passage, discounted by log2(rank + 1),
  divided by the best possible score so 1.0 is a perfect ranking.
Each is averaged over all queries.
"""

import argparse
import asyncio
import math
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient

from goalsmith.capabilities.rag.ingest import ingest_files
from goalsmith.capabilities.rag.models import Chunk, SearchHit
from goalsmith.capabilities.rag.rerank import Reranker
from goalsmith.capabilities.rag.retrieve import HybridRetriever, Mode
from goalsmith.capabilities.rag.store import DocStore
from goalsmith.config import Settings, get_settings
from goalsmith.llm import get_embeddings, get_reranker

EVAL_DIR = Path("evals/retrieval")
COLLECTION = "eval"


class Relevant(BaseModel):
    """Where the answer lives. Labels name sections, not chunk ids, so they stay
    valid when chunk sizes change."""

    source: str
    section: str | None = None
    page: int | None = None

    def matches(self, chunk: Chunk) -> bool:
        return (
            chunk.source == self.source
            and (self.section is None or self.section in chunk.section)
            and (self.page is None or chunk.page == self.page)
        )


class LabelledQuery(BaseModel):
    id: str
    query: str
    relevant: list[Relevant] = Field(min_length=1)


def load_queries(path: Path) -> list[LabelledQuery]:
    return [LabelledQuery.model_validate(item) for item in yaml.safe_load(path.read_text())]


# --- metrics ---------------------------------------------------------------------


def gains(hits: Sequence[SearchHit], relevant: Sequence[Relevant]) -> list[int]:
    """1 where a hit matches a label not already matched higher up, else 0."""
    found: set[int] = set()
    result = []
    for hit in hits:
        new = next(
            (i for i, r in enumerate(relevant) if i not in found and r.matches(hit.chunk)), None
        )
        if new is not None:
            found.add(new)
        result.append(int(new is not None))
    return result


def hit_rate(g: Sequence[int]) -> float:
    return float(any(g))


def reciprocal_rank(g: Sequence[int]) -> float:
    return next((1.0 / rank for rank, gain in enumerate(g, start=1) if gain), 0.0)


def ndcg(g: Sequence[int], n_relevant: int) -> float:
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(g, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_relevant, len(g)) + 1))
    return dcg / ideal if ideal else 0.0


# --- running ---------------------------------------------------------------------


class Scores(BaseModel):
    setup: str
    hit_at_k: float
    mrr: float
    ndcg: float
    ms_per_query: float
    misses: list[str]  # ids of queries with no relevant passage in the top k


async def evaluate(
    setup: str,
    retriever: HybridRetriever,
    queries: Sequence[LabelledQuery],
    *,
    collection: str = COLLECTION,
    k: int = 5,
) -> Scores:
    hits_, rrs, ndcgs, misses = [], [], [], []
    start = time.perf_counter()
    for q in queries:
        g = gains(await retriever.search(q.query, collection, k=k), q.relevant)
        hits_.append(hit_rate(g))
        rrs.append(reciprocal_rank(g))
        ndcgs.append(ndcg(g, len(q.relevant)))
        if not any(g):
            misses.append(q.id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    n = len(queries)
    return Scores(
        setup=setup,
        hit_at_k=sum(hits_) / n,
        mrr=sum(rrs) / n,
        ndcg=sum(ndcgs) / n,
        ms_per_query=elapsed_ms / n,
        misses=misses,
    )


async def run_eval(
    corpus: Path,
    queries: Sequence[LabelledQuery],
    embeddings: Embeddings,
    reranker: Reranker | None,
    *,
    k: int = 5,
    chunk_chars: int = 1200,
    chunk_overlap: int = 150,
    candidates: int = 20,
) -> list[Scores]:
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = DocStore(client)
        await ingest_files(
            [corpus],
            collection=COLLECTION,
            store=store,
            embeddings=embeddings,
            max_chars=chunk_chars,
            overlap=chunk_overlap,
        )
        setups: list[tuple[str, Mode, Reranker | None]] = [
            ("dense", "dense", None),
            ("bm25", "sparse", None),
            ("hybrid (RRF)", "hybrid", None),
        ]
        if reranker is not None:
            setups.append(("hybrid + rerank", "hybrid", reranker))
        results = []
        for name, mode, rr in setups:
            retriever = HybridRetriever(
                store, embeddings, reranker=rr, candidates=candidates, mode=mode
            )
            results.append(await evaluate(name, retriever, queries, k=k))
        return results
    finally:
        await client.close()


def render(scores: Sequence[Scores], *, k: int, n_queries: int, settings: Settings) -> str:
    rag = settings.rag
    lines = [
        "# Retrieval eval",
        "",
        f"{n_queries} labelled queries over `evals/retrieval/corpus`, top {k}. "
        f"Generated by `make eval-retrieval` on {date.today().isoformat()}.",
        "",
        f"- Embeddings: `{rag.embedding_model}`",
        f"- Reranker: `{rag.rerank_model or 'none'}`",
        f"- Chunks: {rag.chunk_chars} chars, {rag.chunk_overlap} overlap; "
        f"{rag.candidates} candidates per retriever",
        "",
        f"| Setup | hit@{k} | MRR | nDCG@{k} | ms/query | Missed |",
        "|---|---|---|---|---|---|",
    ]
    for s in scores:
        missed = ", ".join(s.misses) or "-"
        lines.append(
            f"| {s.setup} | {s.hit_at_k:.3f} | {s.mrr:.3f} | {s.ndcg:.3f} "
            f"| {s.ms_per_query:.0f} | {missed} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--corpus", type=Path, default=EVAL_DIR / "corpus")
    parser.add_argument("--queries", type=Path, default=EVAL_DIR / "queries.yaml")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "report.md")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    settings = get_settings()
    queries = load_queries(args.queries)
    scores = asyncio.run(
        run_eval(
            args.corpus,
            queries,
            get_embeddings(settings),
            get_reranker(settings),
            k=args.k,
            chunk_chars=settings.rag.chunk_chars,
            chunk_overlap=settings.rag.chunk_overlap,
            candidates=settings.rag.candidates,
        )
    )
    report = render(scores, k=args.k, n_queries=len(queries), settings=settings)
    args.out.write_text(report)
    print(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
