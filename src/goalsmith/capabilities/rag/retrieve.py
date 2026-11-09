"""Hybrid retrieval: dense + BM25, fused with reciprocal rank fusion, then reranked.

Why both?
- Dense (embeddings) finds meaning: "time off after having a baby" -> "parental leave".
- BM25 finds exact words: error codes, product names, rare terms like "HR-204".
Each misses things the other finds, so we run both and merge the rankings.
"""

from collections.abc import Sequence
from typing import Literal

from langchain_core.embeddings import Embeddings

from goalsmith.capabilities.rag.models import SearchHit
from goalsmith.capabilities.rag.rerank import Reranker, rerank
from goalsmith.capabilities.rag.sparse import encode_query
from goalsmith.capabilities.rag.store import DocStore, StoreError

Mode = Literal["dense", "sparse", "hybrid"]

RRF_K = 60
"""Damps the gap between top ranks. 60 is the value from the original RRF paper
(Cormack et al., 2009) and works well without tuning."""


def rrf_fuse(rankings: Sequence[Sequence[SearchHit]], k: int = RRF_K) -> list[SearchHit]:
    """Reciprocal rank fusion: score(chunk) = sum over rankings of 1 / (k + rank).

    It uses only *positions*, not raw scores. That matters because cosine scores
    (0..1) and BM25 scores (0..~20) are on different scales and cannot be added.
    A chunk ranked well by both retrievers beats one ranked first by only one.
    """
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.chunk.source_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            hits.setdefault(key, hit)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [SearchHit(chunk=hits[key].chunk, score=scores[key]) for key in ordered]


class HybridRetriever:
    def __init__(
        self,
        store: DocStore,
        embeddings: Embeddings,
        *,
        reranker: Reranker | None = None,
        candidates: int = 20,
        mode: Mode = "hybrid",
    ) -> None:
        self.store = store
        self._embeddings = embeddings
        self._reranker = reranker
        self._candidates = candidates
        self._mode = mode

    async def search(self, query: str, collection: str, k: int = 5) -> list[SearchHit]:
        if not await self.store.has_collection(collection):
            available = ", ".join(await self.store.collections()) or "none yet"
            raise StoreError(f"No collection {collection!r}. Available: {available}")

        rankings: list[list[SearchHit]] = []
        if self._mode in ("dense", "hybrid"):
            vector = await self._embeddings.aembed_query(query)
            rankings.append(await self.store.dense_search(collection, vector, self._candidates))
        if self._mode in ("sparse", "hybrid"):
            sparse = encode_query(query)
            rankings.append(await self.store.sparse_search(collection, sparse, self._candidates))

        fused = rrf_fuse(rankings)[: self._candidates]
        if self._reranker is not None:
            return await rerank(self._reranker, query, fused, top_k=k)
        return fused[:k]
