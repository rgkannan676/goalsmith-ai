"""Reranking: a slower, more accurate second pass over the fused candidates.

Retrieval embeds the query and each chunk *separately* and compares vectors.
A cross-encoder reads the query and a chunk *together* and outputs one relevance
score, so it catches what vector similarity misses. It is too slow for a whole
collection, so it only reorders the top ~20 candidates.
"""

import asyncio
from collections.abc import Sequence
from typing import Protocol

from goalsmith.capabilities.rag.models import SearchHit


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance of each document to the query; higher is better. Blocking."""
        ...


async def rerank(
    reranker: Reranker, query: str, hits: Sequence[SearchHit], top_k: int
) -> list[SearchHit]:
    if not hits:
        return []
    texts = [hit.chunk.context_text() for hit in hits]
    # The model runs on the CPU and blocks, so keep it off the event loop.
    scores = await asyncio.to_thread(reranker.score, query, texts)
    rescored = [
        SearchHit(chunk=hit.chunk, score=score) for hit, score in zip(hits, scores, strict=True)
    ]
    return sorted(rescored, key=lambda hit: hit.score, reverse=True)[:top_k]
