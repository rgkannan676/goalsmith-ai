# ADR 0005: Hybrid retrieval with BM25 inside Qdrant, RRF fusion and a small reranker

- Status: accepted
- Date: 2026-11-09

## Context
M3 adds document search. Answers must cite their sources, retrieval quality must be measured, and unit tests must stay offline. Anthropic has no embeddings API.

## Decisions

**1. Local models via fastembed.**
- Embeddings: `BAAI/bge-small-en-v1.5` (384 dimensions, 67 MB).
- Reranker: a cross-encoder, `Xenova/ms-marco-MiniLM-L-6-v2` (80 MB).

Both run on the CPU through ONNX. They have no cost and no API key, and they work offline after the first download. They are built by `llm.get_embeddings()` and `llm.get_reranker()`, so the model names live only in config, and they load lazily on first use.

**2. BM25 runs as sparse vectors in Qdrant, not as an in-memory index.**
- `sparse.py` computes the per-document term-frequency part of BM25 at ingest.
- Qdrant applies IDF at query time (`Modifier.IDF`).

So dense and keyword search share one store, and there is no separate index to rebuild or keep in sync. Writing the encoder ourselves (about 40 lines) avoids another model download and keeps the BM25 maths visible. It has no stemming, a known limitation.

**3. Reciprocal rank fusion in Python.** Qdrant can fuse server-side, but `rrf_fuse` is a pure function. That makes it easy to unit test, explain and swap. RRF uses ranks, not scores, because cosine and BM25 scores are on different scales.

**4. Structure-aware chunks with readable source ids.**
- A chunk never crosses a heading or a PDF page.
- The heading path is stored with the chunk and prefixed to the embedded text.
- Ids look like `handbook.md#4`, which is short enough for the model to cite.
- Point ids are uuid5 of the source id, and re-ingesting a file deletes its old chunks first.

**5. Qdrant can run in-process.** Setting `GOALSMITH_RAG__QDRANT_PATH` uses embedded mode (no Docker), and unit tests and the eval use `:memory:`. Docker Compose remains the default for the CLI.

**6. Eval labels name sections, not chunk ids**, so they stay valid when chunk size changes.

## Results (`evals/retrieval/report.md`, 40 queries, top 5)

| Setup | hit@5 | MRR | nDCG@5 |
|---|---|---|---|
| dense | 0.975 | 0.963 | 0.966 |
| bm25 | 1.000 | 0.946 | 0.960 |
| hybrid (RRF) | 1.000 | 0.971 | 0.978 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 |

- Dense search misses the exact-term query "extension 4400", which BM25 finds.
- Hybrid beats each retriever alone.
- Reranking puts the right passage first every time, at about 190 ms per query on a laptop CPU.

## Rejected
- **`BAAI/bge-reranker-base` as the default** (the plan's first idea). It is a 1 GB download, 13 times larger than MiniLM. MiniLM already reaches 1.000 MRR on this set, so the larger model cannot show a gain here. Switch with `GOALSMITH_RAG__RERANK_MODEL`.
- **rank-bm25 in memory.** The index would have to be rebuilt from every chunk at startup and kept in sync with Qdrant.
- **langchain-qdrant / LangChain retrievers.** They hide the parts this project is meant to show: fusion, reranking and payload design.

## Consequences
- The corpus is small (44 chunks), so the scores are near the ceiling. The eval mainly guards against regressions and shows what each stage adds. A larger corpus with near-duplicate documents is needed to tell rerankers apart.
- The embedding size is fixed per collection. Changing the embedding model means re-ingesting into a new collection, and `DocStore` refuses a mismatch with a clear error.
