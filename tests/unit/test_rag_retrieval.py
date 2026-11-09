"""Store, ingest pipeline and hybrid retrieval, on Qdrant's in-process mode (no Docker)."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from goalsmith.capabilities.rag.ingest import ingest_files
from goalsmith.capabilities.rag.models import Chunk, SearchHit
from goalsmith.capabilities.rag.retrieve import HybridRetriever, rrf_fuse
from goalsmith.capabilities.rag.store import DocStore, StoreError, check_collection_name
from tests.fixtures.fake_rag import HashEmbeddings, KeywordReranker

HANDBOOK = """\
# Handbook

## Parental leave

Employees get up to 16 weeks of paid parental leave after a birth or adoption.

## Expenses

Submit expense receipts within 30 days using form HR-204.

## Security

Report a lost laptop to the security desk within one hour.
"""


@pytest.fixture
async def store() -> AsyncIterator[DocStore]:
    client = AsyncQdrantClient(location=":memory:")
    yield DocStore(client)
    await client.close()


@pytest.fixture
def embeddings() -> HashEmbeddings:
    return HashEmbeddings()


async def ingest(store: DocStore, embeddings: HashEmbeddings, *paths: Path) -> None:
    await ingest_files(
        paths,
        collection="docs",
        store=store,
        embeddings=embeddings,
        max_chars=500,
        overlap=0,
    )


@pytest.fixture
async def indexed(tmp_path: Path, store: DocStore, embeddings: HashEmbeddings) -> DocStore:
    (tmp_path / "handbook.md").write_text(HANDBOOK)
    await ingest(store, embeddings, tmp_path / "handbook.md")
    return store


def hit(source_id: str) -> SearchHit:
    chunk = Chunk(
        source_id=source_id,
        source="s",
        collection="c",
        index=0,
        title="t",
        section="",
        page=None,
        text=source_id,
    )
    return SearchHit(chunk=chunk, score=0.0)


def test_rrf_rewards_agreement() -> None:
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("c"), hit("d"), hit("b")]
    fused = rrf_fuse([dense, sparse], k=60)

    order = [h.chunk.source_id for h in fused]
    assert order[:2] in (["b", "c"], ["c", "b"])  # in both lists beats first-in-one
    assert set(order) == {"a", "b", "c", "d"}
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_with_one_ranking_keeps_order() -> None:
    assert [h.chunk.source_id for h in rrf_fuse([[hit("x"), hit("y")]])] == ["x", "y"]


async def test_hybrid_search_finds_and_cites(indexed: DocStore, embeddings: HashEmbeddings) -> None:
    retriever = HybridRetriever(indexed, embeddings)
    hits = await retriever.search("how many weeks of parental leave", "docs", k=2)

    top = hits[0].chunk
    assert top.source_id == "handbook.md#0"
    assert top.section == "Parental leave"
    assert "16 weeks" in top.text
    assert embeddings.queries == ["how many weeks of parental leave"]


async def test_bm25_finds_exact_codes(indexed: DocStore, embeddings: HashEmbeddings) -> None:
    retriever = HybridRetriever(indexed, embeddings, mode="sparse")
    hits = await retriever.search("HR-204", "docs", k=1)
    assert hits[0].chunk.section == "Expenses"
    assert embeddings.queries == []  # sparse mode never embeds the query


async def test_reranker_reorders_candidates(indexed: DocStore, embeddings: HashEmbeddings) -> None:
    reranker = KeywordReranker("laptop")
    retriever = HybridRetriever(indexed, embeddings, reranker=reranker)
    hits = await retriever.search("parental leave", "docs", k=1)
    assert hits[0].chunk.section == "Security"
    assert reranker.calls[0][0] == "parental leave"


async def test_missing_collection_lists_available(indexed: DocStore) -> None:
    retriever = HybridRetriever(indexed, HashEmbeddings())
    with pytest.raises(StoreError, match=r"No collection 'specs'\. Available: docs"):
        await retriever.search("x", "specs")


async def test_stopword_query_still_uses_dense(indexed: DocStore) -> None:
    hits = await HybridRetriever(indexed, HashEmbeddings()).search("what is the", "docs")
    assert isinstance(hits, list)


async def test_reingest_replaces_old_chunks(
    tmp_path: Path, store: DocStore, embeddings: HashEmbeddings
) -> None:
    path = tmp_path / "handbook.md"
    path.write_text(HANDBOOK)
    await ingest(store, embeddings, path)
    path.write_text("# Handbook\n\n## Only\n\nOne section now.")
    await ingest(store, embeddings, path)

    hits = await HybridRetriever(store, embeddings).search("leave expenses security", "docs", k=10)
    assert [h.chunk.source_id for h in hits] == ["handbook.md#0"]
    assert hits[0].chunk.text == "One section now."


async def test_embedding_size_mismatch_is_refused(indexed: DocStore) -> None:
    with pytest.raises(StoreError, match="64-dim"):
        await indexed.ensure_collection("docs", dim=384)


@pytest.mark.parametrize("name", ["Docs", "my docs", "../x", ""])
def test_bad_collection_names(name: str) -> None:
    with pytest.raises(StoreError, match="Invalid collection name"):
        check_collection_name(name)
