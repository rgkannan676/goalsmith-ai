"""Against the real Qdrant from docker-compose. Run `make up` first."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest

from goalsmith.capabilities.rag.ingest import ingest_files
from goalsmith.capabilities.rag.retrieve import HybridRetriever
from goalsmith.capabilities.rag.store import DocStore, open_client
from goalsmith.config import RagSettings
from tests.fixtures.fake_rag import HashEmbeddings

pytestmark = pytest.mark.integration


@pytest.fixture
async def store_and_collection() -> AsyncIterator[tuple[DocStore, str]]:
    client = open_client(RagSettings())
    name = f"test-{uuid4().hex[:8]}"
    yield DocStore(client), name
    await client.delete_collection(name)
    await client.close()


async def test_ingest_and_hybrid_search(
    tmp_path: Path, store_and_collection: tuple[DocStore, str]
) -> None:
    store, collection = store_and_collection
    (tmp_path / "handbook.md").write_text(
        "# Handbook\n\n## Leave\n\n16 weeks of parental leave.\n\n"
        "## Expenses\n\nUse form HR-204 within 30 days.\n"
    )
    embeddings = HashEmbeddings()
    await ingest_files(
        [tmp_path / "handbook.md"],
        collection=collection,
        store=store,
        embeddings=embeddings,
        max_chars=500,
        overlap=0,
    )

    hits = await HybridRetriever(store, embeddings).search("HR-204 form", collection, k=1)

    assert hits[0].chunk.source_id == "handbook.md#1"
    assert collection in await store.collections()
