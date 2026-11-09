from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langchain_core.messages import ToolCall, ToolMessage
from qdrant_client import AsyncQdrantClient

from goalsmith.capabilities.rag.ingest import ingest_files
from goalsmith.capabilities.rag.retrieve import HybridRetriever
from goalsmith.capabilities.rag.store import DocStore
from goalsmith.capabilities.tools.docs import search_docs_tool
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.tools import ToolContext, ToolRegistry
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call
from tests.fixtures.fake_rag import HashEmbeddings

POLICY = "# Travel\n\n## Flights\n\nEconomy class for flights under 6 hours.\n"


@pytest.fixture
async def retriever(tmp_path: Path) -> AsyncIterator[HybridRetriever]:
    client = AsyncQdrantClient(location=":memory:")
    store = DocStore(client)
    (tmp_path / "travel.md").write_text(POLICY)
    embeddings = HashEmbeddings()
    await ingest_files(
        [tmp_path / "travel.md"],
        collection="docs",
        store=store,
        embeddings=embeddings,
        max_chars=500,
        overlap=0,
    )
    yield HybridRetriever(store, embeddings)
    await client.close()


async def call(retriever: HybridRetriever, tmp_path: Path, **args: object) -> ToolMessage:
    registry = ToolRegistry([search_docs_tool(retriever)])
    ctx = ToolContext(run_id="r", workspace=tmp_path)
    return await registry.execute(ToolCall(name="search_docs", args=args, id="c1"), ctx)


async def test_results_start_with_source_ids(retriever: HybridRetriever, tmp_path: Path) -> None:
    msg = await call(retriever, tmp_path, query="flight class")
    assert msg.status == "success"
    assert str(msg.content).startswith("[travel.md#0] Travel > Flights\nEconomy class")


async def test_unknown_collection_is_a_tool_error(
    retriever: HybridRetriever, tmp_path: Path
) -> None:
    msg = await call(retriever, tmp_path, query="x", collection="specs")
    assert msg.status == "error"
    assert "Available: docs" in str(msg.content)


def test_tool_is_read_risk_and_lists_collections(retriever: HybridRetriever) -> None:
    tool = search_docs_tool(retriever, default_collection="docs", collections=["docs", "specs"])
    assert tool.risk == "read"
    assert "Collections: docs, specs." in tool.description
    assert tool.args_schema.model_validate({"query": "q"}).model_dump()["collection"] == "docs"


async def test_answer_cites_the_source_id(retriever: HybridRetriever, tmp_path: Path) -> None:
    """M3 acceptance: the model sees source ids and the answer carries them through."""
    model = FakeChatModel(
        responses=[
            tool_call("search_docs", {"query": "flight class"}),
            text("Book economy for flights under 6 hours [travel.md#0]."),
        ]
    )
    loop = AgentLoop(
        model,
        ToolRegistry([search_docs_tool(retriever)]),
        ToolContext(run_id="r", workspace=tmp_path),
    )
    result = await loop.run("Which class should I fly?")

    tool_result = model.calls[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert "[travel.md#0]" in str(tool_result.content)
    assert "[travel.md#0]" in result.answer
    assert "source id" in str(model.calls[0][0].content)  # system prompt asks for citations
