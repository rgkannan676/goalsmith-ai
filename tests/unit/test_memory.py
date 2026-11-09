from pathlib import Path

from langchain_core.messages import ToolCall

from goalsmith.capabilities.tools.memory import recall_tool, remember_tool
from goalsmith.harness.tools import ToolContext, ToolRegistry
from goalsmith.memory.store import Memory, embedding_dims, in_memory_store
from tests.fixtures.fake_rag import HashEmbeddings


async def make_memory(user: str = "ana") -> tuple[Memory, Memory]:
    embeddings = HashEmbeddings()
    store = in_memory_store(embeddings, await embedding_dims(embeddings))
    return Memory(store, user), Memory(store, "ben")


async def test_recall_finds_memories_by_similarity() -> None:
    memory, _ = await make_memory()
    await memory.remember("Prefers green tea in the morning")
    await memory.remember("Works in the Berlin office")

    results = await memory.recall("which tea does she like", k=1)

    assert [r.text for r in results] == ["Prefers green tea in the morning"]
    assert results[0].score is not None
    assert results[0].created_at  # timestamps come from the Store


async def test_memories_are_per_user() -> None:
    ana, ben = await make_memory()
    await ana.remember("Prefers green tea")
    assert await ben.recall("tea") == []
    assert ana.namespace == ("users", "ana", "memories")


async def test_embedding_dims_probe() -> None:
    assert await embedding_dims(HashEmbeddings(dim=32)) == 32


async def test_memory_tools(tmp_path: Path) -> None:
    memory, _ = await make_memory()
    registry = ToolRegistry([remember_tool(memory), recall_tool(memory)])
    ctx = ToolContext(run_id="r", workspace=tmp_path)

    assert registry.get("remember").risk == "write"  # gated by approval
    assert registry.get("recall").risk == "read"

    saved = await registry.execute(
        ToolCall(name="remember", args={"text": "Likes short reports"}, id="1"), ctx
    )
    assert "Saved memory" in str(saved.content)

    found = await registry.execute(ToolCall(name="recall", args={"query": "reports"}, id="2"), ctx)
    assert "- Likes short reports (saved " in str(found.content)

    empty, _ = await make_memory("nobody")
    nothing = await recall_tool(empty).fn(recall_tool(empty).args_schema(query="x"), ctx)
    assert nothing == "No saved memories match."


async def test_remember_rejects_empty_text(tmp_path: Path) -> None:
    memory, _ = await make_memory()
    registry = ToolRegistry([remember_tool(memory)])
    msg = await registry.execute(
        ToolCall(name="remember", args={"text": ""}, id="1"),
        ToolContext(run_id="r", workspace=tmp_path),
    )
    assert msg.status == "error"
