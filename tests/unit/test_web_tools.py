from pathlib import Path
from typing import Any

from langchain_core.messages import ToolCall

from goalsmith.capabilities.tools.web import web_search_tool
from goalsmith.harness.tools import ToolContext, ToolRegistry


class FakeSearchClient:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, *, max_results: int) -> dict[str, Any]:
        self.queries.append((query, max_results))
        return {"results": self.results[:max_results]}


async def search(client: FakeSearchClient, tmp_path: Path, **args: object) -> tuple[str, str]:
    registry = ToolRegistry([web_search_tool(client)])
    ctx = ToolContext(run_id="r1", workspace=tmp_path)
    msg = await registry.execute(
        ToolCall(name="web_search", args=args, id="c", type="tool_call"), ctx
    )
    return str(msg.status), str(msg.content)


def test_web_search_is_read_risk() -> None:
    assert web_search_tool(FakeSearchClient([])).risk == "read"


async def test_results_are_numbered_with_urls(tmp_path: Path) -> None:
    client = FakeSearchClient(
        [
            {"title": "Qdrant docs", "url": "https://qdrant.tech", "content": "Vector DB."},
            {"title": "pgvector", "url": "https://github.com/pgvector", "content": "Postgres ext."},
        ]
    )
    status, out = await search(client, tmp_path, query="vector databases", max_results=2)
    assert status == "success"
    assert client.queries == [("vector databases", 2)]
    assert "[1] Qdrant docs" in out
    assert "URL: https://qdrant.tech" in out
    assert "[2] pgvector" in out


async def test_long_snippets_are_shortened(tmp_path: Path) -> None:
    client = FakeSearchClient([{"title": "t", "url": "u", "content": "x" * 5000}])
    _, out = await search(client, tmp_path, query="q")
    assert len(out) < 1500


async def test_no_results(tmp_path: Path) -> None:
    status, out = await search(FakeSearchClient([]), tmp_path, query="zzz")
    assert status == "success"
    assert "No results" in out


async def test_max_results_is_bounded(tmp_path: Path) -> None:
    status, out = await search(FakeSearchClient([]), tmp_path, query="q", max_results=50)
    assert status == "error"
    assert "max_results" in out
