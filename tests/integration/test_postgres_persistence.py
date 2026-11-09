"""M4 acceptance tests against the real Postgres from docker-compose. Run `make up` first."""

import asyncio
import os
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from pydantic import BaseModel

from goalsmith.capabilities.tools.files import write_file_tool
from goalsmith.core.persistence import open_persistence
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.permissions import Decision
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from goalsmith.memory.store import Memory, MemoryStoreError
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call
from tests.fixtures.fake_rag import HashEmbeddings

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
async def run_id(test_dsn: str) -> AsyncIterator[str]:
    run_id = f"it-{uuid4().hex[:8]}"
    yield run_id
    async with open_persistence(test_dsn, HashEmbeddings(), require_durable=True) as saved:
        await saved.checkpointer.adelete_thread(run_id)


# The first half of the run happens in a separate Python process, which exits
# while the run is paused. Nothing survives except what is in Postgres.
PAUSE_SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from goalsmith.capabilities.tools.files import write_file_tool
    from goalsmith.core.persistence import open_persistence
    from goalsmith.harness.loop import AgentLoop
    from goalsmith.harness.tools import ToolContext, ToolRegistry
    from tests.fixtures.fake_llm import FakeChatModel, tool_call
    from tests.fixtures.fake_rag import HashEmbeddings

    async def main(dsn, run_id, workspace):
        model = FakeChatModel(responses=[
            tool_call("write_file", {"path": "plan.md", "content": "v1"}, call_id="w1")
        ])
        ctx = ToolContext(run_id=run_id, workspace=Path(workspace))
        async with open_persistence(dsn, HashEmbeddings(), require_durable=True) as saved:
            loop = AgentLoop(
                model, ToolRegistry([write_file_tool()]), ctx, checkpointer=saved.checkpointer
            )
            result = await loop.run("Write the plan")
        print(result.status)

    asyncio.run(main(*sys.argv[1:]))
    """
)


async def test_approval_survives_a_process_restart(
    tmp_path: Path, run_id: str, test_dsn: str
) -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        PAUSE_SCRIPT,
        test_dsn,
        run_id,
        str(tmp_path),
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(child.communicate(), timeout=60)
    assert child.returncode == 0, stderr.decode()
    assert stdout.decode().strip().endswith("awaiting_approval")
    assert not (tmp_path / "plan.md").exists()

    # A fresh loop in this process, with a new connection: only Postgres links them.
    ctx = ToolContext(run_id=run_id, workspace=tmp_path)
    async with open_persistence(test_dsn, HashEmbeddings(), require_durable=True) as saved:
        loop = AgentLoop(
            FakeChatModel(responses=[text("Plan written.")]),
            ToolRegistry([write_file_tool()]),
            ctx,
            checkpointer=saved.checkpointer,
        )
        assert [p.call_id for p in await loop.pending()] == ["w1"]
        result = await loop.resume({"w1": Decision(action="approve")})

    assert result.status == "completed"
    assert result.steps == 2
    assert (tmp_path / "plan.md").read_text() == "v1"


class _EmailArgs(BaseModel):
    to: str


async def test_a_rejected_action_never_runs(tmp_path: Path, run_id: str, test_dsn: str) -> None:
    sent: list[str] = []

    async def send(args: _EmailArgs, ctx: ToolContext) -> str:
        sent.append(args.to)
        return "sent"

    def registry() -> ToolRegistry:
        email = Tool(
            name="send_email", description="d", args_schema=_EmailArgs, risk="external", fn=send
        )
        return ToolRegistry([email])

    ctx = ToolContext(
        run_id=run_id, workspace=tmp_path, allowed_risks=frozenset({"read", "write", "external"})
    )
    call = tool_call("send_email", {"to": "everyone@example.com"}, call_id="e1")

    async with open_persistence(test_dsn, HashEmbeddings(), require_durable=True) as saved:
        first = AgentLoop(
            FakeChatModel(responses=[call]), registry(), ctx, checkpointer=saved.checkpointer
        )
        assert (await first.run("Email everyone")).status == "awaiting_approval"

    async with open_persistence(test_dsn, HashEmbeddings(), require_durable=True) as saved:
        model = FakeChatModel(responses=[text("OK, no email sent.")])
        second = AgentLoop(model, registry(), ctx, checkpointer=saved.checkpointer)
        result = await second.resume({"e1": Decision(action="reject", reason="too broad")})
        state = (await second.graph.aget_state({"configurable": {"thread_id": run_id}})).values

    assert result.status == "completed"
    assert sent == []  # the tool function was never called
    answers = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert len(answers) == 1
    assert answers[0].status == "error"
    assert "too broad" in str(answers[0].content)


async def test_memories_persist_across_connections(test_dsn: str) -> None:
    user = f"it-{uuid4().hex[:8]}"
    embeddings = HashEmbeddings()

    async with open_persistence(test_dsn, embeddings, require_durable=True) as saved:
        item = await Memory(saved.store, user).remember("Prefers green tea")

    async with open_persistence(test_dsn, embeddings, require_durable=True) as saved:
        memory = Memory(saved.store, user)
        found = await memory.recall("what tea do they like", k=1)
        await saved.store.adelete(memory.namespace, item.id)

    assert [m.text for m in found] == ["Prefers green tea"]


async def test_embedding_size_change_is_a_clear_error(test_dsn: str) -> None:
    async with open_persistence(test_dsn, HashEmbeddings(dim=64), require_durable=True):
        pass  # creates the memory table with 64-dim vectors (if not already there)
    with pytest.raises(MemoryStoreError, match="64-dim vectors but the embedding model makes 32"):
        async with open_persistence(test_dsn, HashEmbeddings(dim=32), require_durable=True):
            pass
