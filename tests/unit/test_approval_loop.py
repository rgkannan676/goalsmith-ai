"""Human approval inside the agent loop: pause, then approve, edit or reject."""

import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.core.serde import checkpoint_serializer
from goalsmith.harness.budget import Budget, BudgetUsage
from goalsmith.harness.loop import AgentLoop, RunStateError
from goalsmith.harness.permissions import Decision
from goalsmith.harness.tools import ToolContext, ToolRegistry
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call

WRITE = tool_call("write_file", {"path": "plan.md", "content": "v1"}, call_id="w1")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="r1", workspace=tmp_path)


def make_loop(
    model: FakeChatModel, ctx: ToolContext, saver: InMemorySaver | None = None
) -> AgentLoop:
    registry = ToolRegistry([read_file_tool(), write_file_tool()])
    return AgentLoop(model, registry, ctx, checkpointer=saver or InMemorySaver())


async def test_write_pauses_before_running(ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[WRITE, text("done")])
    result = await make_loop(model, ctx).run("Write a plan")

    assert result.status == "awaiting_approval"
    assert [(p.call_id, p.tool, p.args["path"]) for p in result.pending] == [
        ("w1", "write_file", "plan.md")
    ]
    assert not (ctx.workspace / "plan.md").exists()  # nothing ran yet
    assert len(model.calls) == 1


async def test_approve_runs_the_tool(ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[WRITE, text("Plan written.")])
    loop = make_loop(model, ctx)
    await loop.run("Write a plan")

    result = await loop.resume({"w1": Decision(action="approve")})

    assert result.status == "completed"
    assert result.answer == "Plan written."
    assert (ctx.workspace / "plan.md").read_text() == "v1"
    assert result.pending == []


async def test_edit_runs_with_the_new_arguments(ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[WRITE, text("ok")])
    loop = make_loop(model, ctx)
    await loop.run("Write a plan")

    await loop.resume(
        {"w1": Decision(action="edit", args={"path": "plan.md", "content": "edited"})}
    )

    assert (ctx.workspace / "plan.md").read_text() == "edited"
    # The model's history shows the edited call, so it knows what actually ran.
    ai = next(m for m in model.calls[1] if isinstance(m, AIMessage) and m.tool_calls)
    assert ai.tool_calls[0]["args"]["content"] == "edited"


async def test_reject_never_runs_and_tells_the_model(ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[WRITE, text("Understood, not writing it.")])
    loop = make_loop(model, ctx)
    await loop.run("Write a plan")

    result = await loop.resume({"w1": Decision(action="reject", reason="not now")})

    assert result.status == "completed"
    assert not (ctx.workspace / "plan.md").exists()
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1  # the rejection answers the call; the tool did not run
    assert tool_msgs[0].status == "error"
    assert "not now" in str(tool_msgs[0].content)


async def test_read_tools_never_pause(ctx: ToolContext) -> None:
    (ctx.workspace / "a.md").write_text("A")
    model = FakeChatModel(responses=[tool_call("read_file", {"path": "a.md"}), text("A it is")])
    result = await make_loop(model, ctx).run("Read a")
    assert result.status == "completed"


async def test_mixed_turn_runs_reads_after_the_decision(ctx: ToolContext) -> None:
    (ctx.workspace / "a.md").write_text("A")
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": "a.md"}, "id": "r", "type": "tool_call"},
            {"name": "write_file", "args": {"path": "b.md", "content": "B"}, "id": "w"},
        ],
    )
    model = FakeChatModel(responses=[both, text("ok")])
    loop = make_loop(model, ctx)
    first = await loop.run("Copy")
    assert [p.call_id for p in first.pending] == ["w"]  # only the write is asked about

    await loop.resume({"w": Decision(action="reject")})

    tool_msgs = {m.tool_call_id: m for m in model.calls[1] if isinstance(m, ToolMessage)}
    assert tool_msgs["r"].content == "A"  # the read still ran
    assert tool_msgs["w"].status == "error"
    assert not (ctx.workspace / "b.md").exists()


async def test_resume_from_a_new_loop_instance(ctx: ToolContext) -> None:
    """The paused state lives in the checkpointer, not in the AgentLoop object.
    (The Postgres version of this, across connections, is an integration test.)"""
    saver = InMemorySaver()
    await make_loop(FakeChatModel(responses=[WRITE]), ctx, saver).run("Write a plan")

    fresh = make_loop(FakeChatModel(responses=[text("Written.")]), ctx, saver)
    assert [p.call_id for p in await fresh.pending()] == ["w1"]
    result = await fresh.resume({"w1": Decision(action="approve")})

    assert result.status == "completed"
    assert result.steps == 2  # the budget usage was saved with the run
    assert (ctx.workspace / "plan.md").read_text() == "v1"


async def test_budget_still_applies_after_resume(ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[WRITE, WRITE, WRITE])
    registry = ToolRegistry([write_file_tool()])
    loop = AgentLoop(model, registry, ctx, budget=Budget(max_steps=1))
    await loop.run("Loop")
    result = await loop.resume({"w1": Decision(action="approve")})
    assert result.status == "budget_exceeded"
    assert len(model.calls) == 1


async def test_resume_errors(ctx: ToolContext) -> None:
    loop = make_loop(FakeChatModel(responses=[text("hi"), WRITE]), ctx)
    with pytest.raises(RunStateError, match="not waiting for approval"):
        await loop.resume({})

    await loop.run("Hi")
    with pytest.raises(RunStateError, match="already exists"):
        await loop.run("Hi again")


async def test_every_pending_call_needs_a_decision(ctx: ToolContext) -> None:
    loop = make_loop(FakeChatModel(responses=[WRITE]), ctx)
    await loop.run("Write")
    with pytest.raises(ValueError, match="No decision for tool call"):
        await loop.resume({})


async def test_checkpoints_restore_budget_without_unsafe_type_warnings(
    ctx: ToolContext, caplog: pytest.LogCaptureFixture
) -> None:
    """Custom state types must be on the checkpoint allowlist (core/serde.py);
    LangGraph will refuse unlisted ones in a future version."""
    saver = InMemorySaver(serde=checkpoint_serializer())
    await make_loop(FakeChatModel(responses=[WRITE]), ctx, saver).run("Write")
    fresh = make_loop(FakeChatModel(responses=[text("ok")]), ctx, saver)
    with caplog.at_level(logging.WARNING):
        result = await fresh.resume({"w1": Decision(action="approve")})
    assert result.steps == 2
    assert "unregistered type" not in caplog.text


def test_serializer_round_trips_budget_usage() -> None:
    serde = checkpoint_serializer()
    assert serde.loads_typed(serde.dumps_typed(BudgetUsage(steps=3))) == BudgetUsage(steps=3)
