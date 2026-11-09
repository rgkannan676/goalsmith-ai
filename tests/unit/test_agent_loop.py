from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.harness.budget import Budget
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="r1", workspace=tmp_path)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([read_file_tool(), write_file_tool()])


async def test_goal_completes_after_tool_use(registry: ToolRegistry, ctx: ToolContext) -> None:
    model = FakeChatModel(
        responses=[
            tool_call("write_file", {"path": "report.md", "content": "# Report"}),
            text("Done. The report is in report.md."),
        ]
    )
    result = await AgentLoop(model, registry, ctx).run("Write a short report")

    assert result.status == "completed"
    assert result.answer == "Done. The report is in report.md."
    assert result.steps == 2
    assert (ctx.workspace / "report.md").read_text() == "# Report"


async def test_loop_stops_at_step_budget(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Acceptance test for M2: a model that never stops is cut off cleanly."""
    endless = [tool_call("read_file", {"path": "x.md"}, call_id=f"c{i}") for i in range(20)]
    model = FakeChatModel(responses=endless)

    result = await AgentLoop(model, registry, ctx, budget=Budget(max_steps=3)).run("Loop forever")

    assert result.status == "budget_exceeded"
    assert result.steps == 3
    assert len(model.calls) == 3  # the model was not called a 4th time
    assert "step budget" in result.answer


async def test_budget_stop_keeps_partial_answer(registry: ToolRegistry, ctx: ToolContext) -> None:
    partial = AIMessage(
        content="Found two sources so far.",
        tool_calls=[{"name": "read_file", "args": {"path": "a"}, "id": "c1", "type": "tool_call"}],
    )
    model = FakeChatModel(responses=[partial])
    result = await AgentLoop(model, registry, ctx, budget=Budget(max_steps=1)).run("Research")
    assert result.status == "budget_exceeded"
    assert "Found two sources so far." in result.answer


async def test_tool_errors_go_back_to_the_model(registry: ToolRegistry, ctx: ToolContext) -> None:
    model = FakeChatModel(responses=[tool_call("delete_everything", {}), text("Sorry, cannot.")])
    result = await AgentLoop(model, registry, ctx).run("Do something odd")

    assert result.status == "completed"
    second_call = model.calls[1]
    tool_msg = second_call[-1]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "Unknown tool" in str(tool_msg.content)


async def test_parallel_tool_calls_in_one_turn(registry: ToolRegistry, ctx: ToolContext) -> None:
    both = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"path": "a.md", "content": "A"}, "id": "1"},
            {"name": "write_file", "args": {"path": "b.md", "content": "B"}, "id": "2"},
        ],
    )
    model = FakeChatModel(responses=[both, text("Both written.")])
    result = await AgentLoop(model, registry, ctx).run("Write two files")

    assert result.status == "completed"
    assert (ctx.workspace / "a.md").read_text() == "A"
    assert (ctx.workspace / "b.md").read_text() == "B"
    tool_msgs = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_msgs] == ["1", "2"]


async def test_model_sees_system_prompt_goal_and_tools(
    registry: ToolRegistry, ctx: ToolContext
) -> None:
    model = FakeChatModel(responses=[text("ok")])
    await AgentLoop(model, registry, ctx, system_prompt="You are a test agent.").run("Hello")

    first = model.calls[0]
    assert isinstance(first[0], SystemMessage)
    assert first[0].content == "You are a test agent."
    assert first[1].content == "Hello"
    assert [t["function"]["name"] for t in model.bound_tools] == ["read_file", "write_file"]


async def test_progress_callback_sees_each_step(registry: ToolRegistry, ctx: ToolContext) -> None:
    seen: list[str] = []
    model = FakeChatModel(responses=[tool_call("read_file", {"path": "x"}), text("done")])
    await AgentLoop(model, registry, ctx).run("Go", on_update=lambda node, _: seen.append(node))
    assert seen == ["agent", "tools", "agent"]


class _Args(BaseModel):
    pass


async def test_tool_risk_policy_applies_inside_loop(tmp_path: Path) -> None:
    async def send(args: _Args, ctx: ToolContext) -> str:
        return "sent"

    registry = ToolRegistry(
        [Tool(name="send_email", description="d", args_schema=_Args, risk="external", fn=send)]
    )
    ctx = ToolContext(run_id="r1", workspace=tmp_path)  # default policy: read + write only
    model = FakeChatModel(responses=[tool_call("send_email", {}), text("blocked")])
    await AgentLoop(model, registry, ctx).run("Email someone")
    tool_msg = model.calls[1][-1]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "not allowed" in str(tool_msg.content)
