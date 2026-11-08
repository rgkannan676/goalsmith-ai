import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import ToolCall
from pydantic import BaseModel, Field

from goalsmith.harness.tools import Tool, ToolContext, ToolError, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)
    times: int = Field(default=1, ge=1, le=3)


async def echo(args: EchoArgs, ctx: ToolContext) -> str:
    return " ".join([args.text] * args.times)


async def fails_expectedly(args: EchoArgs, ctx: ToolContext) -> str:
    raise ToolError("file not found: notes.md")


async def crashes(args: EchoArgs, ctx: ToolContext) -> str:
    raise KeyError("boom")


def make_tool(name: str = "echo", fn=echo, risk="read") -> Tool:  # type: ignore[no-untyped-def]
    return Tool(name=name, description=f"{name} tool", args_schema=EchoArgs, risk=risk, fn=fn)


def call(name: str, args: dict, call_id: str = "c1") -> ToolCall:  # type: ignore[type-arg]
    return ToolCall(name=name, args=args, id=call_id, type="tool_call")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="r1", workspace=tmp_path)


# --- registry ---------------------------------------------------------------


def test_register_and_lookup() -> None:
    registry = ToolRegistry([make_tool("a"), make_tool("b")])
    assert registry.names() == ["a", "b"]
    assert registry.get("a").name == "a"


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        ToolRegistry([make_tool("a"), make_tool("a")])


def test_schemas_use_provider_neutral_function_format() -> None:
    [schema] = ToolRegistry([make_tool()]).schemas()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "echo"
    assert fn["description"] == "echo tool"
    assert set(fn["parameters"]["properties"]) == {"text", "times"}
    assert fn["parameters"]["required"] == ["text"]


# --- execute: errors become messages for the model, never crashes -----------


async def test_execute_valid_call(ctx: ToolContext) -> None:
    msg = await ToolRegistry([make_tool()]).execute(call("echo", {"text": "hi", "times": 2}), ctx)
    assert msg.content == "hi hi"
    assert msg.tool_call_id == "c1"
    assert msg.name == "echo"
    assert msg.status == "success"


async def test_unknown_tool_is_reported_to_model(ctx: ToolContext) -> None:
    msg = await ToolRegistry([make_tool()]).execute(call("nope", {}), ctx)
    assert msg.status == "error"
    assert "Unknown tool 'nope'" in str(msg.content)
    assert "echo" in str(msg.content)


async def test_invalid_args_are_reported_to_model(ctx: ToolContext) -> None:
    msg = await ToolRegistry([make_tool()]).execute(call("echo", {"text": "", "times": 9}), ctx)
    assert msg.status == "error"
    assert "Invalid arguments" in str(msg.content)
    assert "times" in str(msg.content)


async def test_expected_tool_error_is_reported(ctx: ToolContext) -> None:
    registry = ToolRegistry([make_tool(fn=fails_expectedly)])
    msg = await registry.execute(call("echo", {"text": "x"}), ctx)
    assert msg.status == "error"
    assert "file not found: notes.md" in str(msg.content)


async def test_unexpected_exception_is_reported_not_raised(ctx: ToolContext) -> None:
    registry = ToolRegistry([make_tool(fn=crashes)])
    msg = await registry.execute(call("echo", {"text": "x"}), ctx)
    assert msg.status == "error"
    assert "KeyError" in str(msg.content)


async def test_risk_not_allowed_by_policy_is_blocked(tmp_path: Path) -> None:
    ctx = ToolContext(run_id="r1", workspace=tmp_path, allowed_risks=frozenset({"read"}))
    msg = await ToolRegistry([make_tool(risk="external")]).execute(call("echo", {"text": "x"}), ctx)
    assert msg.status == "error"
    assert "not allowed" in str(msg.content)


async def test_long_output_is_truncated(ctx: ToolContext) -> None:
    registry = ToolRegistry([make_tool()], max_output_chars=10)
    msg = await registry.execute(call("echo", {"text": "abcdefghijklmnop"}), ctx)
    assert str(msg.content).startswith("abcdefghij")
    assert "truncated" in str(msg.content)


async def test_execute_all_runs_concurrently_and_keeps_order(ctx: ToolContext) -> None:
    started: list[str] = []

    async def slow(args: EchoArgs, ctx: ToolContext) -> str:
        started.append(args.text)
        await asyncio.sleep(0.05)
        return args.text

    registry = ToolRegistry([make_tool(fn=slow)])
    calls = [call("echo", {"text": t}, call_id=t) for t in ["a", "b", "c"]]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    msgs = await registry.execute_all(calls, ctx)
    elapsed = loop.time() - t0
    assert [m.content for m in msgs] == ["a", "b", "c"]
    assert elapsed < 0.12  # three 0.05 s calls in parallel, not 0.15 s in sequence
