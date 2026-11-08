from pathlib import Path

import pytest
from langchain_core.messages import ToolCall

from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.harness.tools import ToolContext, ToolRegistry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "workspaces" / "r1"
    workspace.mkdir(parents=True)
    return ToolContext(run_id="r1", workspace=workspace)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([read_file_tool(), write_file_tool()])


async def run(
    registry: ToolRegistry, ctx: ToolContext, name: str, **args: object
) -> tuple[str, str]:
    msg = await registry.execute(ToolCall(name=name, args=args, id="c", type="tool_call"), ctx)
    return str(msg.status), str(msg.content)


def test_risk_levels() -> None:
    assert read_file_tool().risk == "read"
    assert write_file_tool().risk == "write"


async def test_write_then_read(registry: ToolRegistry, ctx: ToolContext) -> None:
    status, out = await run(registry, ctx, "write_file", path="notes/summary.md", content="# Hi")
    assert status == "success"
    assert "notes/summary.md" in out
    assert (ctx.workspace / "notes" / "summary.md").read_text() == "# Hi"

    status, out = await run(registry, ctx, "read_file", path="notes/summary.md")
    assert (status, out) == ("success", "# Hi")


@pytest.mark.parametrize("bad_path", ["../escape.txt", "a/../../escape.txt", "/etc/passwd"])
async def test_paths_outside_workspace_are_rejected(
    registry: ToolRegistry, ctx: ToolContext, bad_path: str
) -> None:
    status, out = await run(registry, ctx, "write_file", path=bad_path, content="x")
    assert status == "error"
    assert "outside the workspace" in out
    assert not (ctx.workspace.parent / "escape.txt").exists()


async def test_symlink_escape_is_rejected(
    registry: ToolRegistry, ctx: ToolContext, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (ctx.workspace / "link.txt").symlink_to(secret)
    status, out = await run(registry, ctx, "read_file", path="link.txt")
    assert status == "error"
    assert "top secret" not in out


async def test_read_missing_file(registry: ToolRegistry, ctx: ToolContext) -> None:
    status, out = await run(registry, ctx, "read_file", path="missing.md")
    assert status == "error"
    assert "not found" in out
