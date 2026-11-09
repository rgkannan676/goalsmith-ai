"""read_file and write_file, limited to the run's own workspace folder."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from goalsmith.harness.tools import Tool, ToolContext, ToolError

_PATH_HELP = "Path relative to the run workspace, e.g. 'notes/summary.md'"


class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1, description=_PATH_HELP)


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1, description=_PATH_HELP)
    content: str = Field(description="The full text to write. Replaces the file if it exists.")


def resolve_in_workspace(ctx: ToolContext, path: str) -> Path:
    """Turn a model-supplied path into a real path, refusing anything outside the workspace.

    `resolve()` follows `..` and symlinks first, so tricks like `../x`, `/etc/passwd`
    or a symlink pointing outside are all caught by the same check.
    """
    root = ctx.workspace.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ToolError(
            f"Path {path!r} is outside the workspace. Use a relative path like 'notes/summary.md'."
        )
    return target


async def _read_file(args: ReadFileArgs, ctx: ToolContext) -> str:
    target = resolve_in_workspace(ctx, args.path)
    try:
        # File I/O blocks, so run it in a thread to keep the event loop free.
        return await asyncio.to_thread(target.read_text, encoding="utf-8")
    except FileNotFoundError:
        raise ToolError(f"File not found: {args.path}") from None
    except IsADirectoryError:
        raise ToolError(f"{args.path} is a directory, not a file") from None
    except UnicodeDecodeError:
        raise ToolError(f"{args.path} is not a UTF-8 text file") from None


async def _write_file(args: WriteFileArgs, ctx: ToolContext) -> str:
    target = resolve_in_workspace(ctx, args.path)

    def write() -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args.content, encoding="utf-8")

    await asyncio.to_thread(write)
    return f"Wrote {len(args.content)} characters to {args.path}"


def read_file_tool() -> Tool:
    return Tool(
        name="read_file",
        description="Read a text file from the run workspace.",
        args_schema=ReadFileArgs,
        risk="read",
        fn=_read_file,
    )


def write_file_tool() -> Tool:
    return Tool(
        name="write_file",
        description="Create or overwrite a text file in the run workspace.",
        args_schema=WriteFileArgs,
        risk="write",
        fn=_write_file,
    )
