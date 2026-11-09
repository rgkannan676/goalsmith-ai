"""remember and recall: the agent's long-term memory, as tools.

`remember` is a *write* tool, so it needs the user's approval by default. Memory
outlives the run, so a bad memory (for example one planted by a malicious web page
the agent read) would affect every later run. A human check stops that.
"""

from pydantic import BaseModel, Field

from goalsmith.harness.tools import Tool, ToolContext
from goalsmith.memory.store import Memory


class RememberArgs(BaseModel):
    text: str = Field(
        min_length=3,
        max_length=500,
        description="One self-contained fact or preference, e.g. 'Prefers British English'",
    )


class RecallArgs(BaseModel):
    query: str = Field(min_length=1, description="What to look for in saved memories")
    k: int = Field(default=5, ge=1, le=20, description="How many memories to return")


def remember_tool(memory: Memory) -> Tool:
    async def remember(args: RememberArgs, ctx: ToolContext) -> str:
        item = await memory.remember(args.text)
        return f"Saved memory {item.id[:8]}: {item.text}"

    return Tool(
        name="remember",
        description=(
            "Save a durable fact or preference about the user for future runs. "
            "Only for things the user would want kept; not for temporary task details."
        ),
        args_schema=RememberArgs,
        risk="write",
        fn=remember,
    )


def recall_tool(memory: Memory) -> Tool:
    async def recall(args: RecallArgs, ctx: ToolContext) -> str:
        items = await memory.recall(args.query, k=args.k)
        if not items:
            return "No saved memories match."
        return "\n".join(f"- {item.text} (saved {item.created_at[:10]})" for item in items)

    return Tool(
        name="recall",
        description="Search memories saved in earlier runs: the user's preferences and facts.",
        args_schema=RecallArgs,
        risk="read",
        fn=recall,
    )
