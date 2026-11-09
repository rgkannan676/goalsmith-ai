"""Tool registry and dispatch: the single path every tool call goes through.

A model asks for a tool by name with JSON arguments. The registry:
1. finds the tool (unknown names become an error message),
2. checks its risk level against the run's policy,
3. validates the arguments with the tool's Pydantic schema,
4. runs it, and
5. turns the result, or any error, into a ToolMessage for the model.

A tool error is information for the model, not a crash for the user.

Each call is traced as a "tool" span (arguments, result, error status) and logged.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import ToolCall, ToolMessage
from pydantic import BaseModel, ValidationError

from goalsmith.observability.logs import get_logger
from goalsmith.observability.spans import mark_error, observation, set_output

log = get_logger(__name__)

Risk = Literal["read", "write", "external"]
"""read: no side effects. write: changes the run's own workspace.
external: affects the outside world (GitHub, email, ...)."""

ToolFn = Callable[[Any, "ToolContext"], Awaitable[str]]


class ToolError(Exception):
    """An expected tool failure. The message is shown to the model."""


@dataclass(frozen=True)
class ToolContext:
    """Per-run information every tool receives."""

    run_id: str
    workspace: Path
    allowed_risks: frozenset[Risk] = field(default_factory=lambda: frozenset({"read", "write"}))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_schema: type[BaseModel]
    risk: Risk
    fn: ToolFn

    def schema(self) -> dict[str, Any]:
        """The tool as the model sees it, in the provider-neutral "function" format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), *, max_output_chars: int = 20_000) -> None:
        self._tools: dict[str, Tool] = {}
        self._max_output_chars = max_output_chars
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute_all(self, calls: Sequence[ToolCall], ctx: ToolContext) -> list[ToolMessage]:
        """Run several tool calls concurrently. Results keep the order of `calls`."""
        return list(await asyncio.gather(*(self.execute(call, ctx) for call in calls)))

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolMessage:
        name = call["name"]
        started = time.perf_counter()
        with observation(
            f"tool {name}", "tool", input=call["args"], metadata={"call_id": call.get("id")}
        ) as span:
            message = await self._execute(call, ctx)
            content = str(message.content)
            set_output(span, content)
            if message.status == "error":
                mark_error(span, content)
        log.info(
            "tool.finished",
            tool=name,
            call_id=call.get("id"),
            status=message.status,
            ms=round((time.perf_counter() - started) * 1000),
        )
        return message

    async def _execute(self, call: ToolCall, ctx: ToolContext) -> ToolMessage:
        name = call["name"]
        call_id = call.get("id") or ""

        def reply(content: str, status: Literal["success", "error"] = "success") -> ToolMessage:
            return ToolMessage(content=content, tool_call_id=call_id, name=name, status=status)

        tool = self._tools.get(name)
        if tool is None:
            return reply(
                f"Error: Unknown tool {name!r}. Available tools: {', '.join(self.names())}",
                "error",
            )

        if tool.risk not in ctx.allowed_risks:
            return reply(
                f"Error: tool {name!r} has risk level {tool.risk!r}, "
                "which is not allowed in this run.",
                "error",
            )

        try:
            args = tool.args_schema.model_validate(call["args"])
        except ValidationError as exc:
            return reply(f"Error: Invalid arguments for {name!r}: {_describe(exc)}", "error")

        try:
            output = await tool.fn(args, ctx)
        except ToolError as exc:
            return reply(f"Error: {exc}", "error")
        except Exception as exc:  # any bug in a tool must not crash the run
            log.exception("tool.crashed", tool=name, call_id=call_id)
            return reply(
                f"Error: {name!r} failed unexpectedly ({type(exc).__name__}: {exc})", "error"
            )

        return reply(self._truncate(output))

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        dropped = len(text) - self._max_output_chars
        return f"{text[: self._max_output_chars]}\n...[truncated {dropped} characters]"


def _describe(exc: ValidationError) -> str:
    """Short, model-readable validation errors, e.g. `times: Input should be <= 3`."""
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'args'}: {err['msg']}"
        for err in exc.errors()
    )
