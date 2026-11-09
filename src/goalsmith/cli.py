"""Command line: `goalsmith run "<goal>"`."""

import asyncio
from datetime import datetime
from typing import Annotated, Any
from uuid import uuid4

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel

from goalsmith import __version__
from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.capabilities.tools.web import tavily_client, web_search_tool
from goalsmith.config import Settings, get_settings
from goalsmith.harness.budget import Budget
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from goalsmith.llm import LLMConfigError, get_chat_model

app = typer.Typer(help="Goalsmith: from goal to done.", no_args_is_help=True)
console = Console()

EXIT_BUDGET_EXCEEDED = 2


def _show_version(value: bool) -> None:
    if value:
        console.print(f"goalsmith {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Goalsmith: from goal to done."""


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="What you want done, in plain English.")],
    max_steps: Annotated[
        int | None, typer.Option(min=1, help="Maximum model calls (default from settings).")
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Name for this run's workspace.")] = None,
) -> None:
    """Run a single agent on GOAL and print the result."""
    settings = get_settings()
    try:
        model = get_chat_model("worker", settings)
    except LLMConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    run_id = run_id or _new_run_id()
    workspace = settings.workspace_root / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    context = ToolContext(run_id=run_id, workspace=workspace)
    budget = Budget(max_steps=max_steps or settings.max_steps)

    loop = AgentLoop(model, ToolRegistry(_build_tools(settings)), context, budget=budget)
    console.print(f"[bold]Run[/bold] {run_id}  ·  budget {budget.max_steps} steps")
    result = asyncio.run(loop.run(goal, on_update=_print_update))

    console.print(Panel(Markdown(result.answer), title=f"{result.status} · {result.steps} steps"))
    console.print(f"Workspace: {workspace}")
    if result.status == "budget_exceeded":
        raise typer.Exit(EXIT_BUDGET_EXCEEDED)


def _build_tools(settings: Settings) -> list[Tool]:
    tools = [read_file_tool(), write_file_tool()]
    if settings.tavily_api_key is None:
        console.print("[yellow]web_search is off: TAVILY_API_KEY is not set.[/yellow]")
    else:
        tools.append(web_search_tool(tavily_client(settings.tavily_api_key.get_secret_value())))
    return tools


def _print_update(node: str, update: dict[str, Any]) -> None:
    """Live progress: one line per tool call and per tool result."""
    for message in update.get("messages", []):
        if node == "agent":
            for call in getattr(message, "tool_calls", []):
                args = ", ".join(f"{k}={_short(v)}" for k, v in call["args"].items())
                console.print(f"  [cyan]→ {call['name']}[/cyan]({escape(args)})")
        elif node == "tools":
            ok = getattr(message, "status", "success") == "success"
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {mark} {message.name}: {escape(_short(message.content, 80))}")


def _short(value: object, limit: int = 40) -> str:
    text = repr(value) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
