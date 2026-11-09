"""Command line: `goalsmith run "<goal>"`, `goalsmith ingest <files>`, `goalsmith search "<q>"`."""

import asyncio
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from langchain_core.language_models import BaseChatModel
from qdrant_client.http.exceptions import ApiException
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from goalsmith import __version__
from goalsmith.capabilities.rag.ingest import IngestError, IngestResult, ingest_files
from goalsmith.capabilities.rag.models import SearchHit
from goalsmith.capabilities.rag.retrieve import HybridRetriever
from goalsmith.capabilities.rag.store import DocStore, StoreError, open_client
from goalsmith.capabilities.tools.docs import search_docs_tool
from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.capabilities.tools.web import tavily_client, web_search_tool
from goalsmith.config import Settings, get_settings
from goalsmith.core.state import RunResult
from goalsmith.harness.budget import Budget
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from goalsmith.llm import LLMConfigError, get_chat_model, get_embeddings, get_reranker

app = typer.Typer(help="Goalsmith: from goal to done.", no_args_is_help=True)
console = Console()

EXIT_BUDGET_EXCEEDED = 2

CollectionOption = Annotated[
    str | None, typer.Option("--collection", "-c", help="Document collection (default: docs).")
]


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


# --- run -------------------------------------------------------------------------


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

    console.print(f"[bold]Run[/bold] {run_id}  ·  budget {budget.max_steps} steps")
    result = asyncio.run(_run(goal, model, context, budget, settings))

    console.print(Panel(Markdown(result.answer), title=f"{result.status} · {result.steps} steps"))
    console.print(f"Workspace: {workspace}")
    if result.status == "budget_exceeded":
        raise typer.Exit(EXIT_BUDGET_EXCEEDED)


async def _run(
    goal: str, model: BaseChatModel, context: ToolContext, budget: Budget, settings: Settings
) -> RunResult:
    # One event loop for the whole run: the Qdrant client must live on the same loop
    # as the agent that calls search_docs.
    async with _open_store(settings) as store:
        tools = await _build_tools(settings, store)
        loop = AgentLoop(model, ToolRegistry(tools), context, budget=budget)
        return await loop.run(goal, on_update=_print_update)


async def _build_tools(settings: Settings, store: DocStore) -> list[Tool]:
    tools = [read_file_tool(), write_file_tool()]
    if settings.tavily_api_key is None:
        _warn("web_search is off: TAVILY_API_KEY is not set.")
    else:
        tools.append(web_search_tool(tavily_client(settings.tavily_api_key.get_secret_value())))

    docs_tool = await _docs_tool(settings, store)
    if docs_tool is not None:
        tools.append(docs_tool)
    return tools


async def _docs_tool(settings: Settings, store: DocStore) -> Tool | None:
    try:
        collections = await store.collections()
    except ApiException:
        _warn(f"search_docs is off: Qdrant is not reachable at {settings.rag.qdrant_url}.")
        return None
    if not collections:
        _warn("search_docs is off: no documents yet. Add some with `goalsmith ingest <files>`.")
        return None
    return search_docs_tool(
        _retriever(settings, store),
        default_collection=settings.rag.default_collection,
        collections=collections,
    )


# --- documents -------------------------------------------------------------------


@app.command()
def ingest(
    paths: Annotated[list[Path], typer.Argument(help="Files or folders (.md .html .pdf .txt).")],
    collection: CollectionOption = None,
) -> None:
    """Add documents to a collection so search_docs can find them."""
    settings = get_settings()
    name = collection or settings.rag.default_collection
    results = _run_docs_command(_ingest(paths, name, settings), settings)

    table = Table("File", "Chunks", title=f"Ingested into {name!r}")
    for result in results:
        table.add_row(escape(result.source), str(result.chunks))
    console.print(table)


async def _ingest(paths: list[Path], collection: str, settings: Settings) -> list[IngestResult]:
    async with _open_store(settings) as store:
        return await ingest_files(
            paths,
            collection=collection,
            store=store,
            embeddings=get_embeddings(settings),
            max_chars=settings.rag.chunk_chars,
            overlap=settings.rag.chunk_overlap,
        )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    collection: CollectionOption = None,
    k: Annotated[int, typer.Option("-k", min=1, max=20, help="Number of results.")] = 5,
) -> None:
    """Search documents directly, without the agent. Handy for checking retrieval."""
    settings = get_settings()
    name = collection or settings.rag.default_collection
    hits = _run_docs_command(_search(query, name, k, settings), settings)
    if not hits:
        console.print("No results.")
    for hit in hits:
        console.print(
            f"[cyan]{escape(f'[{hit.chunk.source_id}]')}[/cyan] {escape(hit.chunk.location())}"
            f"  [dim]score {hit.score:.3f}[/dim]"
        )
        console.print(f"  {escape(_short(hit.chunk.text, 200))}")


async def _search(query: str, collection: str, k: int, settings: Settings) -> list[SearchHit]:
    async with _open_store(settings) as store:
        return await _retriever(settings, store).search(query, collection, k=k)


def _run_docs_command[T](coro: Coroutine[Any, Any, T], settings: Settings) -> T:
    """Run an ingest/search coroutine and turn expected failures into a clean exit."""
    try:
        return asyncio.run(coro)
    except (IngestError, StoreError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except ApiException as exc:
        console.print(
            f"[red]Qdrant is not reachable at {settings.rag.qdrant_url}.[/red] "
            "Start it with `make up`, or set GOALSMITH_RAG__QDRANT_PATH to run it in-process."
        )
        raise typer.Exit(1) from exc


# --- helpers ---------------------------------------------------------------------


@asynccontextmanager
async def _open_store(settings: Settings) -> AsyncIterator[DocStore]:
    client = open_client(settings.rag)
    try:
        yield DocStore(client)
    finally:
        await client.close()


def _retriever(settings: Settings, store: DocStore) -> HybridRetriever:
    return HybridRetriever(
        store,
        get_embeddings(settings),
        reranker=get_reranker(settings),
        candidates=settings.rag.candidates,
    )


def _warn(message: str) -> None:
    console.print(f"[yellow]{escape(message)}[/yellow]")


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
