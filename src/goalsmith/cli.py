"""Command line: `goalsmith run`, `resume`, `ingest` and `search`."""

import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import typer
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore
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
from goalsmith.capabilities.tools.memory import recall_tool, remember_tool
from goalsmith.capabilities.tools.web import tavily_client, web_search_tool
from goalsmith.config import Settings, get_settings
from goalsmith.core.persistence import PersistenceError, open_persistence
from goalsmith.core.state import RunResult
from goalsmith.harness.budget import Budget
from goalsmith.harness.loop import AgentLoop, RunStateError
from goalsmith.harness.permissions import ApprovalRequest, Decision, Permissions
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from goalsmith.llm import LLMConfigError, get_chat_model, get_embeddings, get_reranker
from goalsmith.memory.store import Memory, MemoryStoreError
from goalsmith.observability.logs import configure_logging
from goalsmith.observability.spans import observation, trace_attributes
from goalsmith.observability.tracing import setup_tracing, trace_url

app = typer.Typer(help="Goalsmith: from goal to done.", no_args_is_help=True)
console = Console()

EXIT_BUDGET_EXCEEDED = 2
EXIT_PAUSED = 3  # stopped at an approval prompt; continue with `goalsmith resume`

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


# --- run and resume --------------------------------------------------------------

MaxStepsOption = Annotated[
    int | None, typer.Option(min=1, help="Maximum model calls (default from settings).")
]
YesOption = Annotated[
    bool, typer.Option("--yes", "-y", help="Approve every action without asking.")
]


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="What you want done, in plain English.")],
    max_steps: MaxStepsOption = None,
    run_id: Annotated[str | None, typer.Option(help="Name for this run's workspace.")] = None,
    yes: YesOption = False,
) -> None:
    """Run a single agent on GOAL and print the result.

    Before a tool that writes or acts on the outside world runs, you are asked to
    approve, edit or reject it.
    """
    settings = get_settings()
    model = _worker_model(settings)
    context = _context(settings, run_id or _new_run_id())
    budget = Budget(max_steps=max_steps or settings.max_steps)

    console.print(f"[bold]Run[/bold] {context.run_id}  ·  budget {budget.max_steps} steps")
    with _observed(settings, "goalsmith run", context.run_id, goal):
        result = _run_session(
            _start(goal, model, context, budget, settings, auto_approve=yes), settings
        )
        _finish(result, context, settings)


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="The run to continue.")],
    max_steps: MaxStepsOption = None,
    yes: YesOption = False,
) -> None:
    """Continue a run that is waiting for approval, even after a restart."""
    settings = get_settings()
    model = _worker_model(settings)
    context = _context(settings, run_id)
    budget = Budget(max_steps=max_steps or settings.max_steps)

    console.print(f"[bold]Resume[/bold] {run_id}")
    with _observed(settings, "goalsmith resume", run_id, run_id):
        result = _run_session(
            _continue(model, context, budget, settings, auto_approve=yes), settings
        )
        _finish(result, context, settings)


@contextmanager
def _observed(settings: Settings, name: str, run_id: str, input: str) -> Iterator[None]:
    """Logs, and one trace for the whole command (run plus any approvals).

    Spans are sent in batches, so `shutdown()` flushes the last ones before exit.
    """
    configure_logging(settings.log_level, settings.log_format)
    tracing = setup_tracing(settings)
    if tracing.warning:
        _warn(tracing.warning)
    try:
        with (
            trace_attributes(name=name, session_id=run_id, user_id=settings.user_id, tags=["cli"]),
            observation(name, input=input),
        ):
            yield
    finally:
        tracing.shutdown()


async def _start(
    goal: str,
    model: BaseChatModel,
    context: ToolContext,
    budget: Budget,
    settings: Settings,
    *,
    auto_approve: bool,
) -> RunResult:
    # One event loop for the whole run: the Qdrant and Postgres clients must live on
    # the same loop as the agent that uses them.
    async with _open_session(settings, model, context, budget) as (loop, durable):
        result = await loop.run(goal, on_update=_print_update)
        return await _handle_approvals(loop, result, durable, auto_approve=auto_approve)


async def _continue(
    model: BaseChatModel,
    context: ToolContext,
    budget: Budget,
    settings: Settings,
    *,
    auto_approve: bool,
) -> RunResult:
    async with _open_session(settings, model, context, budget, require_durable=True) as (
        loop,
        durable,
    ):
        pending = await loop.pending()
        if not pending:
            raise RunStateError(f"Run {context.run_id!r} is not waiting for approval.")
        result = RunResult(
            run_id=context.run_id, status="awaiting_approval", answer="", steps=0, pending=pending
        )
        return await _handle_approvals(loop, result, durable, auto_approve=auto_approve)


@asynccontextmanager
async def _open_session(
    settings: Settings,
    model: BaseChatModel,
    context: ToolContext,
    budget: Budget,
    *,
    require_durable: bool = False,
) -> AsyncIterator[tuple[AgentLoop, bool]]:
    """Open document search and persistence, build the tools and the agent loop."""
    embeddings = get_embeddings(settings)
    async with (
        _open_store(settings) as docs,
        open_persistence(
            settings.postgres_dsn(), embeddings, require_durable=require_durable
        ) as saved,
    ):
        if not saved.durable:
            _warn("Postgres is off: approvals and memories last only for this run (`make up`).")
        tools = await _build_tools(settings, docs, embeddings, saved.store)
        loop = AgentLoop(
            model,
            ToolRegistry(tools),
            context,
            budget=budget,
            permissions=Permissions(needs_approval=settings.approval_risks),
            checkpointer=saved.checkpointer,
            prices=settings.prices,
        )
        yield loop, saved.durable


async def _handle_approvals(
    loop: AgentLoop, result: RunResult, durable: bool, *, auto_approve: bool
) -> RunResult:
    """Ask the human about each paused tool call, resume, and repeat until done."""
    while result.status == "awaiting_approval":
        if auto_approve:
            decisions = {r.call_id: Decision(action="approve") for r in result.pending}
            console.print(f"  [dim]auto-approved {len(decisions)} action(s)[/dim]")
        else:
            try:
                decisions = {r.call_id: _ask(r) for r in result.pending}
            except typer.Abort:  # Ctrl+C or end of input at the prompt
                console.print()
                if durable:
                    console.print(
                        f"[yellow]Paused.[/yellow] Continue later with: "
                        f"goalsmith resume {result.run_id}"
                    )
                else:
                    _warn("Stopped. This run was not saved (Postgres is off), so it cannot resume.")
                raise typer.Exit(EXIT_PAUSED) from None
        result = await loop.resume(decisions, on_update=_print_update)
    return result


def _ask(request: ApprovalRequest) -> Decision:
    body = "\n".join(
        f"[bold]{escape(key)}[/bold]: {escape(_short(value, 300))}"
        for key, value in request.args.items()
    )
    console.print(
        Panel(body or "(no arguments)", title=f"Approve {request.tool}? (risk: {request.risk})")
    )
    choice = ""
    while choice not in ("a", "e", "r"):
        # Typer bundles its own click, so validate here instead of using click.Choice.
        choice = typer.prompt("[a]pprove, [e]dit or [r]eject").strip().lower()[:1]
    if choice == "a":
        return Decision(action="approve")
    if choice == "e":
        return Decision(action="edit", args=_prompt_args(request.args))
    reason = typer.prompt("Reason (optional)", default="", show_default=False)
    return Decision(action="reject", reason=reason)


def _prompt_args(current: dict[str, Any]) -> dict[str, Any]:
    while True:
        raw = typer.prompt("New arguments as JSON", default=json.dumps(current))
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Not valid JSON:[/red] {escape(str(exc))}")
            continue
        if isinstance(value, dict):
            return value
        console.print('[red]Expected a JSON object, e.g. {"path": "notes.md"}[/red]')


def _finish(result: RunResult, context: ToolContext, settings: Settings) -> None:
    console.print(Panel(Markdown(result.answer), title=f"{result.status} · {result.steps} steps"))
    console.print(f"Usage: {result.usage.summary()}")
    console.print(f"Workspace: {context.workspace}")
    if settings.langfuse.enabled and result.trace_id:
        # soft_wrap keeps the link on one line, so terminals can open it.
        console.print(f"Trace: {trace_url(settings, result.trace_id)}", soft_wrap=True)
    if result.status == "budget_exceeded":
        raise typer.Exit(EXIT_BUDGET_EXCEEDED)


def _run_session(coro: Coroutine[Any, Any, RunResult], settings: Settings) -> RunResult:
    try:
        return asyncio.run(coro)
    except (RunStateError, PersistenceError, MemoryStoreError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


def _worker_model(settings: Settings) -> BaseChatModel:
    try:
        return get_chat_model("worker", settings)
    except LLMConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc


def _context(settings: Settings, run_id: str) -> ToolContext:
    workspace = settings.workspace_root / run_id
    workspace.mkdir(parents=True, exist_ok=True)
    # Every risk level may run; write and external ones wait for approval first.
    return ToolContext(
        run_id=run_id, workspace=workspace, allowed_risks=frozenset({"read", "write", "external"})
    )


async def _build_tools(
    settings: Settings, docs: DocStore, embeddings: Embeddings, memory_store: BaseStore
) -> list[Tool]:
    tools = [read_file_tool(), write_file_tool()]
    if settings.tavily_api_key is None:
        _warn("web_search is off: TAVILY_API_KEY is not set.")
    else:
        tools.append(web_search_tool(tavily_client(settings.tavily_api_key.get_secret_value())))

    docs_tool = await _docs_tool(settings, docs, embeddings)
    if docs_tool is not None:
        tools.append(docs_tool)

    memory = Memory(memory_store, settings.user_id)
    tools += [recall_tool(memory), remember_tool(memory)]
    return tools


async def _docs_tool(settings: Settings, store: DocStore, embeddings: Embeddings) -> Tool | None:
    try:
        collections = await store.collections()
    except ApiException:
        _warn(f"search_docs is off: Qdrant is not reachable at {settings.rag.qdrant_url}.")
        return None
    if not collections:
        _warn("search_docs is off: no documents yet. Add some with `goalsmith ingest <files>`.")
        return None
    return search_docs_tool(
        _retriever(settings, store, embeddings),
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


def _retriever(
    settings: Settings, store: DocStore, embeddings: Embeddings | None = None
) -> HybridRetriever:
    return HybridRetriever(
        store,
        embeddings or get_embeddings(settings),
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
        elif node == "approval" and getattr(message, "status", None) == "error":
            console.print(f"  [red]✗ {message.name}: rejected[/red]")
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
