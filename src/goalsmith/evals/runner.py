"""Golden-task eval: run the real agent on known tasks, check and judge the results.

    make eval-smoke     # tasks tagged "smoke": quick, runs on every PR
    make eval-full      # all tasks

Each task runs in its own temporary workspace with in-memory checkpoints and
memory. Tasks that need documents share an in-process Qdrant holding
evals/retrieval/corpus. Approvals are answered automatically: approve, except for
the tools a task lists in `reject_tools`.

Every task run is traced (tags "eval" plus the task's tags), so a failing task can
be opened in Langfuse. The report is a Markdown table, which CI posts on the PR.
"""

import argparse
import asyncio
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from goalsmith.capabilities.rag.ingest import ingest_files
from goalsmith.capabilities.rag.rerank import Reranker
from goalsmith.capabilities.rag.retrieve import HybridRetriever
from goalsmith.capabilities.rag.store import DocStore
from goalsmith.capabilities.tools.docs import search_docs_tool
from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.capabilities.tools.memory import recall_tool, remember_tool
from goalsmith.capabilities.tools.web import tavily_client, web_search_tool
from goalsmith.config import Settings, get_settings
from goalsmith.core.serde import checkpoint_serializer
from goalsmith.core.state import RunResult
from goalsmith.evals.checks import RunOutcome, all_passed, run_checks
from goalsmith.evals.judges import Judge, JudgeResult, render_record
from goalsmith.evals.tasks import (
    EVALS_DIR,
    GoldenTask,
    Need,
    Rubric,
    criteria_for,
    load_rubrics,
    load_tasks,
)
from goalsmith.harness.budget import Budget
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.permissions import Decision
from goalsmith.harness.tools import Tool, ToolContext, ToolRegistry
from goalsmith.llm import LLMConfigError, get_chat_model, get_embeddings, get_reranker
from goalsmith.memory.store import Memory, embedding_dims, in_memory_store
from goalsmith.observability.logs import configure_logging, get_logger
from goalsmith.observability.spans import observation, set_metadata, trace_attributes
from goalsmith.observability.tracing import setup_tracing, trace_url
from goalsmith.observability.usage import TokenUsage
from goalsmith.prompts import load_prompt

log = get_logger(__name__)

CORPUS = EVALS_DIR / "retrieval" / "corpus"
DOCS_COLLECTION = "docs"
EVAL_USER = "eval-user"
REJECTION_REASON = "The user rejected this action."

ModelFactory = Callable[[GoldenTask], BaseChatModel]


class TaskResult(BaseModel):
    id: str
    tags: list[str]
    passed: bool
    status: str = "not run"
    checks: list[tuple[str, bool, str]] = []  # (name, passed, detail)
    judge: JudgeResult | None = None
    steps: int = 0
    usage: TokenUsage = TokenUsage()
    seconds: float = 0.0
    trace_id: str | None = None
    error: str | None = None
    skipped: str | None = None  # why the task was not run

    @property
    def checks_passed(self) -> int:
        return sum(passed for _, passed, _ in self.checks)


@dataclass
class EvalEnv:
    """Everything tasks share: models and tools that are expensive to set up."""

    settings: Settings
    model_for: ModelFactory
    judge: Judge
    embeddings: Embeddings
    rubrics: dict[str, Rubric]
    docs: HybridRetriever | None = None  # None: tasks that need documents are skipped
    web_tool: Tool | None = None  # None: tasks that need the web are skipped
    work_dir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="goalsmith-eval-")))

    def missing(self, needs: Sequence[Need]) -> list[str]:
        available = {"docs": self.docs is not None, "web": self.web_tool is not None}
        return [need for need in needs if not available[need]]


async def run_task(task: GoldenTask, env: EvalEnv) -> TaskResult:
    if missing := env.missing(task.needs):
        return TaskResult(
            id=task.id, tags=task.tags, passed=False, skipped=f"needs {', '.join(missing)}"
        )

    started = time.perf_counter()
    with (
        trace_attributes(
            name=f"eval {task.id}",
            session_id=f"eval-{task.id}",
            user_id=EVAL_USER,
            tags=["eval", *task.tags],
            metadata={"task": task.id},
        ),
        observation("eval.task", input=task.goal, metadata={"task": task.id}) as span,
    ):
        try:
            result = await _run_and_grade(task, env)
        except Exception as exc:  # one broken task must not stop the suite
            log.exception("eval.task_crashed", task=task.id)
            result = TaskResult(
                id=task.id, tags=task.tags, passed=False, error=f"{type(exc).__name__}: {exc}"
            )
        result.seconds = round(time.perf_counter() - started, 1)
        set_metadata(span, passed=result.passed, steps=result.steps, cost_usd=result.usage.cost_usd)
    log.info("eval.task_finished", task=task.id, passed=result.passed, seconds=result.seconds)
    return result


async def _run_and_grade(task: GoldenTask, env: EvalEnv) -> TaskResult:
    workspace = env.work_dir / task.id
    workspace.mkdir(parents=True, exist_ok=True)
    for path, text in task.setup.files.items():
        (workspace / path).parent.mkdir(parents=True, exist_ok=True)
        (workspace / path).write_text(text)

    memory = Memory(
        in_memory_store(env.embeddings, await embedding_dims(env.embeddings)), EVAL_USER
    )
    for text in task.setup.memories:
        await memory.remember(text)

    loop = AgentLoop(
        env.model_for(task),
        ToolRegistry(_tools(task, env, memory)),
        ToolContext(
            run_id=f"eval-{task.id}",
            workspace=workspace,
            allowed_risks=frozenset({"read", "write", "external"}),
        ),
        budget=Budget(max_steps=task.max_steps),
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        prices=env.settings.prices,
    )
    result = await loop.run(task.goal)
    result = await _answer_approvals(loop, result, task)

    messages = await loop.messages()
    outcome = RunOutcome(
        status=result.status,
        answer=result.answer,
        steps=result.steps,
        tool_calls=[call["name"] for m in messages for call in getattr(m, "tool_calls", [])],
        files=_read_workspace(workspace),
        memories=[item.text for item in await memory.all()],
    )
    checks = run_checks(task.checks, outcome)
    criteria = criteria_for(task, env.rubrics)
    judged = await env.judge.grade(
        criteria,
        render_record(task.goal, messages, outcome.answer, outcome.files, status=outcome.status),
    )
    return TaskResult(
        id=task.id,
        tags=task.tags,
        passed=all_passed(checks) and judged.score >= task.pass_score,
        status=result.status,
        checks=[(c.name, c.passed, c.detail) for c in checks],
        judge=judged if criteria else None,
        steps=result.steps,
        usage=result.usage,
        trace_id=result.trace_id,
    )


async def _answer_approvals(loop: AgentLoop, result: RunResult, task: GoldenTask) -> RunResult:
    while result.status == "awaiting_approval":
        decisions = {
            r.call_id: Decision(action="reject", reason=REJECTION_REASON)
            if r.tool in task.reject_tools
            else Decision(action="approve")
            for r in result.pending
        }
        result = await loop.resume(decisions)
    return result


def _tools(task: GoldenTask, env: EvalEnv, memory: Memory) -> list[Tool]:
    tools = [read_file_tool(), write_file_tool(), recall_tool(memory), remember_tool(memory)]
    if env.docs is not None:
        tools.append(
            search_docs_tool(
                env.docs, default_collection=DOCS_COLLECTION, collections=[DOCS_COLLECTION]
            )
        )
    if env.web_tool is not None and "web" in task.needs:
        tools.append(env.web_tool)
    return tools


def _read_workspace(workspace: Path) -> dict[str, str]:
    files = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            try:
                files[path.relative_to(workspace).as_posix()] = path.read_text()
            except UnicodeDecodeError:
                files[path.relative_to(workspace).as_posix()] = "(binary file)"
    return files


async def run_suite(
    tasks: Sequence[GoldenTask], env: EvalEnv, *, concurrency: int = 3
) -> list[TaskResult]:
    gate = asyncio.Semaphore(concurrency)

    async def one(task: GoldenTask) -> TaskResult:
        async with gate:
            return await run_task(task, env)

    return list(await asyncio.gather(*(one(t) for t in tasks)))


@asynccontextmanager
async def docs_retriever(
    embeddings: Embeddings, reranker: Reranker | None, settings: Settings, corpus: Path = CORPUS
) -> AsyncIterator[HybridRetriever]:
    """The eval corpus in an in-process Qdrant, for tasks that need search_docs."""
    client = AsyncQdrantClient(location=":memory:")
    try:
        store = DocStore(client)
        await ingest_files(
            [corpus],
            collection=DOCS_COLLECTION,
            store=store,
            embeddings=embeddings,
            max_chars=settings.rag.chunk_chars,
            overlap=settings.rag.chunk_overlap,
        )
        yield HybridRetriever(
            store, embeddings, reranker=reranker, candidates=settings.rag.candidates
        )
    finally:
        await client.close()


# --- report ------------------------------------------------------------------------


@dataclass(frozen=True)
class SuiteInfo:
    name: str
    worker_model: str
    judge_model: str
    min_pass_rate: float
    seconds: float


def pass_rate(results: Sequence[TaskResult]) -> float:
    ran = [r for r in results if r.skipped is None]
    return sum(r.passed for r in ran) / len(ran) if ran else 0.0


def render(results: Sequence[TaskResult], info: SuiteInfo, settings: Settings) -> str:
    ran = [r for r in results if r.skipped is None]
    usage = sum((r.usage for r in ran), TokenUsage())
    judged = [r.judge.score for r in ran if r.judge is not None]
    rate = pass_rate(results)
    verdict = "✅" if rate >= info.min_pass_rate else "❌"
    prompt = load_prompt("agent_system")
    judge_prompt = load_prompt("judge")
    lines = [
        f"## Goalsmith eval: {info.name}",
        "",
        f"{verdict} **{sum(r.passed for r in ran)}/{len(ran)} tasks passed ({rate:.0%})**, "
        f"threshold {info.min_pass_rate:.0%}"
        + (f" · mean judge score {sum(judged) / len(judged):.2f}" if judged else "")
        + f" · {usage.summary()} · {info.seconds:.0f}s",
        "",
        f"Worker `{info.worker_model}` · judge `{info.judge_model}` · "
        f"prompts `{prompt.ref}` ({prompt.sha}), `{judge_prompt.ref}` ({judge_prompt.sha})",
        "",
        "| Task | Result | Checks | Judge | Steps | Cost | Trace |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.skipped:
            lines.append(f"| `{r.id}` | ⏭️ skipped ({r.skipped}) | | | | | |")
            continue
        mark = "✅" if r.passed else ("💥" if r.error else "❌")
        judge = f"{r.judge.score:.2f}" if r.judge else "-"
        lines.append(
            f"| `{r.id}` | {mark} {r.status} | {r.checks_passed}/{len(r.checks)} | {judge} "
            f"| {r.steps} | ${r.usage.cost_usd:.4f} | {_trace_cell(r, settings)} |"
        )
    failures = [r for r in ran if not r.passed]
    if failures:
        lines += ["", "<details><summary>Why tasks failed</summary>", ""]
        for r in failures:
            lines.append(f"**{r.id}**")
            if r.error:
                lines.append(f"- error: {r.error}")
            lines += [
                f"- check failed: {name} ({detail})" for name, ok, detail in r.checks if not ok
            ]
            if r.judge:
                lines += [f"- judge: {c} → {reason}" for c, reason in r.judge.failures()]
            lines.append("")
        lines.append("</details>")
    return "\n".join(lines) + "\n"


def _trace_cell(result: TaskResult, settings: Settings) -> str:
    if not result.trace_id:
        return ""
    if settings.langfuse.enabled:
        return f"[{result.trace_id[:8]}]({trace_url(settings, result.trace_id)})"
    return f"`{result.trace_id[:8]}`"


# --- command line ------------------------------------------------------------------


async def _main(args: argparse.Namespace, settings: Settings) -> int:
    tags = set(args.tag) if args.tag else None
    tasks = load_tasks(args.tasks, tags)
    if args.only:
        tasks = [t for t in tasks if t.id in set(args.only)]
    if not tasks:
        print("No tasks matched.", file=sys.stderr)
        return 2

    worker = get_chat_model("worker", settings)
    embeddings = get_embeddings(settings)
    web_tool = (
        web_search_tool(tavily_client(settings.tavily_api_key.get_secret_value()))
        if settings.tavily_api_key
        else None
    )
    started = time.perf_counter()
    async with docs_retriever(embeddings, get_reranker(settings), settings) as docs:
        env = EvalEnv(
            settings=settings,
            model_for=lambda task: worker,
            judge=Judge(get_chat_model("judge", settings)),
            embeddings=embeddings,
            rubrics=load_rubrics(args.rubrics),
            docs=docs,
            web_tool=web_tool,
        )
        results = await run_suite(tasks, env, concurrency=args.concurrency)

    info = SuiteInfo(
        name=args.name or (", ".join(sorted(tags)) if tags else "all tasks"),
        worker_model=settings.worker.model,
        judge_model=settings.judge.model,
        min_pass_rate=args.min_pass_rate,
        seconds=time.perf_counter() - started,
    )
    report = render(results, info, settings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    if args.json:
        args.json.write_text(
            "[\n" + ",\n".join(r.model_dump_json(indent=2) for r in results) + "\n]\n"
        )
    print(report)
    print(f"Report written to {args.out}", file=sys.stderr)
    return 0 if pass_rate(results) >= args.min_pass_rate else 1


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run golden-task evals against the agent.")
    parser.add_argument("--tag", action="append", help="Only tasks with this tag (repeatable).")
    parser.add_argument("--only", action="append", help="Only this task id (repeatable).")
    parser.add_argument("--name", help="Suite name in the report.")
    parser.add_argument("--tasks", type=Path, default=EVALS_DIR / "tasks")
    parser.add_argument("--rubrics", type=Path, default=EVALS_DIR / "rubrics")
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "results" / "report.md")
    parser.add_argument("--json", type=Path, help="Also write full results as JSON.")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    tracing = setup_tracing(settings)
    if tracing.warning:
        print(tracing.warning, file=sys.stderr)
    try:
        code = asyncio.run(_main(args, settings))
    except LLMConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        code = 2
    finally:
        tracing.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
