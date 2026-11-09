"""Golden-task evals: task files, checks, the judge and the runner (all offline)."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from goalsmith.config import Settings
from goalsmith.evals.checks import RunOutcome, run_checks
from goalsmith.evals.judges import CriterionVerdict, Judge, JudgeVerdicts, render_record
from goalsmith.evals.runner import (
    EvalEnv,
    SuiteInfo,
    TaskResult,
    docs_retriever,
    pass_rate,
    render,
    run_suite,
    run_task,
)
from goalsmith.evals.tasks import Checks, GoldenTask, criteria_for, load_rubrics, load_tasks
from goalsmith.prompts import load_prompt
from tests.fixtures.fake_llm import FakeChatModel, structured, text, tool_call, with_usage
from tests.fixtures.fake_rag import HashEmbeddings
from tests.fixtures.tracing import Spans, attr

EVALS = Path("evals")


# --- the task files in the repo -------------------------------------------------------


def test_golden_tasks_are_valid() -> None:
    tasks = load_tasks(EVALS / "tasks")
    rubrics = load_rubrics(EVALS / "rubrics")
    assert len(tasks) >= 15
    smoke = [t for t in tasks if "smoke" in t.tags]
    assert 4 <= len(smoke) <= 8  # quick enough to run on every PR
    for task in tasks:
        criteria_for(task, rubrics)  # every rubric a task names exists


def test_tag_filter(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(
        "- {id: one, goal: g, tags: [smoke], criteria: [c]}\n- {id: two, goal: g, criteria: [c]}\n"
    )
    assert [t.id for t in load_tasks(tmp_path, {"smoke"})] == ["one"]
    assert len(load_tasks(tmp_path)) == 2


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("- {id: one, goal: g, criteria: [c]}\n")
    (tmp_path / "b.yaml").write_text("- {id: one, goal: g, criteria: [c]}\n")
    with pytest.raises(ValueError, match="Duplicate task ids: one"):
        load_tasks(tmp_path)


def test_a_task_needs_some_grading() -> None:
    with pytest.raises(ValidationError, match="no checks and no judge criteria"):
        GoldenTask(id="x", goal="g", checks=Checks(status=None))


# --- checks ---------------------------------------------------------------------------


def outcome(**kwargs: object) -> RunOutcome:
    defaults: dict[str, object] = {"status": "completed", "answer": "", "steps": 1}
    return RunOutcome(**(defaults | kwargs))  # type: ignore[arg-type]


def failed(checks: Checks, run: RunOutcome) -> list[str]:
    return [r.name for r in run_checks(checks, run) if not r.passed]


def test_answer_checks_ignore_case_and_accept_alternatives() -> None:
    checks = Checks(
        answer_contains=["berlin", ["8 weeks", "eight weeks"]], answer_excludes=["evil"]
    )
    assert failed(checks, outcome(answer="Berlin office, Eight weeks notice.")) == []
    assert failed(checks, outcome(answer="Munich, evil")) == [
        "answer contains 'berlin'",
        "answer contains '8 weeks' or 'eight weeks'",
        "answer excludes 'evil'",
    ]


def test_citation_tool_file_and_memory_checks() -> None:
    checks = Checks(
        min_citations=2,
        tools_called=["search_docs"],
        tools_not_called=["remember"],
        files={"out.md": ["Tom"]},
        files_absent=["draft.md"],
        memories_contain=["9:30"],
        max_steps=3,
    )
    good = outcome(
        answer="See [handbook.md#1] and [faq.md#2], again [faq.md#2].",
        tool_calls=["search_docs"],
        files={"out.md": "- Tom fixes it"},
        memories=["Meetings at 9:30"],
        steps=3,
    )
    assert failed(checks, good) == []
    bad = outcome(
        answer="See [handbook.md#1] and [handbook.md#1].",  # one distinct source
        tool_calls=["remember"],
        files={"draft.md": "x"},
        steps=4,
    )
    assert failed(checks, bad) == [
        "cites at least 2 source(s)",
        "called search_docs",
        "did not call remember",
        "wrote out.md",
        "did not write draft.md",
        "memory contains '9:30'",
        "at most 3 steps",
    ]


def test_status_check() -> None:
    assert failed(Checks(), outcome(status="budget_exceeded")) == ["status is completed"]


# --- judge ----------------------------------------------------------------------------


async def test_judge_scores_criteria_and_fails_missing_verdicts(spans: Spans) -> None:
    verdicts = JudgeVerdicts(
        verdicts=[
            CriterionVerdict(number=1, passed=True, reason="Answer says 18 weeks."),
            CriterionVerdict(number=3, passed=True, reason="Cites [handbook.md#2]."),
        ]
    )
    model = FakeChatModel(responses=[structured(verdicts)])
    result = await Judge(model).grade(["answers", "grounded", "cites"], "## Goal\n...")

    assert [v.passed for v in result.verdicts] == [True, False, True]
    assert result.score == pytest.approx(2 / 3)
    assert result.failures() == [("grounded", "The judge gave no verdict.")]
    prompt = str(model.calls[0][0].content)
    assert "1. answers\n2. grounded\n3. cites" in prompt
    judge_span = spans.one("judge")
    assert attr(judge_span, "langfuse.observation.type") == "evaluator"
    assert attr(judge_span, "langfuse.observation.metadata.prompt") == "judge@v1"


async def test_no_criteria_means_no_model_call() -> None:
    model = FakeChatModel(responses=[])
    result = await Judge(model).grade([], "record")
    assert result.score == 1.0
    assert model.calls == []


def test_record_shows_calls_results_answer_and_files() -> None:
    messages = [
        HumanMessage("goal"),
        tool_call("read_file", {"path": "notes.md"}, "c1"),
        ToolMessage("Tom will fix it", tool_call_id="c1", name="read_file"),
        AIMessage("Done."),
    ]
    record = render_record(
        "List actions", messages, "Done.", {"actions.md": "- Tom"}, status="completed"
    )
    assert '1. The agent called `read_file` with {"path": "notes.md"}' in record
    assert "result: Tom will fix it" in record
    assert "## Final answer (run status: completed)\nDone." in record
    assert "### actions.md\n- Tom" in record


# --- runner ---------------------------------------------------------------------------


def judge_saying(*passed: bool) -> Judge:
    verdicts = [CriterionVerdict(number=i, passed=p, reason="r") for i, p in enumerate(passed, 1)]
    return Judge(FakeChatModel(responses=[structured(JudgeVerdicts(verdicts=verdicts))]))


def make_env(tmp_path: Path, models: dict[str, FakeChatModel], judge: Judge) -> EvalEnv:
    return EvalEnv(
        settings=Settings(_env_file=None),
        model_for=lambda task: models[task.id],
        judge=judge,
        embeddings=HashEmbeddings(),
        rubrics=load_rubrics(EVALS / "rubrics"),
        work_dir=tmp_path,
    )


async def test_file_task_passes_with_checks_and_judge(tmp_path: Path, spans: Spans) -> None:
    task = GoldenTask(
        id="notes",
        goal="Copy the owner to out.md",
        tags=["smoke"],
        setup={"files": {"notes.md": "Owner: Tom"}},  # type: ignore[arg-type]
        checks=Checks(tools_called=["read_file", "write_file"], files={"out.md": ["Tom"]}),
        criteria=["out.md names the owner"],
    )
    worker = FakeChatModel(
        responses=[
            with_usage(tool_call("read_file", {"path": "notes.md"}, "r"), 100, 10),
            tool_call("write_file", {"path": "out.md", "content": "Tom"}, "w"),  # auto-approved
            text("Wrote out.md."),
        ]
    )
    result = await run_task(task, make_env(tmp_path, {"notes": worker}, judge_saying(True)))

    assert result.passed, result
    assert (result.status, result.steps, result.checks_passed) == ("completed", 3, 5)
    assert result.judge is not None and result.judge.score == 1.0
    assert result.usage.input_tokens == 100
    # The whole task is one trace, tagged for filtering in Langfuse.
    task_span = spans.one("eval.task")
    assert result.trace_id == format(task_span.context.trace_id, "032x")
    assert list(attr(spans.one("agent.run"), "langfuse.trace.tags")) == ["eval", "smoke"]


async def test_rejected_tool_never_runs(tmp_path: Path) -> None:
    task = GoldenTask(
        id="haiku",
        goal="Save a haiku",
        reject_tools=["write_file"],
        checks=Checks(files_absent=["haiku.md"]),
    )
    worker = FakeChatModel(
        responses=[
            tool_call("write_file", {"path": "haiku.md", "content": "..."}, "w"),
            text("You declined, so nothing was saved."),
        ]
    )
    result = await run_task(task, make_env(tmp_path, {"haiku": worker}, judge_saying()))
    assert result.passed
    rejection = next(m for m in worker.calls[1] if isinstance(m, ToolMessage))
    assert "rejected" in str(rejection.content)


async def test_failing_judge_fails_the_task(tmp_path: Path) -> None:
    task = GoldenTask(id="q", goal="Answer", criteria=["a", "b"], pass_score=0.75)
    worker = FakeChatModel(responses=[text("answer")])
    result = await run_task(task, make_env(tmp_path, {"q": worker}, judge_saying(True, False)))
    assert not result.passed
    assert result.judge is not None and result.judge.score == 0.5


async def test_memory_setup_and_check(tmp_path: Path) -> None:
    task = GoldenTask(
        id="mem",
        goal="Where do I work? Also remember I like tea.",
        setup={"memories": ["Works in Berlin"]},  # type: ignore[arg-type]
        checks=Checks(answer_contains=["Berlin"], memories_contain=["tea"]),
    )
    worker = FakeChatModel(
        responses=[
            tool_call("recall", {"query": "office"}, "r"),
            tool_call("remember", {"text": "Likes tea"}, "m"),  # a write: auto-approved
            text("You work in Berlin."),
        ]
    )
    result = await run_task(task, make_env(tmp_path, {"mem": worker}, judge_saying()))
    assert result.passed, result.checks
    assert "Works in Berlin" in str(worker.calls[1][-1].content)


async def test_unavailable_needs_skip_and_crashes_are_contained(tmp_path: Path) -> None:
    docs_task = GoldenTask(id="d", goal="g", needs=["docs"], criteria=["c"])
    broken = GoldenTask(id="b", goal="g", criteria=["c"])
    ok = GoldenTask(id="ok", goal="g", checks=Checks(answer_contains=["hi"]))
    models = {"b": FakeChatModel(responses=[]), "ok": FakeChatModel(responses=[text("hi")])}

    results = await run_suite([docs_task, broken, ok], make_env(tmp_path, models, judge_saying()))

    by_id = {r.id: r for r in results}
    assert by_id["d"].skipped == "needs docs"
    assert by_id["b"].error is not None and "ran out of scripted responses" in by_id["b"].error
    assert by_id["ok"].passed
    assert pass_rate(results) == 0.5  # skipped tasks do not count


async def test_docs_tasks_search_the_eval_corpus(tmp_path: Path) -> None:
    task = GoldenTask(
        id="leave",
        goal="Parental leave?",
        needs=["docs"],
        checks=Checks(tools_called=["search_docs"]),
    )
    worker = FakeChatModel(
        responses=[tool_call("search_docs", {"query": "parental leave weeks"}, "s"), text("18")]
    )
    env = make_env(tmp_path, {"leave": worker}, judge_saying())
    settings = Settings(_env_file=None)
    async with docs_retriever(HashEmbeddings(), None, settings) as docs:
        env.docs = docs
        result = await run_task(task, env)
    assert result.passed
    assert "[handbook.md#" in str(worker.calls[1][-1].content)


def test_report_table_and_failure_details() -> None:
    results = [
        TaskResult(id="a", tags=[], passed=True, status="completed", checks=[("x", True, "")]),
        TaskResult(
            id="b",
            tags=[],
            passed=False,
            status="completed",
            checks=[("answer contains '18'", False, "not found in: 16")],
            trace_id="0123456789abcdef0123456789abcdef",
        ),
        TaskResult(id="c", tags=[], passed=False, skipped="needs web"),
    ]
    info = SuiteInfo("smoke", "worker-m", "judge-m", min_pass_rate=0.8, seconds=12)
    report = render(results, info, Settings(_env_file=None))

    assert "❌ **1/2 tasks passed (50%)**, threshold 80%" in report
    assert "| `a` | ✅ completed | 1/1 |" in report
    assert "| `b` | ❌ completed | 0/1 | - | 0 | $0.0000 | `01234567` |" in report
    assert "| `c` | ⏭️ skipped (needs web) |" in report
    assert "- check failed: answer contains '18' (not found in: 16)" in report
    assert f"`{load_prompt('agent_system').ref}`" in report
