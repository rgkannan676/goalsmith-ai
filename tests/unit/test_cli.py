from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr
from qdrant_client.http.exceptions import ResponseHandlingException
from typer.testing import CliRunner

from goalsmith import cli
from goalsmith.capabilities.rag.store import DocStore
from goalsmith.config import LangfuseSettings, RagSettings, Settings
from goalsmith.core.persistence import Persistence
from goalsmith.memory.store import in_memory_store
from goalsmith.observability.tracing import Tracing
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call, with_usage
from tests.fixtures.fake_rag import HashEmbeddings
from tests.fixtures.tracing import Spans, attr

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(
        _env_file=None,
        workspace_root=tmp_path / "workspaces",
        rag=RagSettings(qdrant_path=tmp_path / "qdrant"),  # in-process Qdrant, no Docker
        postgres_url=None,  # in-memory checkpoints and memory, no Docker
    )
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    monkeypatch.setattr(cli, "get_embeddings", lambda settings=None: HashEmbeddings())
    monkeypatch.setattr(cli, "get_reranker", lambda settings=None: None)
    return s


def use_model(monkeypatch: pytest.MonkeyPatch, model: FakeChatModel) -> None:
    monkeypatch.setattr(cli, "get_chat_model", lambda role, settings=None: model)


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "goalsmith 0.1.0" in result.output


def test_run_prints_answer_and_workspace(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeChatModel(
        responses=[
            tool_call("write_file", {"path": "out.md", "content": "hello"}),
            text("All done."),
        ]
    )
    use_model(monkeypatch, model)

    result = runner.invoke(cli.app, ["run", "Write hello", "--run-id", "t1"], input="a\n")

    assert result.exit_code == 0, result.output
    assert "Approve write_file?" in result.output
    assert "All done." in result.output
    assert "write_file" in result.output  # live progress shows tool calls
    assert "completed" in result.output
    assert (settings.workspace_root / "t1" / "out.md").read_text() == "hello"


def test_run_without_search_key_still_works(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[text("ok")]))
    result = runner.invoke(cli.app, ["run", "Hi", "--run-id", "t2"])
    assert result.exit_code == 0
    assert "TAVILY_API_KEY" in result.output  # warns that web_search is off


def test_budget_exceeded_exits_nonzero(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    endless = [tool_call("read_file", {"path": "x"}, call_id=f"c{i}") for i in range(10)]
    use_model(monkeypatch, FakeChatModel(responses=endless))
    result = runner.invoke(cli.app, ["run", "Loop", "--run-id", "t3", "--max-steps", "2"])
    assert result.exit_code == 2
    assert "budget_exceeded" in result.output


def test_missing_api_key_is_a_friendly_error(settings: Settings) -> None:
    result = runner.invoke(cli.app, ["run", "Hi", "--run-id", "t4"])
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


HANDBOOK = "# Handbook\n\n## Parental leave\n\nUp to 16 weeks of paid parental leave.\n"


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text(HANDBOOK)
    return folder


def test_ingest_then_search(settings: Settings, docs: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", str(docs)])
    assert result.exit_code == 0, result.output
    assert "handbook.md" in result.output
    assert "'docs'" in result.output

    result = runner.invoke(cli.app, ["search", "parental leave weeks", "-k", "1"])
    assert result.exit_code == 0, result.output
    assert "[handbook.md#0]" in result.output
    assert "16 weeks" in result.output


def test_ingest_missing_file_exits_1(settings: Settings, tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["ingest", str(tmp_path / "nope.md")])
    assert result.exit_code == 1
    assert "No such file" in result.output


def test_search_unknown_collection_exits_1(settings: Settings, docs: Path) -> None:
    runner.invoke(cli.app, ["ingest", str(docs)])
    result = runner.invoke(cli.app, ["search", "leave", "--collection", "specs"])
    assert result.exit_code == 1
    assert "Available: docs" in result.output


def test_run_gets_search_docs_once_documents_exist(
    settings: Settings, docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(cli.app, ["ingest", str(docs)])
    model = FakeChatModel(
        responses=[
            tool_call("search_docs", {"query": "parental leave"}),
            text("16 weeks [handbook.md#0]."),
        ]
    )
    use_model(monkeypatch, model)

    result = runner.invoke(cli.app, ["run", "How long is parental leave?", "--run-id", "t5"])

    assert result.exit_code == 0, result.output
    assert "search_docs" in [t["function"]["name"] for t in model.bound_tools]
    assert "[handbook.md#0]" in result.output


def test_run_without_documents_warns(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeChatModel(responses=[text("ok")])
    use_model(monkeypatch, model)
    result = runner.invoke(cli.app, ["run", "Hi", "--run-id", "t6"])
    assert result.exit_code == 0
    assert "no documents yet" in result.output
    assert "search_docs" not in [t["function"]["name"] for t in model.bound_tools]


def test_run_without_qdrant_still_works(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unreachable(self: DocStore) -> list[str]:
        raise ResponseHandlingException(ConnectionError("refused"))

    monkeypatch.setattr(DocStore, "collections", unreachable)
    use_model(monkeypatch, FakeChatModel(responses=[text("ok")]))
    result = runner.invoke(cli.app, ["run", "Hi", "--run-id", "t7"])
    assert result.exit_code == 0
    assert "Qdrant is not reachable" in result.output


# --- approvals and resume ------------------------------------------------------------

WRITE_PLAN = tool_call("write_file", {"path": "plan.md", "content": "v1"}, call_id="w1")


@pytest.fixture
def durable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand-in for Postgres: one saver and store shared by every CLI call in the test,
    so a run paused by one `goalsmith` call can be resumed by the next."""
    saver = InMemorySaver()
    store = in_memory_store(HashEmbeddings(), 64)

    @asynccontextmanager
    async def fake(*args: object, **kwargs: object) -> AsyncIterator[Persistence]:
        yield Persistence(checkpointer=saver, store=store, durable=True)

    monkeypatch.setattr(cli, "open_persistence", fake)


def test_reject_at_the_prompt(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[WRITE_PLAN, text("Not written.")]))
    result = runner.invoke(cli.app, ["run", "Plan", "--run-id", "t8"], input="r\nwrong file\n")
    assert result.exit_code == 0, result.output
    assert "write_file: rejected" in result.output
    assert not (settings.workspace_root / "t8" / "plan.md").exists()


def test_edit_at_the_prompt(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[WRITE_PLAN, text("ok")]))
    edited = '{"path": "plan.md", "content": "edited"}'
    result = runner.invoke(
        cli.app, ["run", "Plan", "--run-id", "t9"], input=f"e\nnot json\n{edited}\n"
    )
    assert result.exit_code == 0, result.output
    assert "Not valid JSON" in result.output  # asked again after bad input
    assert (settings.workspace_root / "t9" / "plan.md").read_text() == "edited"


def test_yes_flag_skips_prompts(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[WRITE_PLAN, text("ok")]))
    result = runner.invoke(cli.app, ["run", "Plan", "--run-id", "t10", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Approve write_file?" not in result.output
    assert (settings.workspace_root / "t10" / "plan.md").exists()


def test_pause_then_resume_in_a_later_command(
    settings: Settings, durable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[WRITE_PLAN]))
    paused = runner.invoke(cli.app, ["run", "Plan", "--run-id", "t11"], input="")  # EOF
    assert paused.exit_code == cli.EXIT_PAUSED, paused.output
    assert "goalsmith resume t11" in paused.output
    assert not (settings.workspace_root / "t11" / "plan.md").exists()

    use_model(monkeypatch, FakeChatModel(responses=[text("Plan saved.")]))
    resumed = runner.invoke(cli.app, ["resume", "t11"], input="a\n")
    assert resumed.exit_code == 0, resumed.output
    assert "Plan saved." in resumed.output
    assert (settings.workspace_root / "t11" / "plan.md").read_text() == "v1"


def test_pause_without_postgres_cannot_resume(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[WRITE_PLAN]))
    paused = runner.invoke(cli.app, ["run", "Plan", "--run-id", "t12"], input="")
    assert paused.exit_code == cli.EXIT_PAUSED
    assert "cannot resume" in paused.output

    resumed = runner.invoke(cli.app, ["resume", "t12"])
    assert resumed.exit_code == 1
    assert "Postgres is not reachable" in resumed.output


def test_resume_a_run_that_is_not_paused(
    settings: Settings, durable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[]))
    result = runner.invoke(cli.app, ["resume", "never-started"])
    assert result.exit_code == 1
    assert "not waiting for approval" in result.output


def test_run_id_cannot_be_reused(
    settings: Settings, durable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_model(monkeypatch, FakeChatModel(responses=[text("one"), text("two")]))
    assert runner.invoke(cli.app, ["run", "Hi", "--run-id", "t13"]).exit_code == 0
    again = runner.invoke(cli.app, ["run", "Hi", "--run-id", "t13"])
    assert again.exit_code == 1
    assert "already exists" in again.output


def test_run_prints_usage_and_trace_link(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, spans: Spans
) -> None:
    traced = settings.model_copy(
        update={"langfuse": LangfuseSettings(public_key="pk", secret_key=SecretStr("sk"))}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: traced)
    # The test tracer provider is already installed; don't send anything to Langfuse.
    monkeypatch.setattr(cli, "setup_tracing", lambda s: Tracing(provider=None))
    reply = with_usage(text("Hi."), 1000, 100, model="claude-sonnet-5-5")
    use_model(monkeypatch, FakeChatModel(responses=[reply]))

    result = runner.invoke(cli.app, ["run", "Say hi", "--run-id", "t9"])

    assert result.exit_code == 0, result.output
    assert "1,000 in / 100 out tokens · $0.0030" in result.output  # 1000x$2 + 100x$10 per M
    root = spans.one("goalsmith run")
    trace_id = format(root.context.trace_id, "032x")
    assert f"Trace: http://localhost:3000/project/goalsmith/traces/{trace_id}" in result.output
    run = spans.one("agent.run")
    assert run.parent is not None and run.parent.span_id == root.context.span_id
    assert attr(run, "langfuse.trace.name") == "goalsmith run"  # outermost name wins
    assert attr(run, "langfuse.user.id") == "local"
