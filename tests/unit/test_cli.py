from pathlib import Path

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from typer.testing import CliRunner

from goalsmith import cli
from goalsmith.capabilities.rag.store import DocStore
from goalsmith.config import RagSettings, Settings
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call
from tests.fixtures.fake_rag import HashEmbeddings

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(
        _env_file=None,
        workspace_root=tmp_path / "workspaces",
        rag=RagSettings(qdrant_path=tmp_path / "qdrant"),  # in-process Qdrant, no Docker
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

    result = runner.invoke(cli.app, ["run", "Write hello", "--run-id", "t1"])

    assert result.exit_code == 0, result.output
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
