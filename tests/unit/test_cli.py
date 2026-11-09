from pathlib import Path

import pytest
from typer.testing import CliRunner

from goalsmith import cli
from goalsmith.config import Settings
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call

runner = CliRunner()


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    s = Settings(_env_file=None, workspace_root=tmp_path / "workspaces")
    monkeypatch.setattr(cli, "get_settings", lambda: s)
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
