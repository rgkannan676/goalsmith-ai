"""Tracing, token/cost accounting and structured logs (M5)."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from langchain_core.messages import AIMessage

from goalsmith.capabilities.tools.files import read_file_tool, write_file_tool
from goalsmith.config import ModelPrice, Settings
from goalsmith.harness.loop import AgentLoop
from goalsmith.harness.permissions import AUTO_APPROVE, Decision
from goalsmith.harness.tools import ToolContext, ToolRegistry
from goalsmith.observability.logs import bind_run, configure_logging, get_logger
from goalsmith.observability.spans import observation, set_capture_content, trace_attributes
from goalsmith.observability.tracing import (
    OTLP_PATH,
    ModelCallTracer,
    langfuse_exporter,
    langfuse_headers,
    langfuse_reachable,
    setup_tracing,
    trace_url,
)
from goalsmith.observability.usage import TokenUsage, cost_usd, usage_from_message
from goalsmith.prompts import load_prompt
from tests.fixtures.fake_llm import FakeChatModel, text, tool_call, with_usage
from tests.fixtures.tracing import Spans, attr

PRICES = {"fake-model": ModelPrice(input=2.0, output=10.0, cache_read=0.2, cache_write=2.5)}


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(run_id="run-1", workspace=tmp_path)


def traced_model(responses: list[AIMessage]) -> FakeChatModel:
    model = FakeChatModel(responses=responses)
    model.callbacks = [ModelCallTracer(PRICES)]  # what llm.get_chat_model attaches
    return model


# --- tokens and cost ------------------------------------------------------------------


def test_cost_prices_cached_input_separately() -> None:
    message = with_usage(
        text("hi"), input_tokens=1_000_000, output_tokens=100_000, cache_read=400_000
    )
    usage = usage_from_message(message, PRICES)
    # 600k uncached x $2 + 400k cache reads x $0.20 + 100k output x $10, per million
    assert usage.cost_usd == pytest.approx(1.2 + 0.08 + 1.0)
    assert usage.cache_read_tokens == 400_000
    assert usage.unpriced_calls == 0


def test_unknown_model_counts_tokens_but_flags_missing_cost() -> None:
    usage = usage_from_message(with_usage(text("hi"), 10, 5, model="mystery"), PRICES)
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (10, 5, 0.0)
    assert usage.unpriced_calls == 1
    assert "unpriced" in usage.summary()


def test_no_usage_reported_means_zero() -> None:
    assert usage_from_message(text("hi"), PRICES) == TokenUsage()


def test_usage_adds_up() -> None:
    total = TokenUsage(input_tokens=1, output_tokens=2, cost_usd=0.5) + TokenUsage(
        input_tokens=3, output_tokens=4, cost_usd=0.25
    )
    assert (total.input_tokens, total.output_tokens, total.cost_usd) == (4, 6, 0.75)
    assert total.summary() == "4 in / 6 out tokens · $0.7500"


def test_default_prices_cover_the_default_models() -> None:
    settings = Settings(_env_file=None)
    for role in ("planner", "worker", "router", "judge"):
        assert settings.model_for(role).model in settings.prices  # type: ignore[arg-type]
    price = settings.prices["claude-sonnet-5-5"]
    assert cost_usd(TokenUsage(input_tokens=1_000_000), price) == pytest.approx(2.0)


# --- the agent loop's traces ----------------------------------------------------------


async def test_run_trace_has_agent_model_and_tool_spans(ctx: ToolContext, spans: Spans) -> None:
    (ctx.workspace / "a.md").write_text("A")
    model = traced_model(
        [
            with_usage(tool_call("read_file", {"path": "a.md"}), 100, 20),
            with_usage(text("It says A."), 150, 10),
        ]
    )
    loop = AgentLoop(model, ToolRegistry([read_file_tool()]), ctx, prices=PRICES)
    result = await loop.run("What does a.md say?")

    run = spans.one("agent.run")
    generations = spans.of_type("generation")
    tool = spans.one("tool read_file")
    assert len(generations) == 2
    # One trace: model and tool spans are children of the run span.
    assert {s.context.trace_id for s in spans.all()} == {run.context.trace_id}
    assert all(s.parent and s.parent.span_id == run.context.span_id for s in [*generations, tool])
    assert result.trace_id == format(run.context.trace_id, "032x")

    # Every span carries the session (= run id) and the prompt version.
    prompt = load_prompt("agent_system")
    for span in spans.all():
        assert attr(span, "langfuse.session.id") == "run-1"
        assert attr(span, "langfuse.trace.metadata.prompt_version") == str(prompt.version)

    first = generations[0]
    assert attr(first, "langfuse.observation.model.name") == "fake-model"
    assert json.loads(attr(first, "langfuse.observation.usage_details"))["input"] == 100
    assert json.loads(attr(first, "langfuse.observation.cost_details"))["total"] > 0
    assert attr(first, "langfuse.observation.metadata.prompt_sha") == prompt.sha
    assert json.loads(attr(tool, "langfuse.observation.input")) == {"path": "a.md"}
    assert attr(tool, "langfuse.observation.output") == "A"

    # Tokens and cost for the whole run, on the result and on the run span.
    assert (result.usage.input_tokens, result.usage.output_tokens) == (250, 30)
    expected = (250 * 2.0 + 30 * 10.0) / 1_000_000
    assert result.usage.cost_usd == pytest.approx(expected)
    assert attr(run, "langfuse.observation.metadata.cost_usd") == pytest.approx(expected)
    assert attr(run, "langfuse.observation.output") == "It says A."


async def test_failed_tool_call_is_an_error_span(ctx: ToolContext, spans: Spans) -> None:
    model = traced_model([tool_call("read_file", {"path": "missing.md"}), text("Not found.")])
    await AgentLoop(model, ToolRegistry([read_file_tool()]), ctx).run("Read missing.md")

    tool = spans.one("tool read_file")
    assert attr(tool, "langfuse.observation.level") == "ERROR"
    assert not tool.status.is_ok


async def test_approval_pause_and_resume_share_a_session(ctx: ToolContext, spans: Spans) -> None:
    write = with_usage(tool_call("write_file", {"path": "p.md", "content": "x"}, "w1"), 50, 5)
    model = traced_model([write, with_usage(text("Written."), 60, 5)])
    loop = AgentLoop(model, ToolRegistry([write_file_tool()]), ctx, prices=PRICES)

    paused = await loop.run("Write p.md")
    requested = spans.one("approval.requested")
    assert json.loads(attr(requested, "langfuse.observation.input"))[0]["call_id"] == "w1"

    done = await loop.resume({"w1": Decision(action="approve")})
    resume = spans.one("agent.resume")
    decided = spans.one("approval.decided")
    assert json.loads(attr(decided, "langfuse.observation.input"))["w1"]["action"] == "approve"
    # Run and resume are separate traces, grouped by the session id.
    assert paused.trace_id != done.trace_id
    assert attr(resume, "langfuse.session.id") == attr(
        spans.one("agent.run"), "langfuse.session.id"
    )
    # Usage was checkpointed with the run, so the total covers both parts.
    assert done.usage.input_tokens == 110


async def test_budget_stop_is_a_warning(ctx: ToolContext, spans: Spans) -> None:
    from goalsmith.harness.budget import Budget

    model = traced_model([tool_call("read_file", {"path": "x"}, f"c{i}") for i in range(3)])
    loop = AgentLoop(model, ToolRegistry([read_file_tool()]), ctx, budget=Budget(max_steps=1))
    await loop.run("Loop")
    assert attr(spans.one("agent.run"), "langfuse.observation.level") == "WARNING"


async def test_content_capture_can_be_turned_off(ctx: ToolContext, spans: Spans) -> None:
    set_capture_content(False)
    model = traced_model([text("secret answer")])
    await AgentLoop(model, ToolRegistry(), ctx, permissions=AUTO_APPROVE).run("secret goal")
    for span in spans.all():
        assert attr(span, "langfuse.observation.input") is None
        assert attr(span, "langfuse.observation.output") is None


def test_trace_attributes_nest_and_tags_accumulate(spans: Spans) -> None:
    with (
        trace_attributes(name="eval", user_id="u1", tags=["eval"]),
        trace_attributes(session_id="s1", tags=["smoke"]),
        observation("work"),
    ):
        pass
    span = spans.one("work")
    assert attr(span, "langfuse.trace.name") == "eval"
    assert attr(span, "langfuse.user.id") == "u1"
    assert attr(span, "langfuse.session.id") == "s1"
    assert list(attr(span, "langfuse.trace.tags")) == ["eval", "smoke"]


# --- setup ----------------------------------------------------------------------------


def test_tracing_is_off_without_langfuse_keys() -> None:
    tracing = setup_tracing(Settings(_env_file=None))
    assert not tracing.enabled
    assert tracing.warning is None  # not configured: nothing to warn about


def test_unreachable_langfuse_turns_tracing_off_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOALSMITH_LANGFUSE__PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("GOALSMITH_LANGFUSE__SECRET_KEY", "sk-test")
    tracing = setup_tracing(Settings(_env_file=None), reachable=lambda host: False)
    assert not tracing.enabled
    assert tracing.warning is not None and "make obs-up" in tracing.warning


def test_reachability_check_handles_a_bad_url() -> None:
    assert not langfuse_reachable("not a url")


def test_langfuse_exporter_uses_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALSMITH_LANGFUSE__PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("GOALSMITH_LANGFUSE__SECRET_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert langfuse_exporter(settings)._endpoint == "http://localhost:3000" + OTLP_PATH
    headers = langfuse_headers(settings)
    assert headers["Authorization"] == "Basic cGstdGVzdDpzay10ZXN0"  # base64 of pk-test:sk-test
    assert headers["x-langfuse-ingestion-version"] == "4"
    assert trace_url(settings, "abc") == "http://localhost:3000/project/goalsmith/traces/abc"


# --- logs -----------------------------------------------------------------------------


@pytest.fixture
def json_logs(capsys: pytest.CaptureFixture[str]) -> Iterator[None]:
    configure_logging("INFO", "json")  # binds to pytest's captured stderr
    yield
    structlog.reset_defaults()  # later tests must not write to that closed stream


@pytest.mark.usefixtures("json_logs")
def test_logs_carry_run_and_trace_ids(capsys: pytest.CaptureFixture[str], spans: Spans) -> None:
    with bind_run("run-9"), observation("work"):
        get_logger("test").info("something.happened", n=1)
    get_logger("test").debug("hidden")

    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert line["event"] == "something.happened"
    assert line["run_id"] == "run-9"
    assert line["trace_id"] == format(spans.one("work").context.trace_id, "032x")
