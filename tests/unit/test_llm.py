import socket

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from goalsmith import llm
from goalsmith.config import ModelSpec, Settings
from tests.fixtures.fake_llm import FakeChatModel, structured, text, tool_call


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


# --- get_chat_model ---------------------------------------------------------


def test_anthropic_model_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-test",
        GOALSMITH_LLM_TIMEOUT_S="30",
        GOALSMITH_LLM_MAX_RETRIES="4",
    )
    model = llm.get_chat_model("worker", settings)
    assert isinstance(model, ChatAnthropic)
    assert model.model == settings.worker.model
    assert model.default_request_timeout == 30
    assert model.max_retries == 4
    assert model.max_tokens == settings.worker.max_tokens


def test_each_role_gets_its_own_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(
        monkeypatch,
        ANTHROPIC_API_KEY="sk-test",
        GOALSMITH_PLANNER__MODEL="model-a",
        GOALSMITH_ROUTER__MODEL="model-b",
    )
    planner = llm.get_chat_model("planner", settings)
    router = llm.get_chat_model("router", settings)
    assert isinstance(planner, ChatAnthropic)
    assert isinstance(router, ChatAnthropic)
    assert (planner.model, router.model) == ("model-a", "model-b")


def test_missing_api_key_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(monkeypatch)
    with pytest.raises(llm.LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm.get_chat_model("worker", settings)


def test_unknown_provider_lists_known_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings(monkeypatch, GOALSMITH_JUDGE__PROVIDER="nope")
    with pytest.raises(llm.LLMConfigError, match=r"nope.*anthropic"):
        llm.get_chat_model("judge", settings)


def test_switching_provider_is_an_env_change(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeChatModel(responses=[text("hi")])

    def build_fake(spec: ModelSpec, settings: Settings) -> BaseChatModel:
        return fake

    monkeypatch.setitem(llm._PROVIDERS, "fake", build_fake)
    settings = make_settings(monkeypatch, GOALSMITH_WORKER__PROVIDER="fake")
    assert llm.get_chat_model("worker", settings) is fake


def test_register_provider_adds_a_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_PROVIDERS", dict(llm._PROVIDERS))
    llm.register_provider("other", lambda spec, settings: FakeChatModel(responses=[]))
    assert "other" in llm.known_providers()


# --- structured output ------------------------------------------------------


class Verdict(BaseModel):
    score: int
    reason: str


async def test_generate_structured_returns_schema_instance() -> None:
    expected = Verdict(score=4, reason="cites sources")
    fake = FakeChatModel(responses=[structured(expected)])
    result = await llm.generate_structured(fake, Verdict, "Grade this draft")
    assert result == expected
    assert fake.bound_tools == [Verdict]


async def test_generate_structured_rejects_bad_output() -> None:
    fake = FakeChatModel(responses=[tool_call("Verdict", {"score": "not a number"})])
    with pytest.raises(ValueError):
        await llm.generate_structured(fake, Verdict, "Grade this draft")


# --- fake model and test isolation -----------------------------------------


def test_fake_model_replays_in_order_and_records_calls() -> None:
    fake = FakeChatModel(responses=[text("one"), tool_call("web_search", {"query": "x"})])
    first = fake.invoke("a")
    second = fake.invoke("b")
    assert first.content == "one"
    assert second.tool_calls[0]["name"] == "web_search"
    assert second.tool_calls[0]["args"] == {"query": "x"}
    assert [c[0].content for c in fake.calls] == ["a", "b"]


def test_fake_model_fails_loudly_when_script_runs_out() -> None:
    fake = FakeChatModel(responses=[])
    with pytest.raises(IndexError, match="ran out"):
        fake.invoke("hello")


def test_unit_tests_cannot_reach_the_network() -> None:
    with pytest.raises(RuntimeError, match="Network access"):
        socket.create_connection(("example.com", 443))
