import pytest

from goalsmith.config import ROLES, Settings


def make_settings() -> Settings:
    # _env_file=None: ignore any real .env file on the developer's machine.
    return Settings(_env_file=None)


def test_every_role_has_a_default_model() -> None:
    settings = make_settings()
    for role in ROLES:
        spec = settings.model_for(role)
        assert spec.provider == "anthropic"
        assert spec.model.startswith("claude-")


def test_model_can_be_changed_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALSMITH_WORKER__MODEL", "claude-haiku-5-5")
    assert make_settings().model_for("worker").model == "claude-haiku-5-5"


def test_partial_env_override_keeps_other_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    default = make_settings().model_for("router")
    monkeypatch.setenv("GOALSMITH_ROUTER__MAX_TOKENS", "256")
    router = make_settings().model_for("router")
    assert router.max_tokens == 256
    assert router.model == default.model
    assert router.provider == default.provider


def test_timeout_and_retries_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOALSMITH_LLM_TIMEOUT_S", "12.5")
    monkeypatch.setenv("GOALSMITH_LLM_MAX_RETRIES", "5")
    settings = make_settings()
    assert settings.llm_timeout_s == 12.5
    assert settings.llm_max_retries == 5


def test_api_key_read_from_standard_name_and_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    settings = make_settings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)


def test_api_key_is_optional_until_used() -> None:
    assert make_settings().anthropic_api_key is None
