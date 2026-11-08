"""The only place that knows about model providers.

Everything else asks for a model by *role*:

    model = get_chat_model("planner")

Which provider and model serve that role is configuration (see config.py),
so switching providers is an env change, not a code change.
"""

from collections.abc import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from pydantic import BaseModel

from goalsmith.config import ModelSpec, Role, Settings, get_settings

ProviderBuilder = Callable[[ModelSpec, Settings], BaseChatModel]


class LLMConfigError(ValueError):
    """The model configuration is wrong or incomplete (bad provider, missing key)."""


def _build_anthropic(spec: ModelSpec, settings: Settings) -> BaseChatModel:
    if settings.anthropic_api_key is None:
        raise LLMConfigError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # model_validate takes the documented parameter names. Calling ChatAnthropic(...)
    # directly trips mypy on LangChain's field aliases.
    return ChatAnthropic.model_validate(
        {
            "model": spec.model,
            "api_key": settings.anthropic_api_key,
            "temperature": spec.temperature,
            "max_tokens": spec.max_tokens,
            "timeout": settings.llm_timeout_s,
            "max_retries": settings.llm_max_retries,
        }
    )


_PROVIDERS: dict[str, ProviderBuilder] = {"anthropic": _build_anthropic}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Add a provider (e.g. "openai") without touching the rest of the code."""
    _PROVIDERS[name] = builder


def known_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_chat_model(role: Role, settings: Settings | None = None) -> BaseChatModel:
    """Build the chat model configured for `role`."""
    settings = settings or get_settings()
    spec = settings.model_for(role)
    builder = _PROVIDERS.get(spec.provider)
    if builder is None:
        raise LLMConfigError(
            f"Unknown provider {spec.provider!r} for role {role!r}. "
            f"Known providers: {', '.join(known_providers())}"
        )
    return builder(spec, settings)


async def generate_structured[SchemaT: BaseModel](
    model: BaseChatModel, schema: type[SchemaT], messages: LanguageModelInput
) -> SchemaT:
    """Ask `model` for an answer shaped like `schema`, validated by Pydantic.

    Raises pydantic.ValidationError (a ValueError) if the model's output does not fit.
    """
    result = await model.with_structured_output(schema).ainvoke(messages)
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)
