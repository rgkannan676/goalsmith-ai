"""A scripted fake chat model for unit tests.

It returns pre-written AIMessages in order, so tests control exactly what the
"model" says (plain text, tool calls, or structured output) with no network.
"""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class FakeChatModel(BaseChatModel):
    """Replays `responses` one per call and records what it was sent."""

    responses: list[AIMessage]
    model_name: str = "fake-model"  # reported to callbacks, like a real model's name
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if self._index >= len(self.responses):
            raise IndexError(
                f"FakeChatModel ran out of scripted responses after {len(self.responses)} calls"
            )
        message = self.responses[self._index].model_copy()
        self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # Real models send tool schemas to the provider; the fake just remembers them.
        self.bound_tools = list(tools)
        return self


def text(content: str) -> AIMessage:
    """A plain text reply."""
    return AIMessage(content=content)


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    """A reply in which the model asks to call one tool."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id or f"call_{name}", "type": "tool_call"}
        ],
    )


def with_usage(
    message: AIMessage,
    input_tokens: int,
    output_tokens: int,
    *,
    model: str = "fake-model",
    cache_read: int = 0,
) -> AIMessage:
    """`message` with token counts and a model name, as a real provider reports them."""
    return message.model_copy(
        update={
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_token_details": {"cache_read": cache_read},
            },
            "response_metadata": {"model_name": model},
        }
    )


def structured(obj: BaseModel) -> AIMessage:
    """A reply that `with_structured_output(type(obj))` will parse back into `obj`."""
    return tool_call(type(obj).__name__, obj.model_dump())
