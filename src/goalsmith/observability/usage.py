"""Token counts and cost for model calls.

LangChain reports tokens on each reply as `AIMessage.usage_metadata`:
- `input_tokens` is the *total* input, including tokens read from or written to
  the prompt cache;
- `input_token_details` splits out `cache_read` and the cache writes.
Cache reads cost about a tenth of normal input, so they are priced separately.
"""

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from goalsmith.config import ModelPrice

_CACHE_WRITE_KEYS = ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")


class TokenUsage(BaseModel):
    """Tokens and cost, for one model call or summed over a run."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0  # all input, cached or not
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0  # calls to a model with no price: their cost is missing

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            unpriced_calls=self.unpriced_calls + other.unpriced_calls,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        """e.g. `12,345 in / 678 out tokens · $0.0381`."""
        cost = f"${self.cost_usd:.4f}"
        if self.unpriced_calls:
            cost += f" (+{self.unpriced_calls} unpriced call(s))"
        return f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens · {cost}"


def model_name(message: AIMessage) -> str | None:
    name = message.response_metadata.get("model_name") or message.response_metadata.get("model")
    return str(name) if name else None


def usage_from_message(
    message: AIMessage, prices: Mapping[str, ModelPrice], model: str | None = None
) -> TokenUsage:
    """Tokens and cost of the call that produced `message`. Zero if not reported."""
    meta = message.usage_metadata
    if meta is None:
        return TokenUsage()
    details: Mapping[str, Any] = meta.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = sum(int(details.get(key) or 0) for key in _CACHE_WRITE_KEYS)
    usage = TokenUsage(
        input_tokens=meta["input_tokens"],
        output_tokens=meta["output_tokens"],
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    price = prices.get(model or model_name(message) or "")
    if price is None:
        return usage.model_copy(update={"unpriced_calls": 1})
    return usage.model_copy(update={"cost_usd": cost_usd(usage, price)})


def cost_usd(usage: TokenUsage, price: ModelPrice) -> float:
    uncached = max(usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens, 0)
    return (
        uncached * price.input
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
        + usage.output_tokens * price.output
    ) / 1_000_000
