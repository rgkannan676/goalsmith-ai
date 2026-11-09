"""The only place that knows about model providers.

Everything else asks for a model by *role*:

    model = get_chat_model("planner")

Which provider and model serve that role is configuration (see config.py),
so switching providers is an env change, not a code change.

Embeddings and the reranker for document search are local models run by
fastembed (ONNX on the CPU): no API key, no cost, and they work offline once
downloaded.
"""

import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from pydantic import BaseModel

from goalsmith.capabilities.rag.rerank import Reranker
from goalsmith.config import ModelSpec, Role, Settings, get_settings
from goalsmith.observability.tracing import ModelCallTracer

if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

ProviderBuilder = Callable[[ModelSpec, Settings], BaseChatModel]


class LLMConfigError(ValueError):
    """The model configuration is wrong or incomplete (bad provider, missing key)."""


def _build_anthropic(spec: ModelSpec, settings: Settings) -> BaseChatModel:
    if settings.anthropic_api_key is None:
        raise LLMConfigError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    # The Anthropic SDK traces its own requests when a tracer provider is installed.
    # ModelCallTracer already records every model call (for any provider), so the
    # SDK's spans would count each call twice. Set the variable to "true" to keep both.
    os.environ.setdefault("ANTHROPIC_OPEN_TELEMETRY", "false")
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
    """Build the chat model configured for `role`.

    Every call it makes is traced (model, messages, tokens, cost) by ModelCallTracer.
    """
    settings = settings or get_settings()
    spec = settings.model_for(role)
    builder = _PROVIDERS.get(spec.provider)
    if builder is None:
        raise LLMConfigError(
            f"Unknown provider {spec.provider!r} for role {role!r}. "
            f"Known providers: {', '.join(known_providers())}"
        )
    model = builder(spec, settings)
    model.callbacks = [ModelCallTracer(settings.prices)]
    return model


async def generate_structured[SchemaT: BaseModel](
    model: BaseChatModel, schema: type[SchemaT], messages: LanguageModelInput
) -> SchemaT:
    """Ask `model` for an answer shaped like `schema`, validated by Pydantic.

    Raises pydantic.ValidationError (a ValueError) if the model's output does not fit.
    """
    result = await model.with_structured_output(schema, **_structured_kwargs(model)).ainvoke(
        messages
    )
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _structured_kwargs(model: BaseChatModel) -> dict[str, Any]:
    """Claude has native structured output (a JSON schema the reply must match).
    LangChain's default, forcing a tool call, is rejected by current Claude models."""
    if isinstance(model, ChatAnthropic):
        return {"method": "json_schema"}
    return {}


# --- local models for document search -----------------------------------------------


class FastEmbedEmbeddings(Embeddings):
    """LangChain's Embeddings interface over a fastembed model.

    The model (tens of MB) is downloaded and loaded on first use, not at startup,
    so commands that never search stay fast.
    """

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._lock = threading.Lock()  # LangChain's async methods call us from threads

    def _load(self) -> "TextEmbedding":
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name, cache_dir=str(self._cache_dir))
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [cast(list[float], v.tolist()) for v in self._load().passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect a short instruction before queries; query_embed adds it.
        vector = next(iter(self._load().query_embed(text)))
        return cast(list[float], vector.tolist())


class FastEmbedReranker:
    """A cross-encoder reranker run by fastembed. Loaded on first use."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self.model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None
        self._lock = threading.Lock()

    def _load(self) -> "TextCrossEncoder":
        with self._lock:
            if self._model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._model = TextCrossEncoder(self.model_name, cache_dir=str(self._cache_dir))
            return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._load().rerank(query, documents)]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    return FastEmbedEmbeddings(settings.rag.embedding_model, settings.rag.model_cache_dir)


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    """The configured reranker, or None when `GOALSMITH_RAG__RERANK_MODEL` is empty."""
    settings = settings or get_settings()
    if not settings.rag.rerank_model:
        return None
    return FastEmbedReranker(settings.rag.rerank_model, settings.rag.model_cache_dir)
