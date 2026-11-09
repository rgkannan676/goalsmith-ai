"""Sending traces to Langfuse, and tracing every model call.

- `setup_tracing(settings)` installs an OpenTelemetry tracer provider that exports
  spans over OTLP/HTTP to Langfuse. Call it once at program start and `shutdown()`
  the result at the end: spans are sent in batches, and a short CLI run would
  otherwise exit before the last batch is sent.
- `ModelCallTracer` is a LangChain callback handler. `llm.get_chat_model` attaches
  it to every model it builds, so each model call becomes a "generation" span with
  the model name, messages, tokens and cost, without any tracing code in the nodes.
"""

import base64
import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span

from goalsmith import __version__
from goalsmith.config import ModelPrice, Settings
from goalsmith.observability.spans import (
    TraceAttributesProcessor,
    mark_error,
    set_capture_content,
    set_input,
    set_metadata,
    set_output,
    tracer,
)
from goalsmith.observability.usage import TokenUsage, model_name, usage_from_message

OTLP_PATH = "/api/public/otel/v1/traces"
_ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


@dataclass
class Tracing:
    """Handle returned by `setup_tracing`. Call `shutdown()` before exiting."""

    provider: TracerProvider | None
    warning: str | None = None  # why tracing is off although it was configured

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    def shutdown(self) -> None:
        if self.provider is not None:
            self.provider.shutdown()  # flushes spans still waiting in the batch


def setup_tracing(settings: Settings, reachable: Callable[[str], bool] | None = None) -> Tracing:
    """Export traces to Langfuse when its keys are configured; otherwise do nothing.

    Installs the process-wide tracer provider, which OpenTelemetry allows only once.
    If Langfuse does not answer, tracing stays off for this process, rather than
    every span batch failing and retrying (which slows down exit).
    """
    set_capture_content(settings.trace_content)
    if not settings.langfuse.enabled:
        return Tracing(provider=None)
    host = settings.langfuse.host
    if not (reachable or langfuse_reachable)(host):
        return Tracing(
            provider=None,
            warning=f"Tracing is off: Langfuse is not reachable at {host} (`make obs-up`).",
        )
    provider = build_provider(langfuse_exporter(settings))
    trace.set_tracer_provider(provider)
    return Tracing(provider=provider)


def langfuse_reachable(host: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/public/health", timeout=timeout_s):
            return True
    except (OSError, ValueError):  # connection refused, timeout, HTTP error, bad URL
        return False


def build_provider(exporter: SpanExporter, *, batch: bool = True) -> TracerProvider:
    """A tracer provider that sends finished spans to `exporter`.

    Batching sends spans in the background; tests use `batch=False` to see each
    span as soon as it ends.
    """
    provider = TracerProvider(
        resource=Resource.create({"service.name": "goalsmith", "service.version": __version__})
    )
    provider.add_span_processor(TraceAttributesProcessor())
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider


def langfuse_exporter(settings: Settings) -> OTLPSpanExporter:
    return OTLPSpanExporter(
        endpoint=settings.langfuse.host.rstrip("/") + OTLP_PATH,
        headers=langfuse_headers(settings),
        timeout=5,
    )


def langfuse_headers(settings: Settings) -> dict[str, str]:
    """HTTP basic auth with the project's public and secret key."""
    lf = settings.langfuse
    if not (lf.public_key and lf.secret_key):
        raise ValueError("Langfuse keys are not set (GOALSMITH_LANGFUSE__PUBLIC_KEY/SECRET_KEY)")
    pair = f"{lf.public_key}:{lf.secret_key.get_secret_value()}"
    return {
        "Authorization": "Basic " + base64.b64encode(pair.encode()).decode(),
        # Ingest in real time instead of after a delay of up to minutes.
        "x-langfuse-ingestion-version": "4",
    }


def trace_url(settings: Settings, trace_id: str) -> str:
    lf = settings.langfuse
    return f"{lf.host.rstrip('/')}/project/{lf.project_id}/traces/{trace_id}"


def session_url(settings: Settings, session_id: str) -> str:
    lf = settings.langfuse
    return f"{lf.host.rstrip('/')}/project/{lf.project_id}/sessions/{session_id}"


class ModelCallTracer(BaseCallbackHandler):
    """Turns each chat model call into a Langfuse "generation" span.

    LangChain calls `on_chat_model_start` and then `on_llm_end` (or `on_llm_error`)
    for every model call, keyed by a run id. The span's parent is whatever span is
    current when the call starts, e.g. the agent run.
    """

    run_inline = True  # call us directly, in the caller's context (keeps the parent span)

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = prices
        self._spans: dict[UUID, tuple[Span, str | None]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params: dict[str, Any] = kwargs.get("invocation_params") or {}
        model = (
            params.get("model") or params.get("model_name") or (metadata or {}).get("ls_model_name")
        )
        span = tracer().start_span(
            f"model {model or 'call'}",
            attributes={"langfuse.observation.type": "generation"},
        )
        if model:
            span.set_attribute("langfuse.observation.model.name", str(model))
            span.set_attribute("gen_ai.request.model", str(model))
        set_input(span, [_message_dict(m) for m in messages[0]] if messages else [])
        # Keep the parts of the LangChain/LangGraph metadata worth filtering on.
        keep = {
            k: v
            for k, v in (metadata or {}).items()
            if k.startswith("prompt_") or k in ("langgraph_node", "langgraph_step")
        }
        set_metadata(span, **keep)
        self._spans[run_id] = (span, str(model) if model else None)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, model = entry
        message = _first_message(response)
        if message is not None:
            record_usage(span, usage_from_message(message, self._prices, model), message)
            set_output(span, _message_dict(message))
        span.end()

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        entry = self._spans.pop(run_id, None)
        if entry is None:
            return
        span, _ = entry
        span.record_exception(error)
        mark_error(span, f"{type(error).__name__}: {error}")
        span.end()


def record_usage(span: Span, usage: TokenUsage, message: AIMessage | None = None) -> None:
    """Tokens and cost in Langfuse's format, plus the standard gen_ai.* names."""
    uncached = max(usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens, 0)
    details = {
        "input": uncached,
        "output": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
        "cache_creation_input_tokens": usage.cache_write_tokens,
    }
    span.set_attribute("langfuse.observation.usage_details", json.dumps(details))
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    if not usage.unpriced_calls:
        cost = json.dumps({"total": usage.cost_usd})
        span.set_attribute("langfuse.observation.cost_details", cost)
    if message is not None and (name := model_name(message)):
        span.set_attribute("gen_ai.response.model", name)


def _first_message(response: LLMResult) -> AIMessage | None:
    for generations in response.generations:
        for generation in generations:
            if isinstance(generation, ChatGeneration) and isinstance(generation.message, AIMessage):
                return generation.message
    return None


def _message_dict(message: BaseMessage) -> dict[str, Any]:
    """A message in the chat format Langfuse renders: role, content, tool calls."""
    data: dict[str, Any] = {"role": _ROLES.get(message.type, message.type)}
    data["content"] = message.content
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        data["tool_call_id"] = tool_call_id
    return data
