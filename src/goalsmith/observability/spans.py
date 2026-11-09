"""OpenTelemetry spans with the attributes Langfuse understands.

We use plain OpenTelemetry, not a vendor SDK: any OTLP backend can receive the
traces. Langfuse reads a few extra attributes (`langfuse.*`, `gen_ai.*`) to show
spans as model calls, tool calls and events, with input, output, tokens and cost.

One *trace* is one tree of spans: e.g. a CLI command, with the agent run, each
model call and each tool call nested inside it. Trace-level fields (session = run
id, user, tags, metadata) are copied onto every span by `TraceAttributesProcessor`,
because Langfuse filters on them most reliably that way.

Without a configured tracer provider (see tracing.py) every call here is a cheap
no-op, so code can always create spans.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, Literal

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import Span as ApiSpan
from opentelemetry.trace import Status, StatusCode
from opentelemetry.util.types import AttributeValue

TRACER_NAME = "goalsmith"
MAX_CONTENT_CHARS = 20_000

ObservationType = Literal["span", "generation", "tool", "agent", "event", "evaluator"]

_EMPTY: Mapping[str, AttributeValue] = MappingProxyType({})
_trace_attrs: ContextVar[Mapping[str, AttributeValue]] = ContextVar("trace_attrs", default=_EMPTY)
_capture_content = True


def set_capture_content(enabled: bool) -> None:
    """Whether span input/output (prompts, tool arguments, results) is recorded."""
    global _capture_content
    _capture_content = enabled


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


# --- trace-level attributes -----------------------------------------------------------


@contextmanager
def trace_attributes(
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Attach trace-level fields to every span started inside this block.

    Nested blocks add to (and may override) the outer ones, except `name`: the
    outermost name wins, because it names the whole trace. Tags accumulate.
    """
    attrs: dict[str, AttributeValue] = dict(_trace_attrs.get())
    if name:
        attrs.setdefault("langfuse.trace.name", name)
    if session_id:
        attrs["langfuse.session.id"] = session_id
    if user_id:
        attrs["langfuse.user.id"] = user_id
    if tags:
        old = attrs.get("langfuse.trace.tags", ())
        existing = list(old) if isinstance(old, Sequence) and not isinstance(old, str) else []
        attrs["langfuse.trace.tags"] = [str(t) for t in dict.fromkeys([*existing, *tags])]
    for key, value in (metadata or {}).items():
        attrs[f"langfuse.trace.metadata.{key}"] = value
    token = _trace_attrs.set(attrs)
    try:
        yield
    finally:
        _trace_attrs.reset(token)


class TraceAttributesProcessor(SpanProcessor):
    """Copies the current `trace_attributes` onto each span when it starts."""

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        for key, value in _trace_attrs.get().items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


# --- observations ---------------------------------------------------------------------


@contextmanager
def observation(
    name: str,
    kind: ObservationType = "span",
    *,
    input: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[ApiSpan]:
    """A span that becomes the current parent for everything inside the block.

    An exception leaving the block is recorded on the span and marks it as an error.
    """
    with tracer().start_as_current_span(name, attributes=_base_attrs(kind, metadata)) as span:
        if input is not None:
            set_input(span, input)
        yield span


def event(name: str, *, input: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
    """A point-in-time observation, e.g. "approval requested"."""
    span = tracer().start_span(name, attributes=_base_attrs("event", metadata))
    if input is not None:
        set_input(span, input)
    span.end()


def _base_attrs(kind: ObservationType, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    attrs: dict[str, Any] = {"langfuse.observation.type": kind}
    for key, value in (metadata or {}).items():
        attrs[f"langfuse.observation.metadata.{key}"] = _attr(value)
    return attrs


def set_input(span: ApiSpan, value: Any) -> None:
    if _capture_content:
        span.set_attribute("langfuse.observation.input", _to_json(value))


def set_output(span: ApiSpan, value: Any) -> None:
    if _capture_content:
        span.set_attribute("langfuse.observation.output", _to_json(value))


def set_metadata(span: ApiSpan, **values: Any) -> None:
    for key, value in values.items():
        span.set_attribute(f"langfuse.observation.metadata.{key}", _attr(value))


def mark_error(span: ApiSpan, message: str) -> None:
    """An expected failure (e.g. a tool error): shown as an error, nothing raised."""
    span.set_status(Status(StatusCode.ERROR, message))
    span.set_attribute("langfuse.observation.level", "ERROR")
    span.set_attribute("langfuse.observation.status_message", message[:500])


def mark_warning(span: ApiSpan, message: str) -> None:
    span.set_attribute("langfuse.observation.level", "WARNING")
    span.set_attribute("langfuse.observation.status_message", message[:500])


def current_trace_id() -> str | None:
    """The current trace id as 32 hex characters, or None outside a recorded trace."""
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _to_json(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS] + f"...[truncated {len(text) - MAX_CONTENT_CHARS}]"
    return text


def _attr(value: Any) -> AttributeValue:
    if isinstance(value, str | bool | int | float):
        return value
    return _to_json(value)
