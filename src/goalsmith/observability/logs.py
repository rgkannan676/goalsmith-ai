"""Structured logs (structlog) that carry the run id and the trace id.

Log calls look like `log.info("tool.finished", tool="read_file", ms=12)`: an event
name plus key-value fields, never a formatted sentence. Each line automatically gets:
- every value bound with `bind_run(run_id)` (via structlog's contextvars), and
- `trace_id` / `span_id` of the current OpenTelemetry span, so a log line can be
  matched to its trace in Langfuse, and the other way round.

Logs go to stderr. `console` is readable for local dev; `json` is one object per
line for log collectors.
"""

import logging
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Literal

import structlog
from opentelemetry import trace

LogFormat = Literal["console", "json"]


def configure_logging(level: str = "WARNING", fmt: LogFormat = "console") -> None:
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_trace_ids,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )


def _stderr_logger(*_args: Any) -> structlog.PrintLogger:
    # Look up sys.stderr on each call, not once at configure time, so logs follow
    # it when it is replaced (e.g. by pytest's output capture).
    return structlog.PrintLogger(file=sys.stderr)


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger(component=name)
    return logger


def add_trace_ids(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


@contextmanager
def bind_run(run_id: str) -> Iterator[None]:
    """Add `run_id` to every log line inside the block."""
    with structlog.contextvars.bound_contextvars(run_id=run_id):
        yield
