"""Collect spans in memory so tests can check what would be sent to Langfuse.

OpenTelemetry allows one global tracer provider per process, so it is installed
once, when this module is first imported; the `spans` fixture clears it per test.
"""

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from goalsmith.observability.spans import set_capture_content
from goalsmith.observability.tracing import build_provider

_exporter = InMemorySpanExporter()
trace.set_tracer_provider(build_provider(_exporter, batch=False))


class Spans:
    def all(self) -> list[ReadableSpan]:
        return list(_exporter.get_finished_spans())

    def named(self, name: str) -> list[ReadableSpan]:
        return [s for s in self.all() if s.name == name]

    def one(self, name: str) -> ReadableSpan:
        found = self.named(name)
        assert len(found) == 1, f"expected one span {name!r}, got {[s.name for s in self.all()]}"
        return found[0]

    def of_type(self, kind: str) -> list[ReadableSpan]:
        return [s for s in self.all() if attr(s, "langfuse.observation.type") == kind]


def attr(span: ReadableSpan, key: str) -> Any:
    return (span.attributes or {}).get(key)


@pytest.fixture
def spans() -> Iterator[Spans]:
    _exporter.clear()
    set_capture_content(True)
    yield Spans()
    set_capture_content(True)
