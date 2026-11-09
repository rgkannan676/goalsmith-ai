"""How run state is serialized into checkpoints.

LangGraph stores state with msgpack. Loading a *class* back from stored bytes is a
security risk (a tampered checkpoint could name any class), so LangGraph only
rebuilds classes that are explicitly allowed. Every custom type kept in RunState
must be listed here.
"""

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

CHECKPOINT_TYPES: list[tuple[str, str]] = [
    ("goalsmith.harness.budget", "BudgetUsage"),
    ("goalsmith.observability.usage", "TokenUsage"),
]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
