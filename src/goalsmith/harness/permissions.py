"""Human approval for risky tool calls.

Before a `write` or `external` tool runs, the run pauses (LangGraph `interrupt()`),
its state is saved by the checkpointer, and a human decides for each call:

- approve: run it as the model asked;
- edit: run it with arguments the human changed;
- reject: do not run it; the model is told why.

Everything here is a pure function over messages, so it is easy to test. The
pausing itself happens in the approval node of the agent loop.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from goalsmith.harness.tools import Risk, ToolRegistry

Action = Literal["approve", "edit", "reject"]


class Permissions(BaseModel):
    """Which risk levels pause for a human before running."""

    model_config = ConfigDict(frozen=True)

    needs_approval: frozenset[Risk] = frozenset({"write", "external"})


AUTO_APPROVE = Permissions(needs_approval=frozenset())
"""Never ask. For tests and trusted, non-interactive runs."""


class ApprovalRequest(BaseModel):
    """One tool call waiting for a decision. Shown to the human."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    risk: Risk
    args: dict[str, Any]
    description: str


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Action
    args: dict[str, Any] | None = None  # the new arguments, for "edit"
    reason: str = ""  # optional note, passed to the model on "reject"

    @model_validator(mode="after")
    def _edit_needs_args(self) -> Self:
        if self.action == "edit" and self.args is None:
            raise ValueError("an 'edit' decision needs the new args")
        return self


def approval_requests(
    calls: Sequence[ToolCall], registry: ToolRegistry, permissions: Permissions
) -> list[ApprovalRequest]:
    """The calls that need a human. Unknown tools are skipped: the registry
    turns them into an error message anyway, so there is nothing to approve."""
    requests = []
    for call in calls:
        if call["name"] not in registry.names():
            continue
        tool = registry.get(call["name"])
        if tool.risk in permissions.needs_approval:
            requests.append(
                ApprovalRequest(
                    call_id=call.get("id") or "",
                    tool=tool.name,
                    risk=tool.risk,
                    args=dict(call["args"]),
                    description=tool.description,
                )
            )
    return requests


def missing_decisions(
    requests: Sequence[ApprovalRequest], decisions: Mapping[str, Decision]
) -> list[str]:
    return [r.call_id for r in requests if r.call_id not in decisions]


def apply_decisions(
    message: AIMessage, decisions: Mapping[str, Decision]
) -> tuple[AIMessage, list[ToolMessage]]:
    """Turn decisions into state updates.

    Returns the model's message with edited arguments (same id, so LangGraph's
    `add_messages` replaces it in place), plus a ToolMessage for every rejected
    call. A rejected call keeps its entry in `tool_calls`, because every tool call
    must be answered by a ToolMessage; the rejection *is* that answer, so the tools
    node never runs it.
    """
    calls: list[ToolCall] = []
    rejections: list[ToolMessage] = []
    for call in message.tool_calls:
        decision = decisions.get(call.get("id") or "")
        if decision is not None and decision.action == "edit" and decision.args is not None:
            call = ToolCall(
                name=call["name"], args=decision.args, id=call.get("id"), type="tool_call"
            )
        elif decision is not None and decision.action == "reject":
            rejections.append(_rejection(call, decision.reason))
        calls.append(call)
    edited = message.model_copy(update={"tool_calls": calls})
    return edited, rejections


def _rejection(call: ToolCall, reason: str) -> ToolMessage:
    note = f" Reason: {reason}" if reason else ""
    return ToolMessage(
        content=(
            f"Rejected by the user; {call['name']!r} was not run.{note} "
            "Do not retry the same call. Change your approach or explain what you need."
        ),
        tool_call_id=call.get("id") or "",
        name=call["name"],
        status="error",
    )


def pending_requests(interrupt_value: Any) -> list[ApprovalRequest]:
    """Read the requests back from a saved interrupt (they are stored as plain dicts)."""
    items: list[Any] = (
        interrupt_value.get("requests", []) if isinstance(interrupt_value, dict) else []
    )
    return [ApprovalRequest.model_validate(item) for item in items]


def interrupt_payload(requests: Sequence[ApprovalRequest]) -> dict[str, Any]:
    """What the approval node passes to `interrupt()`: plain JSON, safe to checkpoint."""
    return {"type": "approval", "requests": [r.model_dump(mode="json") for r in requests]}


class _DecisionMap(BaseModel):
    decisions: dict[str, Decision] = Field(default_factory=dict)


def parse_decisions(raw: Any) -> dict[str, Decision]:
    """Validate the resume value, which arrives as plain dicts after a checkpoint."""
    return _DecisionMap.model_validate({"decisions": raw}).decisions
