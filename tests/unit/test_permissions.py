from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolCall
from pydantic import BaseModel, ValidationError

from goalsmith.harness.permissions import (
    AUTO_APPROVE,
    Decision,
    Permissions,
    apply_decisions,
    approval_requests,
    interrupt_payload,
    missing_decisions,
    parse_decisions,
    pending_requests,
)
from goalsmith.harness.tools import Risk, Tool, ToolContext, ToolRegistry


class _Args(BaseModel):
    path: str = "x"


def make_tool(name: str, risk: Risk) -> Tool:
    async def fn(args: _Args, ctx: ToolContext) -> str:
        return "ok"

    return Tool(name=name, description=f"{name} tool", args_schema=_Args, risk=risk, fn=fn)


REGISTRY = ToolRegistry(
    [
        make_tool("read_file", "read"),
        make_tool("write_file", "write"),
        make_tool("email", "external"),
    ]
)


def call(name: str, call_id: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args, id=call_id, type="tool_call")


def test_only_write_and_external_need_approval_by_default() -> None:
    calls = [call("read_file", "1"), call("write_file", "2"), call("email", "3"), call("nope", "4")]
    requests = approval_requests(calls, REGISTRY, Permissions())
    assert [(r.call_id, r.tool, r.risk) for r in requests] == [
        ("2", "write_file", "write"),
        ("3", "email", "external"),
    ]
    assert requests[0].description == "write_file tool"


def test_policy_is_configurable() -> None:
    calls = [call("write_file", "1"), call("email", "2")]
    only_external = Permissions(needs_approval=frozenset({"external"}))
    assert [r.tool for r in approval_requests(calls, REGISTRY, only_external)] == ["email"]
    assert approval_requests(calls, REGISTRY, AUTO_APPROVE) == []


def test_edit_decision_needs_args() -> None:
    with pytest.raises(ValidationError, match="needs the new args"):
        Decision(action="edit")
    assert Decision(action="edit", args={"path": "y"}).args == {"path": "y"}


def test_apply_decisions() -> None:
    message = AIMessage(
        content="",
        id="m1",
        tool_calls=[
            call("write_file", "a", path="keep.md"),
            call("write_file", "b", path="old.md"),
            call("write_file", "c", path="bad.md"),
        ],
    )
    decisions = {
        "a": Decision(action="approve"),
        "b": Decision(action="edit", args={"path": "new.md"}),
        "c": Decision(action="reject", reason="wrong folder"),
    }

    edited, rejections = apply_decisions(message, decisions)

    assert edited.id == "m1"  # same id: add_messages replaces the message in place
    assert [c["args"]["path"] for c in edited.tool_calls] == ["keep.md", "new.md", "bad.md"]
    assert len(rejections) == 1
    rejected = rejections[0]
    assert rejected.tool_call_id == "c"
    assert rejected.status == "error"
    assert "wrong folder" in str(rejected.content)
    assert "Do not retry" in str(rejected.content)


def test_payload_round_trip_and_missing_decisions() -> None:
    requests = approval_requests([call("write_file", "1", path="a")], REGISTRY, Permissions())
    payload = interrupt_payload(requests)
    assert pending_requests(payload) == requests
    assert pending_requests("not a payload") == []

    assert missing_decisions(requests, {}) == ["1"]
    assert missing_decisions(requests, {"1": Decision(action="approve")}) == []


def test_parse_decisions_from_plain_dicts() -> None:
    parsed = parse_decisions({"1": {"action": "reject", "reason": "no"}})
    assert parsed == {"1": Decision(action="reject", reason="no")}
    with pytest.raises(ValidationError):
        parse_decisions({"1": {"action": "maybe"}})
