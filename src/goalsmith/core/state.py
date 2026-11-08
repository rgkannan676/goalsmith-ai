"""The state that flows through the run graph.

LangGraph passes a `RunState` to each node. A node returns only the fields it
changes (a partial update), and LangGraph merges that into the state.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from goalsmith.harness.budget import BudgetUsage

RunStatus = Literal["running", "completed", "budget_exceeded"]


class RunState(TypedDict):
    goal: str
    # `add_messages` is a reducer: new messages are appended, not replaced.
    messages: Annotated[list[AnyMessage], add_messages]
    budget: BudgetUsage
    status: RunStatus
    answer: str | None


class RunResult(BaseModel):
    """What a finished run hands back to its caller (CLI, API, tests)."""

    run_id: str
    status: RunStatus
    answer: str
    steps: int
