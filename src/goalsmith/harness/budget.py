"""Run budgets: hard limits that stop a run cleanly instead of looping forever.

M2 limits model calls ("steps"). M11 adds tokens, wall time and cost.
"""

from pydantic import BaseModel, ConfigDict, Field

from goalsmith.observability.usage import TokenUsage


class Budget(BaseModel):
    """The limits for one run."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=15, ge=1, description="Maximum model calls in one run")

    def exhausted_by(self, usage: "BudgetUsage") -> bool:
        return usage.steps >= self.max_steps


class BudgetUsage(BaseModel):
    """What a run has used so far. Lives in the run state, so it survives a pause
    and resume (the checkpoint holds it)."""

    model_config = ConfigDict(frozen=True)

    steps: int = 0
    tokens: TokenUsage = TokenUsage()

    def add_step(self, tokens: TokenUsage | None = None) -> "BudgetUsage":
        return self.model_copy(
            update={"steps": self.steps + 1, "tokens": self.tokens + (tokens or TokenUsage())}
        )
