"""Run budgets: hard limits that stop a run cleanly instead of looping forever.

M2 limits model calls ("steps"). M11 adds tokens, wall time and cost.
"""

from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """The limits for one run."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=15, ge=1, description="Maximum model calls in one run")

    def exhausted_by(self, usage: "BudgetUsage") -> bool:
        return usage.steps >= self.max_steps


class BudgetUsage(BaseModel):
    """What a run has used so far. Lives in the run state."""

    model_config = ConfigDict(frozen=True)

    steps: int = 0

    def add_step(self) -> "BudgetUsage":
        return self.model_copy(update={"steps": self.steps + 1})
