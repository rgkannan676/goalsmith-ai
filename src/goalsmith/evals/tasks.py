"""Golden tasks and rubrics: what the agent eval runs and how it is graded.

A golden task is a goal with a known good outcome. It is graded two ways:
1. **Checks**: cheap, deterministic assertions (the answer contains "18", the
   file was written, `search_docs` was called, nothing was saved to memory...).
2. **Judge criteria**: things only a reader can decide ("every claim is supported
   by a tool result"). An LLM judge grades each criterion pass/fail.

Tasks live in evals/tasks/*.yaml (a list of tasks per file) and rubrics, which are
shared lists of criteria, in evals/rubrics/*.yaml.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

EVALS_DIR = Path("evals")
Need = Literal["docs", "web"]


class TaskSetup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: dict[str, str] = Field(default_factory=dict)  # written into the workspace first
    memories: list[str] = Field(default_factory=list)  # saved to the user's memory first


Phrase = str | list[str]
"""Text that must appear (case-insensitive), or a list of alternatives, any of which counts."""


class Checks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = "completed"
    answer_contains: list[Phrase] = Field(default_factory=list)
    answer_excludes: list[str] = Field(default_factory=list)
    min_citations: int = 0  # distinct [source#n] ids cited in the answer
    tools_called: list[str] = Field(default_factory=list)
    tools_not_called: list[str] = Field(default_factory=list)
    files: dict[str, list[Phrase]] = Field(default_factory=dict)  # path -> phrases it contains
    files_absent: list[str] = Field(default_factory=list)
    memories_contain: list[Phrase] = Field(default_factory=list)
    max_steps: int | None = None


class GoldenTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    goal: str
    tags: list[str] = Field(default_factory=list)
    needs: list[Need] = Field(default_factory=list)  # skipped when these are unavailable
    max_steps: int = Field(default=10, ge=1)
    setup: TaskSetup = TaskSetup()
    reject_tools: list[str] = Field(default_factory=list)  # the "user" rejects these calls
    checks: Checks = Checks()
    rubric: str | None = None
    criteria: list[str] = Field(default_factory=list)
    pass_score: float = Field(default=0.75, ge=0, le=1)  # judge score needed to pass

    @model_validator(mode="after")
    def _has_some_grading(self) -> "GoldenTask":
        # "Finished with status completed" alone says nothing about the result.
        checks = self.checks.model_dump(exclude_defaults=True, exclude={"status"})
        if not (checks or self.rubric or self.criteria):
            raise ValueError(f"Task {self.id!r} has no checks and no judge criteria")
        return self


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    criteria: list[str] = Field(min_length=1)


def load_tasks(tasks_dir: Path, tags: set[str] | None = None) -> list[GoldenTask]:
    """All tasks in `tasks_dir`, or only those with at least one of `tags`."""
    tasks: list[GoldenTask] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        items = yaml.safe_load(path.read_text()) or []
        if not isinstance(items, list):
            raise ValueError(f"{path} must contain a list of tasks")
        tasks += [GoldenTask.model_validate(item) for item in items]
    ids = [t.id for t in tasks]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate task ids: {', '.join(duplicates)}")
    if tags:
        tasks = [t for t in tasks if tags & set(t.tags)]
    return tasks


def load_rubrics(rubrics_dir: Path) -> dict[str, Rubric]:
    rubrics = {}
    for path in sorted(rubrics_dir.glob("*.yaml")):
        rubric = Rubric.model_validate(yaml.safe_load(path.read_text()))
        if rubric.name != path.stem:
            raise ValueError(f"{path} declares name {rubric.name!r}; it must match the file name")
        rubrics[rubric.name] = rubric
    return rubrics


def criteria_for(task: GoldenTask, rubrics: dict[str, Rubric]) -> list[str]:
    """The rubric's criteria followed by the task's own."""
    shared: list[str] = []
    if task.rubric:
        if task.rubric not in rubrics:
            raise ValueError(f"Task {task.id!r} uses unknown rubric {task.rubric!r}")
        shared = rubrics[task.rubric].criteria
    return [*shared, *task.criteria]
