"""LLM judge: grades a run against numbered criteria, pass or fail.

Pass/fail per criterion, rather than a 1-10 score, because:
- binary answers are more consistent between runs and between judge models;
- each verdict carries a reason, so a failure is easy to check by hand;
- the score is simply the fraction of criteria passed.

The judge sees a plain-text record of the run (goal, tool calls and results, final
answer, files written), not the raw message objects. Its prompt is versioned in
prompts/judge.md, like the agent's.
"""

import json
from collections.abc import Mapping, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from goalsmith.llm import generate_structured
from goalsmith.observability.spans import observation, set_metadata, set_output
from goalsmith.prompts import Prompt, load_prompt

# Per tool result in the record. The judge must see the evidence a claim rests on;
# at 1,500 characters it could not see web search results and failed grounded answers.
MAX_RESULT_CHARS = 8000
MAX_FILE_CHARS = 4000


class CriterionVerdict(BaseModel):
    number: int = Field(description="The criterion's number, starting at 1")
    passed: bool
    reason: str = Field(description="One sentence pointing to evidence in the record")


class JudgeVerdicts(BaseModel):
    """What the judge model returns."""

    verdicts: list[CriterionVerdict]


class JudgeResult(BaseModel):
    criteria: list[str]
    verdicts: list[CriterionVerdict]  # one per criterion, in order

    @property
    def score(self) -> float:
        if not self.verdicts:
            return 1.0
        return sum(v.passed for v in self.verdicts) / len(self.verdicts)

    def failures(self) -> list[tuple[str, str]]:
        return [
            (criterion, verdict.reason)
            for criterion, verdict in zip(self.criteria, self.verdicts, strict=True)
            if not verdict.passed
        ]


class Judge:
    def __init__(self, model: BaseChatModel, prompt: Prompt | None = None) -> None:
        self._model = model
        self._prompt = prompt or load_prompt("judge")

    @property
    def prompt(self) -> Prompt:
        return self._prompt

    async def grade(self, criteria: Sequence[str], record: str) -> JudgeResult:
        if not criteria:
            return JudgeResult(criteria=[], verdicts=[])
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
        message = HumanMessage(self._prompt.format(criteria=numbered, record=record))
        with observation(
            "judge", "evaluator", input=numbered, metadata={"prompt": self._prompt.ref}
        ) as span:
            raw = await generate_structured(self._model, JudgeVerdicts, [message])
            result = JudgeResult(criteria=list(criteria), verdicts=_align(raw, len(criteria)))
            set_output(span, [v.model_dump() for v in result.verdicts])
            set_metadata(span, score=result.score)
        return result


def _align(raw: JudgeVerdicts, count: int) -> list[CriterionVerdict]:
    """One verdict per criterion, in order. A missing verdict counts as a fail."""
    by_number = {v.number: v for v in raw.verdicts}
    return [
        by_number.get(n)
        or CriterionVerdict(number=n, passed=False, reason="The judge gave no verdict.")
        for n in range(1, count + 1)
    ]


def render_record(
    goal: str,
    messages: Sequence[AnyMessage],
    answer: str,
    files: Mapping[str, str],
    *,
    status: str,
) -> str:
    """The run as plain text for the judge: what was asked, done, answered, written."""
    lines = ["## Goal", goal, "", "## Steps"]
    step = 0
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                step += 1
                args = json.dumps(call["args"], ensure_ascii=False)
                lines.append(f"{step}. The agent called `{call['name']}` with {args}")
        elif isinstance(message, ToolMessage):
            status_word = "error" if message.status == "error" else "result"
            lines.append(f"   {status_word}: {_clip(str(message.content), MAX_RESULT_CHARS)}")
    if step == 0:
        lines.append("(no tool calls)")
    lines += ["", f"## Final answer (run status: {status})", answer or "(empty)"]
    lines += ["", "## Files in the workspace after the run"]
    if not files:
        lines.append("(none)")
    for path, text in sorted(files.items()):
        lines += [f"### {path}", _clip(text, MAX_FILE_CHARS)]
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    return (
        text if len(text) <= limit else text[:limit] + f"\n...[{len(text) - limit} more characters]"
    )
