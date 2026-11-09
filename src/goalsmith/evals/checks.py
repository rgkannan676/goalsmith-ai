"""Deterministic checks on a finished run: fast, free and never flaky.

They catch the obvious failures (wrong number, missing file, forbidden tool call)
before an LLM judge looks at the subtler ones.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from goalsmith.evals.tasks import Checks, Phrase

CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")  # e.g. [handbook.md#4]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class RunOutcome:
    """What a run left behind, in a form the checks and the judge can read."""

    status: str
    answer: str
    steps: int
    tool_calls: list[str] = field(default_factory=list)  # tool names, in call order
    files: dict[str, str] = field(default_factory=dict)  # workspace path -> text
    memories: list[str] = field(default_factory=list)


def run_checks(checks: Checks, outcome: RunOutcome) -> list[CheckResult]:
    results: list[CheckResult] = []
    add = results.append

    if checks.status is not None:
        add(_equal("status", checks.status, outcome.status))
    for phrase in checks.answer_contains:
        add(_contains(f"answer contains {_label(phrase)}", outcome.answer, phrase))
    for text in checks.answer_excludes:
        found = text.lower() in outcome.answer.lower()
        add(CheckResult(f"answer excludes {text!r}", not found, "found" if found else ""))
    if checks.min_citations:
        cited = sorted(set(CITATION.findall(outcome.answer)))
        add(
            CheckResult(
                f"cites at least {checks.min_citations} source(s)",
                len(cited) >= checks.min_citations,
                ", ".join(cited) or "no citations",
            )
        )
    for tool in checks.tools_called:
        add(CheckResult(f"called {tool}", tool in outcome.tool_calls, _calls(outcome)))
    for tool in checks.tools_not_called:
        add(CheckResult(f"did not call {tool}", tool not in outcome.tool_calls, _calls(outcome)))
    for path, phrases in checks.files.items():
        content = outcome.files.get(path)
        if content is None:
            add(CheckResult(f"wrote {path}", False, f"files: {sorted(outcome.files) or 'none'}"))
            continue
        add(CheckResult(f"wrote {path}", True))
        for phrase in phrases:
            add(_contains(f"{path} contains {_label(phrase)}", content, phrase))
    for path in checks.files_absent:
        add(CheckResult(f"did not write {path}", path not in outcome.files))
    for phrase in checks.memories_contain:
        add(_contains(f"memory contains {_label(phrase)}", "\n".join(outcome.memories), phrase))
    if checks.max_steps is not None:
        add(
            CheckResult(
                f"at most {checks.max_steps} steps",
                outcome.steps <= checks.max_steps,
                f"{outcome.steps} steps",
            )
        )
    return results


def _contains(name: str, text: str, phrase: Phrase) -> CheckResult:
    options = [phrase] if isinstance(phrase, str) else phrase
    lowered = text.lower()
    passed = any(option.lower() in lowered for option in options)
    return CheckResult(name, passed, "" if passed else f"not found in: {_excerpt(text)}")


def _equal(name: str, expected: str, actual: str) -> CheckResult:
    return CheckResult(f"{name} is {expected}", expected == actual, f"got {actual}")


def _label(phrase: Phrase) -> str:
    return repr(phrase) if isinstance(phrase, str) else " or ".join(map(repr, phrase))


def _calls(outcome: RunOutcome) -> str:
    return "calls: " + (", ".join(outcome.tool_calls) or "none")


def _excerpt(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def all_passed(results: Sequence[CheckResult]) -> bool:
    return all(r.passed for r in results)
