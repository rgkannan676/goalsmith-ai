# Goalsmith: notes for Claude Code

## What this is
Goalsmith is a multi-agent workbench on LangGraph: goal -> plan -> specialist agents -> checked deliverable.
Read docs/architecture.md and docs/adr/ before changing structure.
The current milestone and its acceptance criteria are in docs/milestones.md.

## Commands
- Setup: `make setup` (uv sync + pre-commit install)
- Lint/format/types: `make lint` (ruff check, ruff format --check, mypy src)
- Unit tests: `uv run pytest -m "not integration"` (or `make test`)
- Integration: `make up` then `uv run pytest -m integration`
- Evals: `make eval-smoke`
- Add a dependency: `uv add <pkg>` (never pip install)

## Architecture rules
- Graph nodes are pure: (state) -> partial update. No I/O in nodes.
- All tool calls go through src/goalsmith/harness. Never call a tool or MCP client directly from an agent.
- Models come only from llm.get_chat_model(role). Never hard-code a provider or model name elsewhere.
- Every shared state field that parallel nodes write must have a reducer.
- New tools need: Pydantic args schema, risk level (read/write/external), unit test.
- New skills live in skills/<name>/SKILL.md with name + description front matter.

## Code style
- Python 3.12, type hints everywhere, mypy strict on src/.
- Async for I/O. Pydantic v2 models at boundaries.
- Small functions, no broad `except Exception` without re-raise or logging.
- No secrets in code, tests or logs. Use config.py settings.

## Testing
- Every change ships with tests. Use the fake chat model from tests/fixtures; unit tests must not hit the network.
- When fixing a bug, write the failing test first.

## Workflow
- Work on the current milestone branch only (`feat/mNN-<name>`). Keep diffs small and focused on the task.
- Before finishing: run make lint and unit tests and report results.
- Do not refactor unrelated code or change public interfaces without asking.
- Check library APIs against the installed version (uv pip show <pkg>) before using them.
- Record rejected or corrected AI suggestions in docs/ai-review-log.md.
