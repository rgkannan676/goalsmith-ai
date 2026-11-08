# Milestones

Each milestone is one branch (`feat/mNN-<name>`). It ends with green tests, an updated README and a merge into `main`.

## M0: Skeleton ✅ (branch `feat/m00-skeleton`)

**Goal:** repo hygiene and CI, with no AI code yet.

**Acceptance criteria**
- [x] uv project with Python 3.12, `pyproject.toml` and `uv.lock`
- [x] ruff, mypy (strict) and pytest configured
- [x] pre-commit with ruff and gitleaks
- [x] Makefile: `setup lint format test eval-smoke eval-full up down`
- [x] `.env.example` committed, `.env` git-ignored
- [x] GitHub Actions CI: lint, format check, mypy, unit tests, secret scan
- [x] `CLAUDE.md`, architecture doc, ADRs, AI review log
- [x] A fresh clone passes `make setup && make test`
- [x] CI is green on GitHub (PR #1)

## M1: Model layer ✅ (branch `feat/m01-model-layer`)

**Goal:** role-based models behind one factory, testable offline.

**Acceptance criteria**
- [x] `config.py` reads settings from the environment with pydantic-settings
- [x] `llm.get_chat_model(role)` for planner, worker, router and judge, with timeouts and retries
- [x] Structured-output helper (`llm.generate_structured`)
- [x] Fake chat model in `tests/fixtures/` that can script tool calls
- [x] Switching model or provider is an env change only (tested)
- [x] All tests run with no network (a fixture blocks sockets in unit tests)

**Moved:** `get_embeddings()` moves to M3, where embeddings are first used (see ADR 0003).

## M2: Single agent and harness v1 (next)

**Goal:** a hand-built agent loop that calls tools through the harness and stops at a budget.

**Acceptance criteria**
- [ ] Minimal `RunState` in `core/state.py`
- [ ] `harness/loop.py`: model node, tool node, conditional edge (LangGraph)
- [ ] `harness/tools.py`: tool registry, Pydantic argument validation, risk levels
- [ ] `harness/budget.py`: max-steps limit with a clean stop
- [ ] Tools: `web_search` (Tavily), `read_file`, `write_file` (run workspace only)
- [ ] `goalsmith run "<goal>"` CLI
- [ ] A simple research goal completes; a test proves the loop stops at the step budget
