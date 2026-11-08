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
- [ ] CI is green on GitHub (after the first push)

## M1: Model layer (next)

**Goal:** role-based models behind one factory, testable offline.

**Acceptance criteria**
- [ ] `config.py` reads settings from the environment with pydantic-settings
- [ ] `llm.get_chat_model(role)` for planner, worker, router and judge, with timeouts and retries
- [ ] Structured-output helper
- [ ] Fake chat model in `tests/fixtures/` that can script tool calls
- [ ] Switching model or provider is an env change only
- [ ] All tests run with no network
