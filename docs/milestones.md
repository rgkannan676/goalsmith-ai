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

## M2: Single agent and harness v1 ✅ (branch `feat/m02-single-agent`)

**Goal:** a hand-built agent loop that calls tools through the harness and stops at a budget.

**Acceptance criteria**
- [x] Minimal `RunState` in `core/state.py`
- [x] `harness/loop.py`: model node, tool node, conditional edge (LangGraph)
- [x] `harness/tools.py`: tool registry, Pydantic argument validation, risk levels
- [x] `harness/budget.py`: max-steps limit with a clean stop and partial result
- [x] Tools: `web_search` (Tavily), `read_file`, `write_file` (run workspace only)
- [x] `goalsmith run "<goal>"` CLI with live progress
- [x] A test proves the loop stops at the step budget (`test_loop_stops_at_step_budget`)
- [x] A live goal completes (write_file run against Claude)
- [ ] A live *research* goal completes (needs `TAVILY_API_KEY`)

## M3: RAG ✅ (branch `feat/m03-rag`)

**Goal:** search uploaded documents with hybrid retrieval and cite sources.

**Acceptance criteria**
- [x] Ingest PDF, markdown and HTML; structure-aware chunks with metadata
- [x] `get_embeddings()` with local bge via fastembed (moved from M1)
- [x] Qdrant dense search + BM25, fused with reciprocal rank fusion
- [x] Cross-encoder reranking of the top results
- [x] `search_docs` tool with collections; every chunk carries a source id
- [x] `docker-compose.yml` with Qdrant; `make up` works
- [x] hit@5, MRR and nDCG reported on 40 labelled queries (`make eval-retrieval`, see `evals/retrieval/report.md`)
- [x] Answers cite source ids (`test_answer_cites_the_source_id`, plus a live run against Claude)

**Notes:** the eval code lives in `src/goalsmith/evals/` (type-checked), and the datasets live in `evals/`. M5 follows the same split. Design and results are in ADR 0005.

## M4: Checkpoints, human approval and memory (next)

**Goal:** runs survive restarts, risky tools wait for a human, and agents remember across runs.

**Acceptance criteria**
- [ ] Postgres + pgvector in Docker Compose
- [ ] `langgraph-checkpoint-postgres`: runs are resumable by `thread_id`
- [ ] `harness/permissions.py`: `interrupt()` before `write` and `external` tools; resume with approve, edit or reject
- [ ] `memory/store.py`: `remember` and `recall` on the LangGraph Store
- [ ] An approval survives a process restart (integration test)
- [ ] A rejected action never runs (integration test)
