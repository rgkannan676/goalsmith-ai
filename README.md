# Goalsmith

**From goal to done.** A multi-agent workbench that plans, researches, analyses and acts with RAG, MCP tools and pluggable skills, with human approval, full tracing and built-in evaluation.

> Status: early development. Built in public, one milestone at a time. See [docs/milestones.md](docs/milestones.md).

## What it will do

Give Goalsmith a goal in plain English. It plans the work, assigns specialist agents (researcher, analyst, coder, writer, critic) and returns a finished, checked result: a cited report, a chart pack, a code review, or an approved action such as a GitHub issue.

## Quick start (developers)

Requirements: [uv](https://docs.astral.sh/uv/) and Docker (Docker is needed from M3).

```bash
git clone <repo-url> goalsmith-ai && cd goalsmith-ai
cp .env.example .env      # add your keys
make setup                # install deps + pre-commit hooks
make lint test            # ruff, mypy --strict, unit tests
```

## Run a goal

```bash
uv run goalsmith run "Find three recent articles on agent harnesses and save a summary to notes.md"
```

The agent works in a loop: Claude decides which tool to call, the **harness** validates and runs it, and the result goes back to Claude, until it gives a final answer or hits the step budget (`--max-steps`, default 15). Files go to `workspaces/<run_id>/`.

| Tool | Risk | What it does |
|---|---|---|
| `web_search` | read | Tavily web search (needs `TAVILY_API_KEY`) |
| `read_file` | read | Read a file from the run workspace |
| `write_file` | write | Write a file in the run workspace |
| `search_docs` | read | Hybrid search over your ingested documents, with citable source ids |
| `recall` | read | Search memories saved in earlier runs |
| `remember` | write | Save a lasting fact or preference about you |

## Approvals, pause and resume

Tools that change something (`write` and `external` risk) **wait for your approval** before they run:

```
╭──────── Approve write_file? (risk: write) ────────╮
│ path: tip.md                                      │
│ content: Tip: spell it "colour" (with a "u")…     │
╰───────────────────────────────────────────────────╯
[a]pprove, [e]dit or [r]eject:
```

- **Approve** runs the call as the model asked.
- **Edit** runs it with the arguments you type (as JSON).
- **Reject** never runs it, and the model is told why.

Every step of a run is saved in Postgres (`make up`). If you quit at the prompt, the run pauses, and you can pick it up later, even after a restart:

```bash
uv run goalsmith resume <run_id>
```

`--yes` approves everything without asking. `GOALSMITH_APPROVAL_RISKS='["external"]'` asks only for outside-world actions.

## Memory

`remember` and `recall` give the agent memory across runs. Memories are stored in Postgres (pgvector) and searched by meaning, so "how do I spell colour?" finds "Prefers British English spelling". Saving a memory needs your approval, because it affects every later run.

## Search your documents (RAG)

```bash
make up                                   # Qdrant and Postgres in Docker
uv run goalsmith ingest path/to/docs      # .md .html .pdf .txt, files or folders
uv run goalsmith search "parental leave"  # check retrieval without the agent
uv run goalsmith run "How much parental leave do I get?"
```

Answers cite the passages they use, e.g. `18 weeks of paid leave [handbook.md#4]`.

How search works:
1. **Ingest:** files are split along their headings (or PDF pages) into chunks. Each chunk keeps its heading path and a source id.
2. **Index:** every chunk gets a dense embedding (local `bge-small`) and a BM25 sparse vector. Both are stored in Qdrant.
3. **Search:** dense and BM25 results are merged with reciprocal rank fusion, then a cross-encoder reranks the top 20.

No Docker? Set `GOALSMITH_RAG__QDRANT_PATH=.qdrant` to run Qdrant inside the process. The local models (about 150 MB) download on first use.

`make eval-retrieval` measures search quality on 40 labelled queries:

| Setup | hit@5 | MRR | nDCG@5 |
|---|---|---|---|
| dense | 0.975 | 0.963 | 0.966 |
| bm25 | 1.000 | 0.946 | 0.960 |
| hybrid (RRF) | 1.000 | 0.971 | 0.978 |
| hybrid + rerank | 1.000 | 1.000 | 1.000 |

See [ADR 0005](docs/adr/0005-hybrid-retrieval.md) for the design and [evals/retrieval/report.md](evals/retrieval/report.md) for the full report.

## Traces, tokens and cost

Every run is traced with OpenTelemetry and sent to a self-hosted [Langfuse](https://langfuse.com):

```bash
make obs-up    # Langfuse at http://localhost:3000 (admin@goalsmith.local / goalsmith-local)
```

Then add the local keys to `.env` (see `.env.example`). Each run prints its usage and a link to its trace:

```
Usage: 3,884 in / 180 out tokens · $0.0096
Trace: http://localhost:3000/project/goalsmith/traces/e167dcdd…
```

The trace shows the run as a tree:
- every model call, with its messages, tokens, cost and prompt version;
- every tool call, with its arguments and result (errors in red);
- every approval request and decision.

Log lines carry the same `run_id` and `trace_id` (`GOALSMITH_LOG_LEVEL=INFO`). Without Langfuse keys, tracing is off and costs nothing. See [ADR 0007](docs/adr/0007-tracing-prompts-evals.md) and [a worked diagnosis](docs/diagnosis/m5-unsupported-advice.md).

## Prompts

Prompts are versioned Markdown files in [`src/goalsmith/prompts/`](src/goalsmith/prompts/). Edit the text, bump `version`, and note the change in its `CHANGELOG.md`. Traces record the version, so you can compare runs before and after a prompt change.

## Evals

```bash
make eval-smoke   # 6 golden tasks, about $0.07 and 20 seconds
make eval-full    # all 15
```

Each golden task in [`evals/tasks/`](evals/tasks/) runs the real agent in a fresh workspace and is graded in two steps:
1. **Deterministic checks.** For example: the answer says "18 weeks", cites a source, never calls `remember`.
2. **An LLM judge.** It grades rubric criteria pass/fail, with a reason for each ("every fact is supported by a tool result").

On every pull request, CI runs the smoke set and posts the results table as a comment. This needs an `ANTHROPIC_API_KEY` repository secret.

## Models

Code never names a model. It asks for a **role**, and configuration picks the model:

```python
from goalsmith.llm import get_chat_model

planner = get_chat_model("planner")  # claude-opus-5-5 by default
```

| Role | Used for | Default |
|---|---|---|
| `planner` | Breaking a goal into steps | `claude-opus-5-5` |
| `worker` | Research, analysis, writing | `claude-sonnet-5-5` |
| `router` | Cheap routing and guard checks | `claude-haiku-5-5` |
| `judge` | Critic and eval scoring | `claude-opus-5-5` |

Override any of them in `.env`, e.g. `GOALSMITH_WORKER__MODEL=claude-haiku-5-5`. See [.env.example](.env.example).

## Project layout

| Path | What lives there |
|---|---|
| `src/goalsmith/` | The Python package |
| `tests/` | Unit tests (no network) and integration tests |
| `src/goalsmith/prompts/` | Versioned prompt files |
| `evals/` | Golden tasks, rubrics, and the retrieval corpus and labelled queries |
| `docs/` | Architecture, ADRs, milestone log, AI review log |
| `CLAUDE.md` | Rules for working on this repo with Claude Code |

## Milestones

| | Milestone | Status |
|---|---|---|
| M0 | Skeleton: tooling, CI, docs | ✅ |
| M1 | Model layer | ✅ |
| M2 | Single agent and harness v1 | ✅ |
| M3 | RAG | ✅ |
| M4 | Checkpoints, human approval, memory | ✅ |
| M5 | Tracing and eval v1 | ✅ |
| M6–M16 | Multi-agent, MCP, skills, guardrails, API, deploy, showcase | ⏳ |

## License

[MIT](LICENSE)
