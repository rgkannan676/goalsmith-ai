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
| `docs/` | Architecture, ADRs, milestone log, AI review log |
| `CLAUDE.md` | Rules for working on this repo with Claude Code |

## Milestones

| | Milestone | Status |
|---|---|---|
| M0 | Skeleton: tooling, CI, docs | ✅ |
| M1 | Model layer | ✅ |
| M2 | Single agent and harness v1 | ⏳ |
| M3 | RAG | ⏳ |
| M4 | Checkpoints, human approval, memory | ⏳ |
| M5 | Tracing and eval v1 | ⏳ |
| M6–M16 | Multi-agent, MCP, skills, guardrails, API, deploy, showcase | ⏳ |

## License

[MIT](LICENSE)
