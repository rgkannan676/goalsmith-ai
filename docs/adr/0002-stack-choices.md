# ADR 0002: Initial stack choices

- Status: accepted
- Date: 2026-11-08

## Context
The project plan lists several open decisions. Each has a default so work can start now. These choices can be revisited in later ADRs.

## Decision

| Area | Choice | Why |
|---|---|---|
| Language and packaging | Python 3.12, uv, `uv.lock` | Fast, reproducible installs |
| Orchestration | LangGraph | Explicit graph, checkpoints, human-in-the-loop |
| Model provider | Anthropic via `langchain-anthropic` | Key already available. `llm.py` stays provider-neutral; model names live only in config. |
| Critic / judge model | A different Claude tier for now | The plan prefers a different model family. Add a second provider later if a key is available. |
| Embeddings | Local `bge` via `fastembed` | Anthropic has no embeddings API. Local models are free and work offline. |
| Web search | Tavily | Free tier, simple API |
| Vector store | Qdrant + BM25 | Hybrid search, filters, easy Docker |
| Observability | Self-hosted Langfuse via OpenTelemetry | Traces, cost, datasets in one place; swappable |
| Job queue | arq on Redis | Async and lighter than Celery |
| UI | Streamlit until M16 | Quick demo surface |
| Deployment | Azure Container Apps, VPS as fallback | Decided for good at M14 |
| Licence | MIT | Simple and permissive |

## Consequences
- `langchain-anthropic` replaces `langchain-openai` in the plan's dependency list.
- The RAG milestone (M3) depends on `fastembed` models being downloaded once.
- Tests never call real providers; they use a fake chat model (M1).
