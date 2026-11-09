# Architecture

Goalsmith turns a goal into a finished, checked deliverable. It has five layers. Requests flow down from the interfaces, and every action an agent takes goes through the **harness** on its way to the capabilities and data below.

```
Interfaces    Web UI (Streamlit) · CLI (goalsmith run "goal") · MCP clients
                    │  HTTP, SSE, MCP
Service       FastAPI (REST, SSE, MCP server) · Redis queue (arq) · Workers
                    │  run jobs
Run engine    LangGraph run graph (planner, researcher, analyst, coder, writer, critic)
              Agent harness (loop, tool dispatch, permissions, budgets, hooks, compaction, subagents)
                    │  tool calls, permission-checked
Capabilities  RAG (hybrid + rerank) · Tools (web, files, code) · MCP client · Skills · Models (llm.py)
                    │  read and write
Data          Postgres + pgvector (checkpoints, runs, memory) · Qdrant (vectors) · Workspace files

Across every layer: guardrails, OpenTelemetry traces to Langfuse, eval suite gating CI
```

## Key ideas

- **The API is stateless.** It only queues runs. Workers execute the LangGraph run graph inside the harness. All state lives in Postgres, so a run survives a restart.
- **The harness is the single gateway** to RAG, tools, MCP servers, skills and models. Permissions, budgets, guardrails and tracing apply everywhere without being repeated in each agent.
- **Run flow:** guard_input → planner → dispatcher → (researcher × N | analyst | coder → human_approval) → writer ⇄ critic (at most one revision) → guard_output.
- **Run lifecycle:** queued → planning → running → (awaiting_approval → running)* → reviewing → completed | failed | cancelled | budget_exceeded.

## Three ways to extend

| Extension point | Adds | How |
|---|---|---|
| Skill | Know-how | Drop a folder `skills/<name>/` with a `SKILL.md` |
| MCP server | Abilities | Add an entry to `mcp_servers.yaml` |
| Agent | A specialist | Add a spec to `agents/<name>.yaml` |

## RAG (M3)

```
ingest:  file -> parse (md / html / pdf) -> sections by heading or page -> chunks (source id, heading path)
         chunk -> bge-small embedding + BM25 sparse vector -> Qdrant point (payload = chunk)
search:  query -> dense top 20  ┐
         query -> BM25 top 20   ┴> RRF fusion -> cross-encoder rerank -> top k -> search_docs output
```

The code is in `capabilities/rag/`:

| File | Job |
|---|---|
| `ingest.py` | Parsers, chunker, ingest pipeline |
| `sparse.py` | BM25 encoder |
| `store.py` | The only code that talks to Qdrant |
| `retrieve.py` | RRF fusion and `HybridRetriever` |
| `rerank.py` | Reranking step |

The model-facing tool is `capabilities/tools/docs.py`. See ADR 0005.

## Approvals, checkpoints and memory (M4)

```
START -> agent --(read-only tool calls)--> tools -> agent ... -> END
               \--(write / external calls)--> approval --interrupt()--> [paused, saved in Postgres]
                                                  ^                          |
                                                  └── Command(resume=decisions) <- human: approve / edit / reject
```

- **Checkpointer:** `AsyncPostgresSaver` saves the state after every node, keyed by `thread_id` = run id.
- **Approval logic:** pure functions in `harness/permissions.py`.
- **Memory:** `memory/store.py` (LangGraph Store + pgvector). The tools live in `capabilities/tools/memory.py`.
- **Wiring:** Postgres or in-memory fallback is chosen in `core/persistence.py`. See ADR 0006.

## Tracing, prompts and evals (M5)

```
goalsmith run (CLI root span, session = run id)
└─ agent.run / agent.resume        AgentLoop: outside the nodes, so nothing repeats on resume
   ├─ model <name>                 ModelCallTracer (LangChain callback): messages, tokens, cost, prompt version
   ├─ tool <name>                  ToolRegistry.execute: arguments, result, error level
   └─ approval.requested/decided   events
        │ OTLP/HTTP
        ▼
Langfuse (make obs-up) ── also any OTLP backend
```

- **Observability** lives in `observability/`:
  - `spans.py`: OpenTelemetry helpers that set Langfuse attributes;
  - `tracing.py`: setup, plus the model-call callback;
  - `usage.py`: tokens and cost;
  - `logs.py`: structlog lines carrying `run_id` and `trace_id`.
- **Prompts:** `src/goalsmith/prompts/*.md`, versioned. The version and a hash of the text go on every span.
- **Evals:**
  - golden tasks in `evals/tasks/`, shared rubrics in `evals/rubrics/`;
  - `evals/checks.py` runs the deterministic checks, `evals/judges.py` the LLM judge, and `evals/runner.py` runs them and writes the report;
  - CI posts the smoke table on each PR. See ADR 0007.

This document grows as milestones land. Decisions are recorded in [adr/](adr/).
