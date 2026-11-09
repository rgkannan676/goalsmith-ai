# ADR 0004: A hand-built agent loop with one tool gateway

- Status: accepted
- Date: 2026-11-09

## Context
M2 needs a single agent that calls tools, can be bounded, and can later grow approvals (M4), tracing (M5), hooks and compaction (M11). LangGraph ships a prebuilt ReAct agent, and LangChain ships its own tool abstraction.

## Options
1. Use LangGraph's prebuilt agent and LangChain `@tool` functions.
2. Build the loop as an explicit `StateGraph` (agent node, tools node, conditional edge), with our own `Tool` and `ToolRegistry`.

## Decision
Option 2.
- **Graph:** `START → agent → (tools → agent)* → END`. The step budget is checked in the agent node *before* each model call, so a run stops cleanly with a partial answer and status `budget_exceeded`. LangGraph's `recursion_limit` is only a backstop.
- **One gateway:** every tool call goes through `ToolRegistry.execute`. It handles unknown tool names, the risk-level policy, Pydantic argument validation, expected `ToolError`s, unexpected exceptions and output truncation. Every outcome becomes a `ToolMessage`, so the model can recover and the run never crashes on a tool.
- **Risk levels:** `read`, `write` and `external`. The default run policy allows `read` and `write`, because writes are confined to the run workspace. `external` tools are blocked until M4 adds human approval.
- **Concurrency:** all tool calls from one model turn run concurrently with `asyncio.gather`, and results keep their order.
- **Workspace safety:** file paths are resolved, following `..` and symlinks, and must stay inside `workspaces/<run_id>/`.
- **Dependency injection:** the model, registry, tool context and search client are passed in, so tests use fakes and need no network.

## Consequences
- Every harness concern has one obvious home. Approvals, hooks and tracing in later milestones wrap `execute` instead of touching each tool.
- There is more code than the prebuilt agent, which is the point for a learning project. Each line can be explained in an interview.
- Tool schemas use the provider-neutral OpenAI "function" format, which LangChain converts for Anthropic and other providers.
