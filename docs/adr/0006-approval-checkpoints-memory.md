# ADR 0006: Human approval via interrupt, Postgres checkpoints, and approval-gated memory

- Status: accepted
- Date: 2026-11-09

## Context
M4 must:
- pause risky tool calls for a human;
- survive a process restart while paused;
- never run a rejected action;
- give agents memory across runs.

## Decisions

**1. A dedicated `approval` node, between `agent` and `tools`.**
- The edge after `agent` routes to `approval` only when a tool call's risk is in `Permissions.needs_approval` (default: `write`, `external`).
- The node calls LangGraph's `interrupt()` with a JSON payload of `ApprovalRequest`s.
- On resume, `interrupt()` returns the decisions, and `apply_decisions` turns them into state updates:
  - **approve**: the call stays as it is.
  - **edit**: the model's message is replaced *in place* (same message id, via the `add_messages` reducer) with the new arguments, so the history shows what actually ran.
  - **reject**: a `ToolMessage` with status `error` answers the call. The tools node skips calls that already have an answer, so a rejected call cannot run.
- Nothing with side effects happens before `interrupt()` in the node. That matters because LangGraph re-runs the whole node on resume.

A separate node (rather than an `interrupt()` inside the tools node) keeps the pause visible in the graph, keeps approval logic pure and testable (`harness/permissions.py`), and leaves the tools node unchanged.

**2. Checkpointer = `AsyncPostgresSaver`, `thread_id` = run id.**
- State is saved after every node.
- `goalsmith resume <run_id>` builds a fresh `AgentLoop` and continues from Postgres.
- The acceptance test proves this across a **real process restart**: a child process starts the run and exits while paused.
- Without Postgres, runs use `InMemorySaver`. Approvals still work, but a paused run cannot outlive the process, and the CLI says so.

**3. Checkpoint type allowlist (`core/serde.py`).**
- LangGraph warns, and will later refuse, when a checkpoint names a class that isn't allowlisted. This guards against tampered checkpoints that try to load arbitrary classes.
- Custom state types (now only `BudgetUsage`) are registered explicitly.

**4. Long-term memory on the LangGraph Store.**
- `AsyncPostgresStore` with a pgvector index over the memory text, using the same local bge embeddings as RAG.
- Namespace: `("users", <user_id>, "memories")`.
- `recall` is a `read` tool. `remember` is a `write` tool, so **saving a memory needs approval**. Memory outlives the run, so a memory planted by prompt injection (for example from a web page) would affect every later run.

**5. Vector size check at startup.** pgvector fixes the vector size when the table is created. A changed embedding model, or a test that used fake 64-dimension vectors, otherwise fails on the first `remember` with a cryptic error. `postgres_store` now compares the stored size (`pg_attribute.atttypmod`) and fails early with a clear message.

**6. Integration tests use a separate `goalsmith_test` database.** One test once created the dev memory table with 64-dimension vectors, which broke the live run (see the review log).

## Rejected
- **`interrupt()` inside the tools node.** It works, but the pause is hidden, and any side effect added before it later would repeat on resume.
- **Asking only for `external` tools.** The plan specifies `write` too. `--yes` and `GOALSMITH_APPROVAL_RISKS` cover trusted, non-interactive runs.
- **Pickle serialization for checkpoints.** It's unsafe to load and opaque to inspect.

## Consequences
- Every `write_file` asks by default, which is safe but chatty. M13's UI will make approving one click.
- `resume` uses the current settings for the step budget (or `--max-steps`), because the original limit is not saved with the run.
- Checkpoints accumulate in Postgres. Retention or cleanup comes with the run history work in M13.
