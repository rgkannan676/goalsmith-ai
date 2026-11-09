# ADR 0007: OpenTelemetry tracing to Langfuse, versioned prompts, golden-task evals

- Status: accepted
- Date: 2026-11-10

## Context
M5 must make every run diagnosable from its trace, track tokens and cost, version the prompts, and grade the agent on every PR.

## Decisions

**1. Plain OpenTelemetry, not the Langfuse SDK.**
- Spans are created with the OpenTelemetry API and exported over OTLP/HTTP to Langfuse's `/api/public/otel/v1/traces`.
- Langfuse reads a few extra attributes to type and enrich them. Examples: `langfuse.observation.type` (agent, generation, tool, event, evaluator), `langfuse.observation.usage_details`, `langfuse.observation.cost_details`, `langfuse.session.id`.
- Any OTLP backend (Jaeger, Honeycomb, Grafana Tempo) can receive the same traces. Swapping backends is a config change.
- Without Langfuse keys no tracer provider is installed, and every span call is a cheap no-op.

**2. Where spans come from.**

| Span | Created by | Why there |
|---|---|---|
| `goalsmith run` (root) | CLI | One trace per command: a run plus its in-process approvals |
| `agent.run` / `agent.resume` | `AgentLoop` | Outside the graph nodes, so nothing repeats when a node re-runs on resume |
| `model <name>` (generation) | `ModelCallTracer`, a LangChain callback attached in `llm.get_chat_model` | Every model, any provider, with no tracing code in nodes |
| `tool <name>` | `ToolRegistry.execute` | The single path every tool call takes |
| `approval.requested` / `approval.decided` (events) | `AgentLoop` | Shows who decided what, and when |
| `eval.task`, `judge` (evaluator) | Eval runner and judge | Eval runs can be filtered by tag `eval` |

- Trace-level fields are copied onto every span by a span processor (`TraceAttributesProcessor`), because Langfuse filters on them most reliably that way. The fields: session = run id, user, tags, and prompt version.
- A paused and resumed run produces two traces with the same session id. Langfuse's session view shows them together.

**3. Turn off the Anthropic SDK's own spans.**
- The Anthropic SDK (2.x) traces its own requests whenever a tracer provider is installed. The first live run showed each model call twice, and Langfuse doubled the trace's cost.
- We keep our provider-neutral generation spans, which also carry the prompt version and the LangGraph node.
- `_build_anthropic` sets `ANTHROPIC_OPEN_TELEMETRY=false`, with `setdefault`, so it can be overridden. langchain-anthropic offers no way to pass the SDK's `open_telemetry` option, so the documented environment variable is the only clean switch.

**4. Tokens and cost in the run state.**
- `usage_from_message` reads LangChain's `usage_metadata`, where input includes cache reads and writes, and prices it from `Settings.prices`.
- Cache reads are priced separately. Prices live in config, like model names.
- The run total sits in `BudgetUsage.tokens`, so it is checkpointed and survives pause and resume. M11 adds token and cost budgets on top of it.
- Our cost matched Langfuse's own price for the same calls exactly ($0.00392 and $0.005648).

**5. Logs: structlog to stderr.**
- Each line is an event name plus fields.
- `run_id` is bound per run, and `trace_id` and `span_id` are added from the current span, so a log line and its trace can be matched in either direction.
- The default level is WARNING, so the CLI stays quiet. `GOALSMITH_LOG_LEVEL=INFO` and `GOALSMITH_LOG_FORMAT=json` give full logs.

**6. Prompts are Markdown files with a YAML header, inside the package.**
- Location: `src/goalsmith/prompts/<name>.md`, with fields `name`, `version`, `description`.
- They sit inside the package, not at the repo root, so they ship with it (Docker in M14) and load the same way from any directory.
- Traces record `prompt_name`, `prompt_version` and `prompt_sha` (a hash of the text). An edit without a version bump shows up as a new hash.
- Changes are explained in `prompts/CHANGELOG.md`.

**7. Golden tasks: deterministic checks first, then an LLM judge.**
- 15 tasks in `evals/tasks/*.yaml`, grouped by area: documents, files, memory, safety and web.
- Checks are free and never flaky: answer phrases, citations, tools called or not called, files written or absent, memories saved, step count.
- The judge grades numbered criteria pass/fail with a one-sentence reason. Pass/fail is more consistent than a 1–10 score, and each failure is easy to verify by hand.
- Score = fraction passed. A task passes when all checks pass and the score is at least `pass_score` (default 0.75).
- Shared criteria live in `evals/rubrics/`. The judge prompt is versioned like the agent's.
- `make eval-smoke` runs the 6 tasks tagged `smoke` (about $0.07, 20 s). CI posts the table as a PR comment and fails below 80%.

**8. Langfuse runs in Compose under the `observability` profile.**
- Langfuse v4 needs six containers: web, worker, Postgres, ClickHouse, Redis and MinIO.
- `make up` stays light; `make obs-up` starts Langfuse.
- On first start it creates the `goalsmith` project with local keys, so tracing works without clicking through the UI.

## Rejected
- **Langfuse Python SDK / LangChain `CallbackHandler`.** It's quicker to start, but it ties the code to one vendor, and the SDK is itself built on OpenTelemetry anyway.
- **Keeping the Anthropic SDK's spans instead of ours.** They are accurate, but provider-specific, they lack the prompt version and graph node, and they are siblings rather than children of our model step.
- **Prompts in Langfuse prompt management.** That would make a running service a dependency for loading prompts. Files are reviewed in PRs and versioned in git.
- **1–10 judge scores.** They drift between runs and between judge models.
- **Running the eval with fake models in CI.** That tests plumbing (unit tests already do), not the agent.

## Consequences
- Every PR with the `ANTHROPIC_API_KEY` secret costs about $0.07 in model calls. Fork PRs get no secrets, so the eval is skipped there, not failed.
- The first eval run found two mistakes in the eval itself, not the agent: a too-strict check and a judge record that cut off the evidence. See the review log. Evals need review like code.
- The judge is the same model family as the worker. A judge from a different family is planned once a second provider is configured.
