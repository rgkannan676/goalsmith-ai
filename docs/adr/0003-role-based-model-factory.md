# ADR 0003: Role-based model factory with a provider registry

- Status: accepted
- Date: 2026-11-08

## Context
Many parts of Goalsmith call language models: the planner, specialists, router, guards and the eval judge. They need different strengths and costs. The plan requires that switching provider or model is configuration only, and that unit tests never hit the network.

## Options
1. Each module builds its own `ChatAnthropic(...)` with a model name.
2. A single factory `get_chat_model(role)`. Settings map each role to a `ModelSpec` (provider, model, max tokens), and a small registry maps provider names to builder functions.
3. Use LiteLLM as a universal provider layer.

## Decision
Option 2.
- Roles: `planner`, `worker`, `router`, `judge`. Code asks for a role, never a model name.
- Settings come from env vars via pydantic-settings, with nested overrides such as `GOALSMITH_WORKER__MODEL`. Partial overrides keep the other defaults.
- Retries and timeouts use the provider SDK's built-in exponential backoff (`max_retries`, `timeout`), so no second retry layer is added.
- Structured output goes through `generate_structured(model, Schema, messages)`, which always returns a validated Pydantic object.
- Tests use `tests/fixtures/fake_llm.FakeChatModel`, which replays scripted replies, including tool calls.
- `get_embeddings()` moves to M3, where embeddings are first used. That avoids pulling in `fastembed` and its model download before they're needed.

## Consequences
- Adding a provider means one builder function plus `register_provider("name", builder)`.
- Model choice and cost tuning happen in `.env`, so evals can compare models without code changes.
- Option 3 (LiteLLM) stays possible later as just another registered provider.
- `ChatAnthropic` is built with `model_validate({...})` because its field aliases confuse mypy strict mode.
