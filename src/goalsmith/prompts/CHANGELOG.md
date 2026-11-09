# Prompt changelog

Bump a prompt's `version` whenever its text changes, and add a line here.
Traces record `prompt_name`, `prompt_version` and `prompt_sha`, so you can filter
runs and eval results by prompt version in Langfuse.

## agent_system
- v1 (M5): moved from `DEFAULT_SYSTEM_PROMPT` in harness/loop.py, text unchanged.
- v2 (M5): added "State only what your tool results support...". Eval task
  `docs-error-code` failed on v1: the answer added an unsupported fix ("may need to be
  paired again"). See docs/diagnosis/m5-unsupported-advice.md.

## judge
- v1 (M5): first version, for golden-task evals.
