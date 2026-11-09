# Diagnosing a bad run from its trace: unsupported advice (M5)

**Run:** eval task `docs-error-code`, prompt `agent_system@v1`, worker `claude-sonnet-5-5`, 2026-11-10.
**Trace:** `f10e0c427ccb8033a3bfb9e4a9284386` (local Langfuse: http://localhost:3000/project/goalsmith/traces/f10e0c427ccb8033a3bfb9e4a9284386)

![The failing trace in Langfuse](../images/m5-bad-run-trace.png)

## Symptom
The smoke eval passed 5 of 6 tasks. `docs-error-code` passed every deterministic check:
- the answer contained "2.4 GHz" and "Wi-Fi";
- it cited a source;
- it called `search_docs`.

The judge scored it 0.67, failing one criterion:

> Every fact in the final answer is supported by a tool result shown in the record; nothing is invented.
> → *"the suggestion that a recurring E-1042 'may need to be paired again' has no source in any tool result"*

## Reading the trace
The trace tree, as Langfuse shows it:

```
eval.task                         (span)
└─ agent.run                      (agent)    prompt agent_system@v1
   ├─ model claude-sonnet-5-5     (generation) → calls search_docs + recall
   ├─ tool search_docs            (tool)     query "Beacon sensor error E-1042"
   ├─ tool recall                 (tool)     "No saved memories match."
   ├─ model claude-sonnet-5-5     (generation) → final answer
   └─ judge                       (evaluator) score 0.67
```

1. **Was retrieval wrong?** No. The `search_docs` output starts with `[beacon-faq.md#3]`: *"E-1042 means the sensor cannot reach the Wi-Fi network… check that the network is not 5 GHz only, then restart the sensor."* That is the correct passage, at rank 1.
2. **Was the answer wrong?** Partly. The second generation's output gives the correct meaning and fix, then adds:
   > *If the error comes back, the sensor may need to be paired again… [beacon-faq.md#1] The FAQ doesn't list this as a fix for E-1042, so treat it as a last resort.*

   `beacon-faq.md#1` is the installation section. The model stitched an unrelated passage into a new "fix", and admitted the source doesn't say so.
3. **Why?** The system prompt (`agent_system@v1`, shown on every span) asked for citations, but never said to stay within what the sources support. A helpful model filled the gap.

The answer was not caught by the checks, because the true facts were all there. It was caught by the judge, and the trace showed in under a minute that the fault was in generation, not retrieval.

## Fix
`agent_system` v2 adds one rule:

> State only what your tool results support. If the sources do not cover part of the question, say so; do not fill the gap with your own suggestions.

## Result

| Prompt | `docs-error-code` | Full suite (15 tasks) |
|---|---|---|
| v1 | failed (judge 0.67) | not run |
| v2 | passed 3 of 3 runs (judge 1.00) | 13/15* |

\* The two failures were mistakes in the eval, not the agent. Both tasks passed once the eval was fixed. See the [AI review log](../ai-review-log.md).

Because traces record `prompt_version`, Langfuse can filter runs on v1 and v2 and compare them.
