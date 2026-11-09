# AI review log

AI-generated suggestions that were rejected or changed, and why. This keeps AI-assisted work honest and gives concrete examples for interviews.

| Date | Milestone | Suggestion | What I did | Why |
|---|---|---|---|---|
| 2026-11-09 | M3 | Plan proposed `bge-reranker-base` as the reranker | Default to `ms-marco-MiniLM-L-6-v2`; bge stays a config option | 1 GB vs 80 MB, and MiniLM already scores MRR 1.000 on the eval (ADR 0005) |
| 2026-11-09 | M3 | AI-written BM25 test compared the gain of 18 extra repeats with the gain of 1 | Rewrote it to compare consecutive repeats | The test was wrong, not the code: BM25 saturation means each *single* repeat adds less |
| 2026-11-09 | M3 | AI-written CLI printed `[handbook.md#0]` inside Rich markup | Escaped the brackets | Rich read the source id as a style tag and hid it; the CLI test caught it |
| 2026-11-09 | M3 | Ruff's autofix turned the stopword string into a 46-line list | Kept a compact string, split at import | Same behaviour, far easier to read and edit |
| 2026-11-09 | M4 | AI-written CLI caught `click.exceptions.Abort` at the approval prompt | Switched to `typer.Abort` and a validated prompt; no direct `click` import | Typer 0.28 bundles its own click, so the `except` never matched. `click` was not even a declared dependency, only a transitive one from huggingface-hub |
| 2026-11-09 | M4 | AI-written integration tests used the developer database | Tests now create and use a separate `goalsmith_test` database | A test created the memory table with 64-dim fake vectors, and the next live run failed with "expected 64 dimensions, not 384" |
| 2026-11-09 | M4 | No vector-size check for the memory store | Added a startup check against `pg_attribute.atttypmod` with a clear error | The pgvector size is fixed at table creation; the failure otherwise appears only at the first `remember` |
| 2026-11-09 | M4 | Default checkpoint serializer | Allowlisted `BudgetUsage` in `core/serde.py` | The live run logged "Deserializing unregistered type"; LangGraph will block unlisted types |
