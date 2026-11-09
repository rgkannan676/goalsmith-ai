# AI review log

AI-generated suggestions that were rejected or changed, and why. This keeps AI-assisted work honest and gives concrete examples for interviews.

| Date | Milestone | Suggestion | What I did | Why |
|---|---|---|---|---|
| 2026-11-09 | M3 | Plan proposed `bge-reranker-base` as the reranker | Default to `ms-marco-MiniLM-L-6-v2`; bge stays a config option | 1 GB vs 80 MB, and MiniLM already scores MRR 1.000 on the eval (ADR 0005) |
| 2026-11-09 | M3 | AI-written BM25 test compared the gain of 18 extra repeats with the gain of 1 | Rewrote it to compare consecutive repeats | The test was wrong, not the code: BM25 saturation means each *single* repeat adds less |
| 2026-11-09 | M3 | AI-written CLI printed `[handbook.md#0]` inside Rich markup | Escaped the brackets | Rich read the source id as a style tag and hid it; the CLI test caught it |
| 2026-11-09 | M3 | Ruff's autofix turned the stopword string into a 46-line list | Kept a compact string, split at import | Same behaviour, far easier to read and edit |
