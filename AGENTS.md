# Operating rules

1. Build a complete, graded, reproducible data-science project: complete chain, motivated choices, competitive validation.
2. Never fabricate numbers. Report metrics must come from `reports/results/*.json|csv`; explicitly mark unrun work.
3. No leakage: fit vocabulary, scaling, selection, encoding and models inside training folds. Test sessions are only transformed; any transduction must be explicitly optional and labelled.
4. Seed 42, pinned dependencies, deterministic splits, `make all` rebuilds the deliverables.
5. Raw data are read-only; exclude data from git and archives. Explain placement in README.
6. Work in verified increments. At each phase run tests and pipeline, update PROGRESS.md, commit meaningful verified changes. Do not advance past failed gates.
7. Ask only for unknowable information; document other decisions in `docs/DECISIONS.md` (context, options, choice, rationale).
8. Honesty over score: report negative results.
9. Report in French; code, docstrings and commit messages in English; short bilingual README.
10. Never probe Kaggle labels or tune on the leaderboard. Select using training-data CV only.

## Commands

`uv sync --python 3.13`; `make test`; `make lint`; `make eda`; `make fast`; `make all`.
Read PROMPT.md and PROGRESS.md before continuing. Preserve existing notebooks and user changes.
