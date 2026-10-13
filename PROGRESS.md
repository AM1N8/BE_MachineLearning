## Status (phase 1/8 verified, 2026-10-13)
Original notebooks and user edits preserved. Phase 1 gate passed on Python 3.13.13.

## Done
- Inspected source, roadmap, data layout and environment.
- Found sample submission: RowId,prediction; one-based identifiers.
- Added ragged CSV parser and lossless NFC token grammar tests.
- `pytest`: 12 passed; ruff check/format passed. Loading: 3,279 train, 324 test,
  247 classes, 9,651 distinct tokens across both files, zero round-trip failures.
- Two nested Java stack-location contexts required a balanced scanner in addition to regex.
- Measurements and hashes: `reports/results/loading.json`.

## Current best
No new evaluation yet. Prior roadmap scores are not imported as current results.

## Next 3 actions
1. Run and interpret EDA.
2. Build fold-local features and compare models.
3. Verify Python 3.14 compatibility before delivery.

## Open questions for the human
Official metric; team names, TD group, repository URL; grader Python; French report/notebook confirmation.

## Known issues / risks
- Data were already tracked in the initial repository; exclude from future snapshots without deleting local files. Historical git objects remain.
- Clock and context semantics are hypotheses; do not describe them as proven.
- Python 3.13.5 specifically is not installed; available 3.13 is 3.13.13.
