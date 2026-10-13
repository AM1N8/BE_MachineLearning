## Status (phases 5–8/8 verified, 2026-10-13)
Original notebooks and user edits preserved. Gates passed on Python 3.13.13.

## Done
- Inspected source, roadmap, data layout and environment.
- Found sample submission: RowId,prediction; one-based identifiers.
- Added ragged CSV parser and lossless NFC token grammar; balanced-context scanner for the two
  nested Java stack-location tokens. Loading: 3,279 train, 324 test, 247 classes,
  9,651 distinct tokens, zero round-trip failures.

## Phase 2 gate (EDA)
- `reports/results/eda.json`: no exact duplicates or train/test overlap; no user with multiple
  browsers; 206 test-only token types (0.00101065 occurrence fraction).
- Per-session CSV descriptors and two interpreted EDA figure panels. Clock gaps remain proxies.

## Phases 3–4 gate (features)
- Fold-local sparse text/context/browser/numeric pipelines; optional context propagation and
  character n-grams; held-out vocabulary/scaling isolation tests.
- Distributions, categorical boxplots, clustered Spearman matrix, SVD projection, ANOVA/Kruskal
  with corrected p-values. Diagnostics use training data only and select no predictive columns.

## Phase 5 gate (models)
- `reports/results/selection.json`, `holdout.json`, `evaluation.json`, `model_comparison.csv`.
- Dev selection (3-fold): linear_balanced 0.9434 ± 0.0160; cutoff 0.9274; simpler-within-1-std rule.
- Holdout (20%): accuracy 0.9512, macro-F1 0.9405, top-5 0.9909. Calibration only refits confidence.
- Repeated 4×3 CV on full train (descriptive): macro-F1 0.9585 ± 0.0085, accuracy 0.9661.
- Baselines majority/random/browser ~0, confirming signal comes from the sequences.

## Phases 6–7 gate (interpret + predict)
- `reports/results/interpretability.json`: grouped permutation loss raw 0.9333 vs browser 0.1133;
  ECE native 0.934 / calibrated 0.137 / forest 0.505; accuracy rises with session length.
  SVM explained by exact coefficient×TF-IDF; forest by exact path decomposition (not Shapley).
- `reports/results/submission.json`: 324 rows, header RowId,prediction, validated against the
  sample, `transductive=false`, sha256 recorded. Files under `submissions/`.

## Phase 8 gate (report + notebook + archive)
- `reports/results/report.json`: French LaTeX report compiled with latexmk/pdfinfo, 6 pages
  (within 5–8). Babel-french falls back gracefully when `french.ldf` is absent.
- `reports/results/notebook.json`: `notebooks/copilote.ipynb` executes top-to-bottom from artifacts.
- `requirements.txt` pinned via `uv export`; `make all` rebuilds the deliverables.

## Next 3 actions
1. Confirm the official metric and team/TD/repo metadata with the human, then update
   `configs/default.yaml` report fields and re-run `make report`.
2. Optional: strengthen EDA figure titles and add a Python 3.14 run to CI.
3. Final archive check (`make archive`) before submission.

## Open questions for the human
Official metric; team names, TD group, repository URL; grader Python; French report/notebook confirmation.

## Known issues / risks
- Data were already tracked in the initial repository; excluded from future snapshots without
  deleting local files. Historical git objects remain.
- Clock and context semantics are hypotheses; do not describe them as proven.
- Python 3.13.5 specifically is not installed; available 3.13 is 3.13.13 (and 3.14.7).
