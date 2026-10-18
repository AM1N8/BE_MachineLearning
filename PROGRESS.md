## Authorized experimental export (2026-10-17)
- User explicitly requested the exact unpromoted contextual-signatures challenger.
- Refit all 24 original folds: predictions exactly reproduced; frozen source/data
  hashes and disjoint folds verified. No new tuning or test-dependent training.
- Trained on all 3279 labels; exported context_signatures_experimental.csv (324 rows).
  One change: RowId 237 czi -> pgm. Baseline CSV and model hashes unchanged.
- Same-process scores and fresh-process predictions reload exactly; sample order,
  columns, labels and missing values checked. 55 tests and lint passed.
- Report: reports/context_signatures_experimental.md. Experimental only: unchanged
  promotion criteria still fail; no automatic Kaggle submission.

## Phase 2 completed gate (2026-10-17)
- Executed 136 outer model evaluations: five behavioral ablations plus nested top-2
  reranking on historical folds; four candidates confirmed on seeds 43 and 44.
- Contextual signatures improved macro-F1/accuracy on all three protocols but failed
  corrected lower-confidence gates. Reranking corrected 14 and damaged 16 repeated
  predictions. KEEP BASELINE; no phase2_challenger.csv generated.
- Results: reports/results/phase2/a1d191a9b558. Reports phase2_research.md and
  phase2_error_analysis.md include paired metrics, persistent errors and length strata.
- `make phase2` completed, reusing fingerprinted fits and regenerating audit/reports;
  24 internal split audits passed and a fresh model reload reproduced all 324 baseline
  predictions. Original best_verified.csv SHA-256 preserved.
- Final checks: 55 tests passed (10 existing SQLite ResourceWarnings), lint passed
  for 39 files. Overall pytest coverage 54%; phase2_models.py 100%; orchestration and
  report generation verified end-to-end rather than covered by pytest.
- Inspected local competition evaluation/rules and both READMEs. F1 averaging remains unspecified.
- Training-only audit: no explicit chronology/group ID; weak row/first-clock correlation,
  no multi-browser users. Excluded empty trigram sets from proximity comparisons after
  inspecting the shortest persistent errors (0–2 events); these are not duplicate evidence.
- Persistent 78 sessions exported with nearest same/other-user similarity and context coverage.
- Five fixed context-aware ablations and a nested global-aware reranker implemented;
  eight new boundary/vocabulary/OOF isolation tests passed. Protocol frozen in phase2_research.md.

## Improvement research gate (2026-10-17)
- User reports public score 0.980; exact Kaggle F1 averaging still inaccessible/unverified.
- Added audit, fold-local error analysis and five fixed challengers in improvement.py
  and improvement_boundaries.py; all eight historical baseline folds reproduced exactly.
- Five challengers evaluated on paired 4×2 splits; three and baseline also evaluated
  on seed 43 (4×2). No candidate passed the unchanged promotion gate.
- Event-only mask gained F1 on historical folds but regressed on seed 43. Rare binary
  features slightly improved accuracy on both seeds without credible macro-F1 gain.
- Decision KEEP BASELINE; best_verified.csv preserved, no challenger declared verified.
- Results: reports/results/improvement/4e7e8c2ad9a6; report leaderboard_improvement.md.
- Final verification: 47 tests passed (16 existing SQLite ResourceWarnings), lint passed
  for 34 files, git diff --check passed. Fresh-process baseline reload reproduced all
  324 predictions and the original CSV hash; final_verification.json records the gate.

## Status (recommended Kaggle submission verified, 2026-10-17)
Original notebooks and user edits preserved. Gates passed on Python 3.13.13.

## Final submission gate (2026-10-17)
- Read the full project requirements, existing experiments, evaluation description and
  sample schema. Finished the existing `extreme` campaign with its original fingerprint,
  candidates, split protocol and promotion rule; predictive sources/config/lock unchanged.
- Completed outer fold 7, the 4×2 nested comparison, and final inner selection. An interrupted
  final Optuna trial was resumed with sampler replay and cached folds, preserving its budget.
- `reports/results/extreme/1ba1ee389f73/nested_validation.json`: reference macro-F1
  0.961251 ± 0.010938, accuracy 0.969350; nested-selected ensemble macro-F1 0.961320.
  Paired gain +0.000069, corrected 95% lower bound −0.002336, only 3/8 improved folds,
  repeat gains −0.000337 / +0.000475. Predeclared promotion gate failed: reference retained.
- `submissions/best_verified.csv` is the recommended upload, identical to `competitive.csv`.
  Exactly 324 rows, `RowId,prediction`, IDs 1–324 in sample order; no unknown/missing labels.
  SHA-256: `ea7174fbb9a23453a84aa3716f43091b96e4e658187e2c1bca655070e251595d`.
- Added `make submission` (also included in `make all`) to refit the locked winner on all
  3,279 training sessions and require the recorded CSV hash. Saved
  `artifacts/best_verified.joblib`; a fresh Python process reproduced all predictions.
  Provenance: `reports/results/best_submission.json`, `submissions/best_verified.json`.
- Generated `extreme_candidate.csv` for traceability; it was not promoted. Report:
  `reports/best_submission.md`. The supplied evaluation says mean F1; macro averaging is
  the local interpretation, exact Kaggle implementation unconfirmed. Kaggle score unknown.
- Verified nested/predict pipelines, `make submission`, `make test` (44 passed), and
  fresh-process model/schema/hash checks; `make lint` (31 formatted files) and
  `git diff --check` passed. Existing Optuna/SQLite ResourceWarnings remain.
  Full historical report/notebook pipeline was not rerun during this finalization.

## Second-round search gate (2026-10-15)
- Added `similarity.py`, `advanced.py`, `configs/advanced.yaml`, and nine isolation/geometry/
  serialization tests. `make advanced` reruns the search with fingerprinted caches and is
  included in `make all`. No dependency changes.
- Evaluated 33 candidates: 31 new variants and two existing references. Families: per-class
  nearest sessions, user centroids, square-root TF-IDF, cosine polynomial kernel SVC/ridge,
  fixed-weight hybrids, BM25, supervised user-IDF and class-entropy feature weighting.
- All learned representations, class weights/profiles and browser support remain fold-local.
  Held-out transforms cannot modify vocabulary/IDF/training matrices; supervised weighting
  has an explicit fit/transform consistency test. Sparse token matrices remain sparse.
- `reports/results/advanced/comparison.csv`: development winner `hellinger_svm_c10`,
  accuracy 0.959203 and macro-F1 0.948678, versus previous 0.958824 / 0.945621.
  Choice frozen before final benchmarks under the existing one-fold-std simplicity rule.
- `reports/results/advanced/validation.json`: repeated accuracy 0.967673 / macro-F1 0.959197,
  versus previous 0.968487 / 0.960052. Mean paired accuracy change -0.000814; improves only
  3/12 folds. Historical holdout accuracy 0.961890, macro-F1 0.945851. No robust gain established.
- The initial similarity-only stage retained the previous model; a second development-only
  expansion tested weighting variants. This is adaptive search, not an independent test;
  repeated CV and the already-used reserve remain descriptive, never tuning objectives.
- `submissions/advanced.csv`: 324 validated predictions from the development-selected model;
  persisted estimator reload reproduced every prediction. Existing competitive submission
  retained. No leaderboard evaluation or tuning; Kaggle score remains unknown.
- Verified `make lint` (25 formatted files), `make test` (37 passed), `make advanced` including
  cache reuse, and `git diff --check`. Full historical report/notebook pipeline not rerun.
- Generated French results: `reports/advanced.md`. Negative results retained; D11 explains
  model families, selection and the failure to confirm the development improvement.

## Competitive extension gate (2026-10-15)
- Added isolated training-only search: `src/copilote/competitive.py`, 52 declared variants in
  `configs/competitive.yaml`; reproducible with `make competitive` (also included in `make all`).
- All 52 candidates evaluated on identical development folds (2,623 sessions, 3 folds).
  Train-only vocabulary/IDF, augmentation, class/browser support, and NB weights.
  Search, selection, repeated validation and submission have matching predictive signatures.
- Frozen choice: `browser_mask_c3`, balanced LinearSVC C=3, raw event TF-IDF 1–2 grams and
  browser, with fold-learned browser support restriction. Best raw development mean was
  `experts_c10`; its small improvement does not justify complexity under the declared rule.
- `reports/results/competitive/validation.json`: repeated 4×3 accuracy 0.968487 (baseline
  0.966148), macro-F1 0.960052 (baseline 0.958467). Paired accuracy gain 0.002339,
  improvement in 9/12 folds; no claim of statistical significance or independent validation.
- Historical holdout: accuracy 0.960366, macro-F1 0.952335, 656 sessions. Reused benchmark,
  not a fresh untouched test; no model selection on these results.
- Negative results recorded for nonlinear kernels, NB-SVM, longer n-grams, numeric/style
  views, alternate regularization, binary/count representations and within-fold augmentation.
- `submissions/competitive.csv`: 324 rows, sample schema/order checked, no missing/unknown
  labels, train-only final fit, `transductive=false`, SHA-256 and provenance recorded.
  Persisted model reloaded in a separate Python process and reproduced all 324 predictions.
- `reports/competitive.md` generated from JSON/CSV; error diagnostics cover development only.
  Raw excerpts are confined to ignored `artifacts/competitive/`, absent from report results.
- Gates passed: `make competitive`, `make lint` (22 files), `make test` (28 passed).
  Original full report/notebook pipeline was not rerun during this extension.
- Kaggle score and official metric remain unknown. No leaderboard-driven tuning or submission
  through the Kaggle API. Local metrics do not establish a score above 0.993.

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

## Phase 4 gate A (forensics) — 2026-10-17
- New module `src/copilote/phase4_forensics.py` (lint clean, 55 tests pass); artifacts in
  `reports/results/phase4/d524c6ba3719/` (manifest signed, seed 42); French report
  `reports/phase4_dataset_forensics.md`.
- Key facts: gaps ∈ {0,5,10,15} s only; zero clock resets; 99.9% sessions have every 5 s tick
  active; 2,270,247 train events; suffix « 1 » = 15.64% of events, on 31/36 actions, per-action
  rates 0.11–0.57; 8.7% of suffix events carry explicit groups (falsifies "same context" rule).
- Timing scalars: KW H up to 1,313 (dwell_top_fraction) but LR probe alone 0.060 macro-F1 →
  complementary sparse fusion only (tempo family planned in Phase C).
- Tokens: 2,961/7,087 user-exclusive in-sample; 47.3% of sessions contain one; 7 users none.
- No train/test shift (KS all p > 0.05; suffix_fraction p 0.064 borderline).
- Next: Phase B error analysis on phase2 OOF artifacts + this timing table.

## Phase 4 gate B (error analysis) — 2026-10-17
- New module `src/copilote/phase4_errors.py` (lint clean, 55 tests pass); artifacts in
  `reports/results/phase4/b2d89ec23226/` (manifest signed, seed 42); French report
  `reports/phase4_error_analysis.md`. Fold-local exclusive tokens recomputed per fold
  (train half only); splits verified identical to phase2/extreme.
- Key facts: errors vs correct medians — length 130 vs 353, OOV 0.0194 vs 0.0026,
  suffix_fraction 0.0204 vs 0.0918 (4.5x), dwell_top_fraction 0.571 vs 0.436.
- Negative result (honest): snap rule (fold-local exclusive-token override) net-negative in
  all 6 variants; worst breaks 950/6357 correct (macro-F1 0.818); best 8 fixed / 61 broken.
  Direction 4 (hard residual correction) falsified; signal anti-correlated with errors
  (correct sessions have MORE truth-exclusive support: 26.9% vs 17.9%).
- Taxonomy of 201 errors: short 6, exclusive_missed 36, top2_recoverable 86, close_margin 60,
  confident_wrong 13. Always-wrong 78: 14 snap_fixable, 32 rerankable, 32 hard; perfect
  top-2 reranker ceiling 41/78.
- Confusion pairs: 113 distinct, all top-10 same-browser; flj->skm 11 errors (9 with
  truth-exclusive support, 7 top-2); azn has only 4 train sessions; ngs->slq never top-2.
- Signatures delta recomputed on historical protocol: 16 corrected / 6 introduced
  (9/16 corrected had truth-exclusive support, 0/6 introduced).
- Phase C priorities: tempo/suffix sparse fusion first (strongest new discriminators),
  then margin fusion baseline+signatures; sequence/markov stays closed (phase 1 negative).

## Phase 4 gates C+D (experiments and validation) — 2026-10-18
- New module `src/copilote/phase4_candidates.py` (ruff clean; 9 tests in `tests/test_phase4.py`,
  64 pass total). Stages `research|confirm|seed44`, manifest-signed dirs, resumable,
  submissions asserted unchanged at the end of every stage.
- Research (seed 42, `85b76c4630ba/comparison.csv`): tempo family mostly weak/negative
  (dwell −0.0019, scalars −0.0006), fusion ≈ baseline as predicted; overlap analysis found
  signatures/burst corrections disjoint with zero antagonism; signatures weight ladder
  0.2→1.0 unimodal (peak w06 +0.0040, collapse at w10 −0.0044); hybrids sig_burst_w05
  (+0.0045) / sig_burst_w06 (+0.0046). Nobody passes lower95>0.
- Confirm (seed 43, `8492d196609f/confirmation.csv`): weight-ladder gains wash out
  (w06 +0.0007); the burst block replicates (sig_burst 8/8 folds, lower95 −0.0006,
  best ever). Seed 44 (`951bd7da5015/seed44_comparison.csv`): sig_burst_w05 +0.0034,
  only candidate with both repeats positive on all three protocols; dominates signatures
  everywhere. Still nobody passes the promotion criterion.
- Pooled secondary evidence (24 folds, post-hoc, `85b76c4630ba/pooled_evidence.json`):
  sig_burst_w05 mean +0.003419, lower95 +0.001842, 19/24 wins.
- French report `reports/phase4_experiment_results.md`; decision record D14 in
  docs/DECISIONS.md.

## Phase 4 gates E+F (final candidate and submission) — 2026-10-18
- New module `src/copilote/phase4_challenger.py` (ruff clean): `reproduce` re-fits all 24
  challenger folds (research/confirm/seed44) — predictions reproduced exactly; `export`
  fits sig_burst_w05 on the 3 279 labelled sessions only and writes
  `submissions/phase4_challenger.csv` (RowId 1..324 aligned with the sample, joblib reload
  exact, transductive=false, automatic_submission=false).
- Challenger export `reports/results/phase4/c8a3d88dbf5c/`: 3/324 changed rows vs
  best_verified (125 zro→bgn, 237 czi→pgm, 290 usy→vsb), 2/324 vs context_signatures
  (125, 290); protected submissions verified unchanged (hashes before=after).
- Promotion FAILED (lower95 negative on all three protocols) → champion stays
  `submissions/best_verified.csv` (Kaggle 0.980); challenger exported as
  EXPERIMENTAL_NOT_PROMOTED, same discipline as phase 3. Recommendation and bounded
  risk analysis in French report `reports/phase4_final_recommendation.md`.
