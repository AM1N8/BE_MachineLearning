# Validation and evidence

## Historical results of the submitted method

These are the recorded original experiment results, not fabricated results of a fresh execution of this folder.

| Evaluation | Macro-F1 mean ± fold SD | Micro-F1 mean | Weighted-F1 mean |
|---|---:|---:|---:|
| `crop128_noidf`, 5 folds × 2 repeats | 0.96974 ± 0.00697 | 0.97210 | 0.97012 |
| `fusion_context`, 5 folds × 2 repeats | 0.96765 ± 0.00677 | 0.97316 | 0.97104 |
| `reference128`, 5 folds × 2 repeats | 0.96890 ± 0.01010 | 0.97393 | 0.97184 |
| B recipe, 5 outer × 3 inner nested CV | 0.97143 ± 0.00845 | 0.97530 | 0.97382 |

The user reported **0.999 public leaderboard score for B** and **0.996 for A**. These observations are not CV estimates or proof of private-leaderboard performance. The exact leaderboard averaging convention remains unverified; macro-F1 is the project's primary selection metric, and micro/weighted F1 are always logged.

## Repeated CV

The code uses `StratifiedKFold(5, shuffle=True)` with seeds 137 and 271. Each fitting partition learns its own vocabularies, IDF, crop weights, classifier and browser support. Validation sessions are transformed only after fitting. OOF probabilities are saved per repeat and session.

Some users have four sessions, so a five-fold validation partition can omit a class. The expected sklearn warning is retained. Macro-F1 follows sklearn's observed true/predicted label union, matching the original experiment; it does not force all 247 classes into every fold's denominator.

Sample standard deviations use `ddof=1`. Repeated folds overlap and are correlated; mean ± SD is not a confidence interval. Macro-F1 on repeat-averaged OOF probabilities measures a different estimator and is labeled separately.

## Weight optimization versus nested evaluation

All-OOF weight fitting reuses the labels on which its score is computed. The original final-weight apparent pooled macro-F1 was **0.9772173831**. This is an optimization diagnostic, not the headline validation score.

Nested validation uses five outer folds with seed 137. In each outer training partition:

1. Three inner folds generate OOF predictions from independently fitted pipelines.
2. Weights are optimized using those inner predictions and labels only.
3. The members are refitted on the outer training partition.
4. The weighted blend is scored on the untouched outer validation partition.

Inner splits use seed `137 + outer_fold`; model seeds are `137 + inner_fold`. Outer model seeds are `137 + outer_fold`, matching the original blend evaluation.

Model identities had already been shortlisted using exploratory CV, so nested evaluation removes weight-fitting leakage but not all model-selection optimism. The standalone package fixes that three-model shortlist rather than reimplementing the earlier broad search.

The original conservative selection rule retained A because the blend gain over the comparable single model did not exceed the single model's fold SD. B was retained as the diverse alternative and later achieved the user-reported higher public score. The failed stacker and discarded experimental models are not required to reproduce B and are not part of this package.

## Duplicate/group sensitivity: original audit

The original data audit found no exact duplicate sessions. At cosine threshold 0.95 on stateless binary action 1–2-gram hashing, there were two same-label near-duplicate pairs and no conflicting pair labels.

A five-fold group-aware audit of `crop128_noidf` scored macro-F1 **0.95812 ± 0.00774**, micro-F1 **0.96645**, and weighted-F1 **0.96367**. Group folds contained 216–224 true classes, versus 244–247 in the stratified first repeat. The score difference is not a causal estimate of duplicate leakage. This audit is historical context; the standalone `cv` and `nested` stages reproduce the B modeling workflow, not the entire original search and audit suite.

## Practical checks

- Unit tests cover parsing, state carry/reset, crop boundaries, session weight conservation, fold-local vocabulary/browser support, soft voting and submission checks.
- `run.py cv --repeats 1 --folds 1` is a real first-fold smoke run, explicitly incomplete.
- Complete repeated CV saves `[2, n_train, n_classes]` OOF arrays and per-class/error/confusion CSVs.
- `run.py nested --folds 1` is a nested smoke run and writes `nested_partial.json`; it does not publish a complete aggregate.
- `run.py verify --reference ... --reference-artifacts ...` checks the final recreation against the original saved probabilities and submission.

Model bundles are generated locally rather than copied from the old package. Fingerprints include input hashes, source hashes and configuration; changed inputs/code/configuration require a fresh output directory.
