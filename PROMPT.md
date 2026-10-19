# MISSION PROMPT — Copilote user identification (MOD 7.2, BE CS1)

> Paste this whole file as the first message to the coding agent (OpenCode / Codex).
> Also copy section **0. Operating rules** into `AGENTS.md` at the repo root so every
> new session re-reads it. Keep `PROGRESS.md` (section 11) up to date: you will lose
> context between sessions, the repo must not.

---

## 0. Operating rules (non-negotiable)

1. **You are building a complete, graded, reproducible data-science project**, not a demo.
   Optimise for: (a) a complete processing chain, (b) every choice explained and motivated,
   (c) a strong score in a competitive Kaggle setting. These are exactly the three grading criteria.
2. **Never fabricate numbers.** Every figure/metric in the report must be produced by code and
   read from an artifact in `reports/results/*.json|csv`. If something was not run, say so.
3. **No leakage.** Any fitted object (TF-IDF vocabulary, scaler, selector, encoder, imputation,
   pseudo-labels) is fit **inside** the CV training fold only. Test data is only ever transformed
   (the only exception is an explicitly labelled, optional transductive step, see 7.6).
4. **Reproducibility:** global seed (`SEED=42`), pinned dependencies, deterministic splits, one
   command to rebuild everything (`make all` or `python -m copilote.run_all`).
5. **Data never goes in git / the archive** (`data/` is git-ignored; README explains where to put
   the files). Raw CSVs are read-only.
6. **Work in small verified increments.** After each phase: run tests, run the pipeline, commit
   with a meaningful message, update `PROGRESS.md`. Do not move on while a phase gate (section 10) fails.
7. **Ask the human only for things you cannot determine** (list in section 12). Otherwise decide,
   document the decision in `docs/DECISIONS.md` (context → options → choice → why), and continue.
8. **Honesty over score:** report negative results (ideas that did not help). They count as
   "capacité à expliquer et motiver" and are valuable in the report.
9. Language: **report in French** (course is French). Code identifiers, docstrings, commit
   messages in English. README: short French + English.
10. Do not use the Kaggle API/leaderboard to probe test labels, and do not tune on the public
    leaderboard. Model selection is done on cross-validation only.

---

## 1. Context

- Course: **MOD 7.2 — Introduction à la science des données**, BE sessions 1–3 (École Centrale de Lyon).
  Teachers: Julien Velcin, Antoine Becquet. Group of 3 students.
- Task: from **usage traces of the software Copilote** (Infologic, agro-food ERP), **identify which
  user produced a session** (multi-class classification, ~247 classes). Kaggle challenge:
  `https://www.kaggle.com/competitions/qui-utilise-mon-appli-challenge/` (private teams per
  TD group are linked by the teachers; the human handles the upload).
- **Deadline: 9 Nov 2026 evening.** Deliverables:
  1. **Python code** re-executable elsewhere (target Python **3.14**, local kernel is 3.13.5) +
     `requirements.txt`. Structured, commented, README. A succinct, very readable notebook is
     allowed *in addition*, but the main code is a package/scripts.
  2. **PDF report, 5 to 8 pages** (hard limits): problem, approach, **features chosen with short
     justification**, classifiers, **results with explicit metrics**, most influential variables,
     short discussion, conclusion. Code URL in the report if hosted on git.
  3. **Predictions on the test set**, uploaded to Moodle **and** the Kaggle group leaderboard.
- Grading: complete pipeline · ability to explain/motivate each treatment · efficiency in a
  competitive setting. Every team member must be able to present *everything* → you must also
  produce `docs/DEFENSE.md` (section 9).
- Course-suggested pipeline (follow this order and name the notebook/report sections accordingly):
  understand problem & data → load raw data → first analysis → feature construction → simple
  statistics/plots on features → classifier → evaluation → discussion (incl. interpretability).
- Course-suggested models to compare: logistic regression, SVM (linear/poly kernels), decision
  tree, ensembles (Random Forest, XGBoost), simple MLP, regularised sparse models. Use
  **scikit-learn** as primary library. Minimum metrics: **accuracy** and **F-measure (F1)**;
  more are encouraged.
- The course's interpretability questions (answer them with a Random Forest, and ideally also with
  the best model): (1) confidence in the prediction for a given row; (2) for one row, which factors
  mattered most and how; (3) which columns are the strongest predictors; (4) which columns are
  redundant with each other.
- Mandatory-looking stub tasks from the course notebook to cover: inspect distinct browsers,
  per-user browser counts, distribution of Y, outlier inspection, categorical encoding of Y
  (`pd.Categorical`), OHE of browser, 3 feature plots, boxplots of one feature for chosen classes
  (e.g. classes `[0, 15, 152, 199]` with feature `actions_means`).

---

## 2. Data — what I measured (trust but verify in `01_eda`)

Files (in `data/raw/`): `train.csv` (3,279 lines, 70 MB), `test.csv` (324 lines, 6.6 MB).
No sample-submission file was provided (see section 12).

**Row format.** One line = one session.
- train: `user_id,browser,action_1,action_2,...`
- test: `browser,action_1,action_2,...` (no label, no id column → identify rows by 0-based line order)
- Variable number of fields per line (2 → ~14,470 actions). **Do not** load with a wide
  `pd.read_csv(names=range(max_len))` (14,470 mostly-NaN columns): parse into **list-of-tokens per
  session** (ragged) and build sparse matrices.
- ⚠ In the files I received, **every line is wrapped in one pair of double quotes** and there are
  no inner quotes. Parser must strip an enclosing quote pair if present and otherwise work on
  un-quoted files (the originals may differ). Use `encoding="utf-8"`.
- Accented characters + two apostrophe variants exist (`'` and `’`, e.g. `Sélection d’un onglet`):
  **normalise carefully** (NFC) but do not merge distinct tokens silently.

**Labels.** 247 distinct users (3-letter codes such as `skm`, `slq`). Class sizes: min 4,
median 12, max 75 (`skm` 75, `slq` 71, `cjr` 46, `flj` 42, `hjs` 37) → **long-tailed, imbalanced.**
Smallest classes have only 4 sessions → stratified CV with ≤4 folds (or repeated stratified
folds); in tiny-class edge cases handle fold emptiness explicitly. Every class in a random 20 %
hold-out was seen in training. Test has 324 sessions → a few correct/incorrect predictions move
the score noticeably: **prefer repeated CV and report mean ± std**.

**Browser** (4 values): train Firefox 1466, Google Chrome 1339, Microsoft Edge 451, Opera 23
(test: 147/133/42/2). A user may use several browsers.

**Session length** (tokens incl. time markers): train min 2, p25 ≈ 192, median ≈ 423, p75 ≈ 1019,
p95 ≈ 2900, p99 ≈ 5660, max 14,468; test min 13, median 450, p95 ≈ 2780, max 7,725.
⚠ Heavy tail. The course says "remove outliers (you can delete them for now)" — **do not blindly
drop long training sessions**: the test set also contains long sessions that must be predicted.
Use log transforms, per-length normalisation, robust scaling, and (optionally) truncation/windowing;
evaluate whether removing extreme rows helps in CV and document it.

**Time markers.** Tokens `tNN` (regex `^t\d+$`) = 5-second ticks; values are cumulative
seconds (max seen `t11790` ≈ 3 h 16). Gaps between consecutive markers encode idle time. They are
**not actions** → separate them from the action stream, but derive timing features from them
(duration, actions per minute, idle-gap stats, burstiness).

**Action token grammar** (verify on all 9,445 distinct raw tokens):
`<Action>[(<Screen>)][<<Config>>][$<Chain>$][1]`
- `Action`: ~36 distinct French UI verbs (e.g. `Exécution d'un bouton` 421k occurrences,
  `Sélection d'un écran`, `Saisie dans un champ`, `Fermeture/Affichage d'une dialogue`,
  `Lancement d'une action générique`, `Sélection d’un onglet`, `Raccourci`, `Création d'un écran`,
  `Retour sur un écran`, `Chainage`, `Double-clic`, `Filtrage / Tri`, `Lancement d'une stat`,
  `Lancement d'un tableau de bord`, `Affichage d'une erreur`, `Fermeture de session`, `Clic long`,
  `Démarrage serveur d'application Tomcat`, …). 65 raw strings exist because many also appear with a trailing `1`.
- `(Screen)`: 418 distinct values; either a Java controller path
  (`infologic.core.accueil.AccueilController`, `infologic.crm.modules.CRM_PLANNING.planning.PlanningController`,
  `infologic.orga.modules.OR_FICHIERTRAVAIL.InstanceForm`, `fr.infologic.core.client...`) or a short
  code (`MAINT`, `DEV`, `MAJ`). The package path encodes **module/domain** (`crm`, `orga`, `compta`,
  `acti`…) → **hierarchical features** (each dotted prefix is a token).
- `<Config>`: 219 distinct screen configurations (`DEF_03/24`, `DEF_07/23`, `STD`, `INSTALLATEUR`,
  `TELEM_MINIMAL`, `001`, `MT_AppelN1`, `ACCUEIL_INST`…).
- `$Chain$`: 76 distinct "chaînes" of the record sheet (`VT`, `GP`, `EDI`, `TEC`, `OT`, `SV`, `GESCO`, `JCP`…).
- 29 tokens contain **more than one** parenthesis group → regex with `findall`, not `search`.
- **Strong hypothesis to test (high value):** the log appears to be *state-compressed*. The full
  context (`(Screen)<Config>$Chain$`) is written when it **changes**, then subsequent actions drop it,
  and the trailing `1` seems to mean "same context as previously stated". Example seen:
  `Saisie dans un champ(MAINT)<DEF_03/24>1`, `Saisie dans un champ1`, `Saisie dans un champ1`, …, later
  `Exécution d'un bouton$TEC$`, `Saisie dans un champ$TEC$`. → **Forward-fill the current
  screen/config/chain state onto every action** (state machine), then compute features on the filled
  stream (time spent per screen, #distinct screens, transitions, dwell). Investigate and document
  what `1` means (does it correlate with long sessions? with a given browser/version? does it
  reset after `Création d'un écran`?). Compare CV with raw tokens vs. forward-filled tokens.
- The course's `filter_action` helper is **buggy** (cuts at the first `1` anywhere, e.g. inside
  `DEF_03/24`; the walrus on `index` is falsy at position 0). Write a regex tokenizer instead and
  unit-test it on all distinct tokens (round-trip: reconstruct token from parsed fields).

**Duplicates / overlap:** 0 duplicated train lines, 0 test lines identical to a train line.

**Sanity baselines I ran (80/20 stratified-ish hold-out, seed 0, rough numbers, not CV):**

| Representation | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| TF-IDF, raw tokens unigrams (+browser) | LinearSVC C=1 | ≈ 0.90 | ≈ 0.86 |
| TF-IDF, raw tokens 1–3-grams (+browser), min_df=2 | LinearSVC C=1 | ≈ 0.93 | ≈ 0.90 |
| TF-IDF, base action only (context stripped) 1–3-grams | LinearSVC C=1 | ≈ 0.89 | ≈ 0.83 |

Take-aways: (1) the problem is very learnable; context (screen/config/chain) carries a lot of
signal; (2) n-grams (action *order*/habits) help; (3) **target to beat ≈ 0.93 accuracy** with a
properly validated pipeline. Reproduce these in `02_baselines` as the reference row of the results table.

---

## 3. Repository layout to generate

```
copilote-users/
├── AGENTS.md                 # section 0 rules (+ how to run)
├── PROGRESS.md               # living log: phase, status, next step, open questions
├── README.md                 # FR+EN: goal, setup, data placement, one-command run, structure, results
├── Makefile                  # setup, lint, test, eda, features, train, predict, report, all
├── pyproject.toml            # ruff + pytest + (optional) mypy config
├── requirements.txt          # pinned, works on 3.13 and 3.14 (see 4.)
├── requirements-optional.txt # xgboost, lightgbm, shap, torch ... (pipeline must run without them)
├── .gitignore                # data/, artifacts/, __pycache__, .venv, LaTeX aux
├── configs/
│   └── default.yaml          # seed, paths, CV scheme, feature switches, model grids
├── data/
│   ├── raw/                  # train.csv, test.csv (git-ignored)
│   └── README.md
├── src/copilote/
│   ├── __init__.py
│   ├── io.py                 # robust reader -> Sessions (ragged); stripping enclosing quotes
│   ├── tokens.py             # regex grammar, NFC, parse_token(), context forward-fill
│   ├── features/
│   │   ├── bow.py            # token / ngram / hierarchical-screen bag-of-words (sparse)
│   │   ├── stats.py          # length, duration, rates, entropy, ratios, idle gaps, diversity
│   │   ├── sequence.py       # transitions (Markov), first-k / last-k actions, repeats
│   │   └── builder.py        # sklearn Pipeline/ColumnTransformer; fit only on train folds
│   ├── models.py             # model zoo + param grids (single registry)
│   ├── cv.py                 # repeated stratified CV, OOF predictions, metrics
│   ├── ensemble.py           # soft-vote / stacking on OOF probabilities
│   ├── interpret.py          # RF importances, permutation importance, SHAP/LIME, redundancy, confidence
│   ├── plots.py              # all figures -> reports/figures (PDF/PNG, consistent style)
│   ├── predict.py            # fit final, write submission
│   └── run_all.py            # orchestrates phases, writes reports/results/*.json
├── notebooks/
│   └── BE_CS1_solution.ipynb # SUCCINCT narrative notebook calling src/copilote (outputs cleared or light)
├── tests/                    # pytest: io, tokens (round-trip), no-leakage, shapes, determinism, submission format
├── reports/
│   ├── figures/
│   ├── results/              # json/csv produced by code, the ONLY source of numbers for the report
│   └── latex/ main.tex, refs.bib, Makefile → main.pdf (5–8 pages)
├── submissions/              # submission_*.csv (+ metadata json with CV score, git hash, config)
└── docs/
    ├── DECISIONS.md
    ├── FEATURES.md           # catalogue of every feature: definition, rationale, expected effect
    ├── EXPERIMENTS.md        # table of every experiment run (id, change, CV mean±std, time, verdict)
    └── DEFENSE.md            # oral-defense prep for each team member (section 9)
```

---

## 4. Environment and dependencies

- Target Python 3.14; must also run on 3.13.5. Create a venv, install, run the tests **on the
  python available**; state which versions you actually tested.
- Core (pinned, with wheels on 3.13/3.14): `numpy`, `pandas`, `scipy`, `scikit-learn`,
  `matplotlib`, `seaborn`, `pyyaml`, `tqdm`, `joblib`, `pytest`, `ruff`.
- Optional (import-guarded, pipeline works without): `xgboost`, `lightgbm`, `shap`, `lime`,
  `torch`. If any lacks a 3.14 wheel, **do not** break the install: leave it in
  `requirements-optional.txt`, and gate code with `try/except ImportError` + a log line.
- Use `joblib`/`n_jobs` carefully (memory: sparse matrices only, never densify 14k×N).
- Cache features/OOF predictions in `artifacts/` (git-ignored) keyed by config hash.

---

## 5. Phase plan with gates

Work through the phases in order. Each phase ends with its **gate**; do not skip.

### Phase 1 — Setup & loading (BE session 1)
- Build the repo skeleton, tooling, `io.py`, `tokens.py`, tests.
- Load both files into ragged token lists; compute shapes; check the quote-wrapping quirk.
- **Gate:** `pytest` green; parser round-trips all distinct tokens; 3,279 train / 324 test sessions;
  247 classes; no NaN-padded wide frame anywhere.

### Phase 2 — EDA (BE session 1)
Produce, with short written interpretations (saved as markdown snippets reused in the report):
- distinct browsers; **browser-per-user table** (rows = user, cols = browser, counts;
  head+tail style like the course's `browsers_per_player`); how many users use >1 browser; is the
  browser informative alone (baseline accuracy)?
- **Distribution of Y** (class-size histogram, long tail, Gini/entropy, majority-class baseline
  accuracy & macro-F1 as the "dumb" reference).
- Session length distribution (linear + log), duration from `tNN`, outliers (IQR / log-z), train vs
  test **distribution shift** check (KS tests / adversarial validation train-vs-test classifier —
  AUC ≈ 0.5 expected; report it).
- Action vocabulary: top actions, screens, configs, chains; Zipf plot; modules by package prefix;
  vocabulary coverage of test by train (OOV rate of test tokens — important for feature design).
- The meaning of the trailing `1` and of context forward-filling (hypothesis in section 2).
- Per-user fingerprints: for 6–8 users show most distinctive screens/actions (log-odds).
- Duplicate/near-duplicate sessions inside a user (are sessions consecutive pieces of the same
  work day? any leakage-like structure such as row order? **Do not exploit row order** in train/test
  unless you can show it's legitimate; mention it in the report if found).
- **Gate:** `reports/results/eda.json` + figures exist; each figure has a 2–4 line interpretation.

### Phase 3 — Feature engineering (BE sessions 1–2)
Implement as sklearn-compatible transformers; document each in `docs/FEATURES.md`.

**A. Sparse "bag" features (the backbone — usually the best performers)**
- A1 token BoW/TF-IDF on **full raw tokens**; A2 on **base action**; A3 on **screen**; A4 on
  **config**; A5 on **chain**; A6 on **module hierarchy** (all dotted prefixes of controller path,
  e.g. `infologic`, `infologic.crm`, `infologic.crm.modules`, …); A7 **action×screen** and **action×chain**
  crosses; A8 on **forward-filled** context tokens.
- N-grams over the action sequence: 1–3 (try up to 4–5 with `min_df`/`max_features`/hashing to
  control size); skip-grams optional; **sublinear TF**, L2 norm; try binary vs counts.
- Optional **char n-grams** on token strings to be robust to small naming variants (compare in CV).

**B. Dense summary statistics (needed for trees/NN, also the interpretability story)**
- length (`log1p`), n_actions, n_time_markers, session duration, actions/min, mean/median/max/std
  idle gap, share of gaps > 30 s / 60 s / 5 min, burstiness, longest idle.
- per-action-type proportions (the ~36 verbs), error rate (`Affichage d'une erreur`), dialog/toast rate,
  shortcut rate (`Raccourci`), double-click rate, `Saisie`/`Exécution` ratio, Shannon entropy of
  actions, entropy of screens, #distinct screens/configs/chains, repeat-rate (immediate repetition),
  longest run of the same action, `Filtrage / Tri` & `Lancement d'une stat/tableau de bord` counts.
- start/end behaviour: first-k and last-k actions (k=3,5,10) as categorical/BoW; ends with
  `Fermeture de session`? contains `Démarrage serveur…`? (rare but potentially user-specific).
- browser (OHE for linear/NN/SVM; ordinal or OHE for trees — compare), and "browser consistency".
- trailing-`1` rate, share of tokens with explicit context, #context switches.
- "most used screen / config / chain" (regex per the course) as both categorical (top-N + OTHER)
  and as target-agnostic frequency features.
- **Target encoding** is *not* appropriate here (247 classes) — don't.

**C. Sequence/behaviour models**
- First-order Markov **transition matrix** between base actions (flattened, normalised) and between
  modules; compute per session; reduce with SVD if wide.
- Optional: per-class n-gram language-model log-likelihood scores as features (Naive-Bayes-style,
  computed out-of-fold to avoid leakage).

**D. Dimensionality / selection (justify in report)**
- `min_df`, `max_df`, chi²/mutual-information top-k **inside folds**, L1-penalised selection,
  TruncatedSVD (LSA) for MLP/kNN. Report feature counts per family.

**E. Preprocessing decisions to document:** outliers (keep vs cap vs drop, evidence from CV),
scaling (needed for SVM/LR/MLP/kNN, not trees), class encoding (`pd.Categorical` codes **and** the
inverse mapping saved), browser encoding, handling of OOV tokens, rare-class handling.

- **Gate:** feature builder is a single sklearn `Pipeline` fit on a fold and transforming another
  with no leakage (a test asserts that `fit` on fold A never reads fold B); feature-family ablation
  table started in `EXPERIMENTS.md`.

### Phase 4 — Statistics & visualisation of features (BE session 2)
- ≥ 3 feature plots side by side (`sharey`), e.g. `actions_means`-like feature, duration,
  error-rate; **boxplots/violin plots of one feature for chosen classes** (`[0, 15, 152, 199]` —
  note codes depend on `pd.Categorical` ordering; plot by user name and give both); interpret
  separability honestly.
- Correlation heatmap (Spearman) among dense features; hierarchical clustering → redundancy groups.
- 2-D projection (TruncatedSVD/UMAP-free: PCA/t-SNE on TF-IDF-SVD) coloured by the top-10 users.
- Univariate tests: ANOVA/Kruskal per dense feature vs user (effect size η²), ranked.
- **Gate:** figures saved; each with interpretation text.

### Phase 5 — Modelling (BE sessions 2–3)
Protocol (identical for every model so results are comparable):
- **Primary CV:** `RepeatedStratifiedKFold` (k=4 because smallest class = 4; ≥3 repeats; fixed seeds),
  features rebuilt per fold. **Also** the course-style single stratified train/validation split
  (80/20) as the simple, readable number for the report.
- Metrics, all computed on OOF predictions **and** per fold (mean ± std): accuracy, **macro-F1**,
  weighted-F1, balanced accuracy, top-3 / top-5 accuracy, multi-class log-loss, plus per-class
  precision/recall tables and the confusion pairs most mixed up. Say which metric the Kaggle page
  uses (section 12) and optimise that one; report accuracy + F1 regardless.
- Baselines to include: majority class; browser-only; stratified-random.

Model zoo (registry in `models.py`, grids in `configs/default.yaml`), tune with nested or at
least CV-disjoint search (`RandomizedSearchCV`/Optuna if available; budget-aware):
1. **Linear on sparse TF-IDF:** `LinearSVC` (C grid), `LogisticRegression` (saga, L2 / L1 / ElasticNet;
   multinomial), `SGDClassifier` (hinge/log/modified_huber), `RidgeClassifier`,
   `ComplementNB`/`MultinomialNB`. **Expect these to be the strongest/most efficient.**
2. **SVM kernels:** linear, poly (degree 2–3), RBF on SVD-reduced or L2-normalised features;
   `CalibratedClassifierCV` wrapper to get probabilities.
3. **Trees:** `DecisionTreeClassifier` (depth/min-samples grid, show a pruned tree for
   interpretability), `RandomForest` and `ExtraTrees` on dense stats + selected sparse columns,
   `HistGradientBoosting` (sklearn) and XGBoost/LightGBM if available (247 classes → mind runtime;
   use feature subset, `max_depth` small, early stopping on a validation fold).
4. **MLP:** `MLPClassifier` on SVD(200–500)+stats, scaled; layers {1,2}×{128,256,512}, dropout-like
   regularisation via `alpha`, early stopping. Optional small PyTorch model (embedding + bag of
   tokens / 1-D CNN / GRU over the action sequence) **only if** it adds CV gain; document the cost.
5. **kNN / cosine similarity** to training sessions on TF-IDF (diverse, cheap, good with n-grams).
6. **Regularisation / sparsity:** L1 & elastic-net comparison; report #non-zero coefficients.
- Learning curve (accuracy vs #training sessions per class) and an ablation by **feature family**
  (drop-one and add-one) — this feeds the "variables les plus influentes" section.
- **Ensembling:** soft-vote of the best 3–5 *diverse* models (weights chosen on OOF), then
  **stacking** (OOF probabilities → multinomial LR meta-learner, nested so the meta-learner is
  never trained on in-fold predictions). Accept the ensemble only if CV gain > 1 std.
- **Imbalance:** compare `class_weight="balanced"` vs none; threshold-free (argmax) predictions;
  note effect on macro-F1.
- **Calibration & confidence:** reliability curve, ECE; "abstain" analysis (accuracy vs coverage)
  — useful for the discussion even if the Kaggle metric forces a hard label.
- **Gate:** `reports/results/model_comparison.csv` (every model: params, CV mean±std for all
  metrics, fit time), best single model, best ensemble; `EXPERIMENTS.md` complete; statistically
  sensible choice (don't pick the best by 0.1 pt noise — prefer simpler within 1 std).

### Phase 6 — Interpretability (BE session 3, graded in discussion)
Answer the **four course questions** with figures and text, on the RF *and* on the best model:
1. **Confidence of a prediction:** RF tree-vote variance / `predict_proba` margin; calibrated
   probabilities for the linear model; show 3 examples (confident-correct, uncertain-correct,
   confident-wrong).
2. **Local explanation of one row:** SHAP (TreeExplainer on RF/GBM; linear contributions
   coef×tfidf for the linear model) and/or LIME on text-like input; show top ± contributing
   tokens/features for the example rows, in plain language ("this user always opens screen X then …").
3. **Strongest predictors globally:** impurity importance (note its bias), **permutation importance**
   (grouped by feature family to handle correlated features), top coefficients per class for a few
   users, ablation results.
4. **Redundant columns:** Spearman-correlation clustering + dendrogram, drop-column CV test,
   VIF-like or cluster-representative selection; report how many features can be removed at ≤ 0.5-pt loss.
- **Error analysis:** which users are confused with which (confusion pairs), whether confusions are
  explained by shared roles/modules, short sessions vs long sessions accuracy, browser effect.
- **Gate:** `reports/results/interpretability.json` + figures + 1 paragraph per question.

### Phase 7 — Final model & submission
- Retrain the chosen pipeline on **100 % of train** with the chosen hyper-parameters, predict test.
- Optional transductive step (7.6): high-confidence **pseudo-labelling** of test sessions added to
  training, accepted only if repeated-CV simulation (treat a CV fold as "test") shows a gain.
- Write `submissions/submission_<timestamp>_<model>.csv` in the **exact format required** (section 12),
  with a sidecar `.json` (CV metrics, config hash, git commit). Validate with a test: row count 324,
  labels ⊂ train classes, no NaN, correct header, correct ids.
- Keep ≤ 3 candidate submissions (best single, best ensemble, safest/simplest). Choose the final
  one by **CV**, not by the public leaderboard (public/private shift risk with only 324 rows).
- **Gate:** submission validator passes; README documents the exact command that regenerates it.

### Phase 8 — Report (PDF, 5–8 pages, French, LaTeX)
Use `reports/latex/` (LaTeX; `latexmk` build). Template: A4, 11 pt, 2 cm margins, figures vector (PDF).
**Page count must be verified after compile (≥ 5 and ≤ 8, excluding nothing)** — add a check
script (`pdfinfo`) in the Makefile target that fails otherwise. All tables auto-generated from
`reports/results/*.csv` (via `pandas.to_latex` or a small script) to avoid copy errors.

Outline (target pages in brackets):
1. **Introduction** (0.5): context Copilote/Infologic, task, why it matters, challenge & metric, team repo URL.
2. **Données et analyse exploratoire** (1.0–1.5): format, ragged sessions, grammar of tokens
   (action/écran/config/chaîne, the `1` suffix and context forward-fill, `tNN`), class imbalance,
   length/outliers, train–test shift. 2–3 figures max.
3. **Construction des caractéristiques** (1.5): families A–E, **why** each (behavioural fingerprint
   idea: users differ in modules used, habits, speed, error behaviour), sizes, preprocessing &
   outlier choice, leakage control. One summary table of feature families.
4. **Méthodes de classification** (1.0): models tried, hyper-parameters searched, CV protocol,
   metrics defined (accuracy, macro-F1, … and the Kaggle metric), ensembling.
5. **Résultats** (1.5): comparison table (mean ± std), ablation by feature family, learning curve,
   confusion analysis, final model and its leaderboard score (filled by human after upload).
6. **Interprétabilité et discussion** (1.0–1.5): the four questions, most influential variables,
   redundancy, confidence/calibration, limits (few sessions per user, drift in time, 324-row test
   noise, privacy/GDPR aspect of identifying users from traces, what an editor like Infologic could do with this).
7. **Conclusion** (0.3): findings, what did not work, next steps.
- Bibliographic items (refs.bib, cite only what is used): scikit-learn (Pedregosa et al. 2011),
  TF-IDF/BoW text-classification basics, SHAP (Lundberg & Lee 2017) if used, XGBoost/LightGBM if used.
- **Gate:** PDF compiles, 5–8 pages, every number traceable to `reports/results/`, no TODO left.

---

## 6. Code quality standards

- Type hints, docstrings (Google or NumPy style), small pure functions, no magic numbers (config).
- `ruff` clean (`ruff check` + `ruff format`), `pytest` ≥ 80 % coverage of `io`, `tokens`, `features`, `cv`, `predict`.
- Logging via `logging` (no stray prints in library code); progress bars for long loops.
- Every script has `--config` and `--seed` CLI args; every random operation takes an explicit seed.
- Figures: one shared style (`plots.py`), readable fonts at report size, colour-blind-safe palette,
  axis labels with units, saved as PDF for LaTeX + PNG for notebook.
- The **notebook** is succinct and linear (sections named exactly as the course pipeline), imports
  from `src/copilote`, shows head/tail tables and key plots, with a *Votre analyse ici* answered in
  2–5 sentences each. It must run top-to-bottom in a fresh kernel in a few minutes using cached artifacts.
- Memory/time: full `make all` must run in a reasonable time on a laptop (target < 30–45 min;
  expensive searches cached; document the "fast" mode `make fast` ≈ 5 min that skips big grids).

---

## 7. Tips for getting the best score (try in this priority order, keep what wins in CV)

1. **Strong sparse baseline first** (TF-IDF 1–3-gram on forward-filled tokens + browser → LinearSVC / LR). Tune `C`, `min_df`, n-gram range, `sublinear_tf`, `class_weight`.
2. **Multi-view concatenation:** hstack of separately-weighted TF-IDF blocks (tokens, screens,
   modules hierarchy, chains, configs, transitions); tune block weights on CV.
3. **Length-robustness:** L2-normalised TF-IDF already handles long vs short; also test truncating
   very long sessions to first/last N tokens or splitting them into windows (train-time augmentation,
   test-time averaging of window probabilities) — long sessions are very informative, short ones
   (13–100 tokens) are the hard cases: **report accuracy by length bucket.**
4. **Session-start fingerprint:** the first 5–20 actions (login flow, landing screen) are often
   user/role-specific → dedicated features/BoW.
5. **Data augmentation** for small classes (4–10 sessions): window-sampling sub-sessions from long
   sessions; verify it doesn't leak across CV folds (augment inside the training fold only).
6. **Probabilistic ensemble** of linear TF-IDF model + RF/GBM on dense stats + kNN-cosine +
   (optional) MLP/sequence NN; stack on OOF probabilities.
7. **Transductive options** (only if CV-validated): pseudo-labelling confident test rows; refit
   vocabulary including test tokens (unsupervised, no labels — acceptable, document it).
8. Don't overfit the 324-row public test: pick by repeated-CV mean, prefer the simplest model within one std.

---

## 8. Things that will silently ruin the project (avoid)

- Fitting TF-IDF / scaler / feature selection on all train before CV (leakage → inflated scores).
- Splitting sessions of the same user's very similar consecutive work across folds is *legitimate*
  here (test is also random sessions), but make sure you never tune on test.
- Densifying the sparse matrix; building the 14,470-column frame from the course snippet.
- Splitting the CSV line on commas **after** the tokens were quoted or containing commas (check:
  all 9,445 tokens must parse; none should contain a stray comma).
- Dropping long sessions then being unable to predict long test sessions.
- Using `pd.Categorical` codes inconsistently between train/val/test or losing the inverse mapping
  for the submission.
- Reporting accuracy only (classes are imbalanced: also macro-F1) or reporting a single lucky split.
- Final report claiming "best model X" without the fold-by-fold variance.
- Python 3.14 incompatibility from a pinned dependency discovered at the last minute → test early.

---

## 9. `docs/DEFENSE.md` (oral defence preparation)

The teacher questions each member during BE session 3 and may ask anyone about anything.
Generate for each pipeline stage a half-page: *what it does · why this choice · what alternative
was tried · what the numbers showed · a likely teacher question and a model answer*. Include:
tokens grammar; why TF-IDF + linear SVM works on this data; why macro-F1 not only accuracy; CV
protocol & leakage; what outlier handling you chose and why; how the RF answers the four
interpretability questions; what the most influential features are and why that makes behavioural
sense; limits and ethics (re-identification of users from usage logs). Also a one-page
"30-second / 3-minute / 10-minute" pitch of the project. Skill target: *Résoudre et arbitrer*
(C2C2) — show that you identified the most relevant methods, applied them, **and chose pertinent
evaluation criteria**.

---

## 10. Definition of done

- [ ] `make all` rebuilds features → CV results → figures → submission → PDF from raw CSVs.
- [ ] `pytest` green, `ruff` clean, fresh-venv install works on the available Python (3.13 verified; 3.14 verified or issue documented).
- [ ] Submission file validated; format matches section 12 answers.
- [ ] `reports/latex/main.pdf` has 5–8 pages, French, all numbers from artifacts, all figures readable.
- [ ] README explains setup in < 10 lines; no data in git; archive builds without data (`make archive`).
- [ ] `docs/FEATURES.md`, `EXPERIMENTS.md`, `DECISIONS.md`, `DEFENSE.md` complete.
- [ ] Notebook runs top-to-bottom, succinct, each "Votre analyse ici" answered.
- [ ] Final CV number reported with std; baselines (majority, browser-only, TF-IDF unigram) in the same table.

---

## 11. `PROGRESS.md` format (update after every phase / session)

```
## Status  (phase N/8, date)
## Done (checklist with commit hashes)
## Current best (model, features, CV acc ± std, CV macro-F1 ± std, submission file)
## Next 3 actions
## Open questions for the human
## Known issues / risks
```

---

## 12. Things you cannot know — ask the human *once*, up front, in a single message (then proceed with the stated defaults)

1. **Kaggle evaluation metric and submission format.** No `sample_submission.csv` was provided.
   Default if unanswered: CSV with header `Id,util` (`Id` = 0-based test line index, `util` = user
   code, e.g. `skm`), and optimise **macro-F1 while tracking accuracy**. Make the id base (0/1) and
   column names **configurable** and add a clearly marked `TODO(human): verify against the Kaggle "Data → sample submission / Evaluation" tab`.
2. Are the real files quote-wrapped (as in my copies) or not? (Parser supports both.)
3. Python version available to the grader (3.14 stated, "à priori") and whether GPU is available (not required).
4. Team members' names / repo URL / group (TD 1–4) for the report header and the Kaggle team link.
5. Whether the human wants the report in French (default **yes**) and the notebook as an extra deliverable (default **yes**).

**Human-only tasks (list them at the end of the run, do not attempt):**
upload the final CSV on the correct Kaggle group page and on Moodle; upload the PDF and the code
archive/URL on Moodle (deadline 9 Nov 2026 evening); read `DEFENSE.md` and make sure each
member can explain every part.

---

## 13. Kick-off instruction

Start now with **Phase 1**. First, reply with (a) a 10-line plan confirming you understood the
task, (b) the questions of section 12 in one message. Without waiting for the answer, create the
repository skeleton, the environment and the parser/tokenizer with tests, then run EDA. Report
at each gate with the numbers produced. Prefer correctness, traceability and a strong,
well-validated linear/ensemble solution over exotic models.